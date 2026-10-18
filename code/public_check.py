"""
Public-data check of the revenue-curve primitives.

Inputs (downloaded by `download_public.sh`):
  data/raw/partd_2024.csv   CMS Medicare Part D Spending by Drug, 2020-2024
  data/raw/partb_2024.csv   CMS Medicare Part B Spending by Drug, 2020-2024
  data/raw/drugsfda.json.zip  openFDA Drugs@FDA (application type NDA / BLA,
                              original approval date, brand names)

Outputs:
  data/public_panel.csv              brand x year Medicare spending with modality and age
  output/public_profiles.csv         mean spending per brand by modality and age
  output/public_check.json           NLS shape estimates, tail shares, level ratio
"""
import json, re, zipfile, pathlib
import numpy as np, pandas as pd
from scipy import optimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)

def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper().replace("*", ""))

# ---- Drugs@FDA: brand -> modality, first approval year ----------------------
z = zipfile.ZipFile(RAW / "drugsfda.json.zip")
apps = json.load(z.open(z.namelist()[0]))["results"]
rows = []
for app in apps:
    an = app.get("application_number", "")
    typ = "BLA" if an.startswith("BLA") else ("NDA" if an.startswith("NDA") else None)
    if typ is None:
        continue
    origs = [s["submission_status_date"] for s in app.get("submissions", [])
             if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP"
             and s.get("submission_status_date")]
    if not origs:
        continue
    yr = int(min(origs)[:4])
    for prod in app.get("products", []):
        bn = prod.get("brand_name", "").strip().upper()
        if bn:
            rows.append(dict(key=norm(bn), type=typ, approved=yr))
fda = (pd.DataFrame(rows).drop_duplicates()
       .groupby("key").agg(approved=("approved", "min"),
                           type=("type", lambda s: s.mode().iloc[0])).reset_index())

# ---- CMS spending by drug (Part D 'Overall' manufacturer rows + Part B) ------
pd_ = pd.read_csv(RAW / "partd_2024.csv")
pd_ = pd_[pd_.Mftr_Name == "Overall"].copy()
pb = pd.read_csv(RAW / "partb_2024.csv")
def long(df, prog):
    out = []
    for y in range(2020, 2025):
        t = df[["Brnd_Name", f"Tot_Spndng_{y}"]].rename(columns={f"Tot_Spndng_{y}": "spend"})
        t["year"] = y; t["prog"] = prog; out.append(t)
    return pd.concat(out)
L = pd.concat([long(pd_, "D"), long(pb, "B")])
L["key"] = L.Brnd_Name.map(norm)
L = L.groupby(["key", "year"], as_index=False).spend.sum()
m = L.merge(fda, on="key", how="inner")
m["age"] = m.year - m.approved
match_share = {int(y): float(m[m.year == y].spend.sum() / L[L.year == y].spend.sum()) for y in range(2020, 2025)}
m = m[m.age >= 0].copy()
(ROOT / "data").mkdir(exist_ok=True)
m.to_csv(ROOT / "data" / "public_panel.csv", index=False)

prof = m.groupby(["type", "age"]).spend.agg(["sum", "count"]).reset_index()
prof["mean"] = prof["sum"] / prof["count"]
prof.to_csv(OUT / "public_profiles.csv", index=False)

# ---- NLS fits to the public age profile --------------------------------------
# The paper's shape g(t) S(t) = exp(-(t-t*)^2/2w^2) exp(-h t) is itself a Gaussian
# with mean t* - h w^2, so the spending profile alone identifies only the composite
# peak and the width.  We therefore report (a) the composite Gaussian fit and
# (b) the fit with the erosion hazard fixed at the Kaplan-Meier medians used in
# the paper, which recovers t* and w on the same footing as the Kythera estimates.
KM_MEDIAN = {"NDA": 14.4, "BLA": 21.5}
RHO = 0.08

def gauss(t, A, mu, w):
    return A * np.exp(-0.5 * ((t - mu) / w) ** 2)

def shape_fixed_M(M):
    return lambda t, A, tstar, w: A * np.exp(-0.5 * ((t - tstar) / w) ** 2) * np.exp(-np.log(2) / M * t)

res = {"match_share_of_spending": match_share, "brands_matched": int(m.key.nunique())}
tt = np.linspace(0, 80, 8001)
for typ in ["NDA", "BLA"]:
    d = prof[(prof.type == typ) & (prof.age <= 25)]
    t = d.age.values.astype(float); y = d["mean"].values / 1e6
    wts = 1 / np.sqrt(d["count"].values)
    pg, cg = optimize.curve_fit(gauss, t, y, p0=[y.max(), 9, 4], sigma=wts, maxfev=20000)
    pf, cf = optimize.curve_fit(shape_fixed_M(KM_MEDIAN[typ]), t, y, p0=[y.max(), 9, 4], sigma=wts, maxfev=20000)
    seg, sef = np.sqrt(np.diag(cg)), np.sqrt(np.diag(cf))
    f = shape_fixed_M(KM_MEDIAN[typ])(tt, *pf) * np.exp(-RHO * tt)
    tails = {str(c): float(np.trapezoid(f[tt >= c], tt[tt >= c]) / np.trapezoid(f, tt)) for c in (9, 11, 13)}
    raw_tail = {str(c): float(d[d.age >= c]["mean"].sum() / d["mean"].sum()) for c in (9, 11, 13)}
    res[typ] = dict(
        composite=dict(peak=float(pg[1]), peak_se=float(seg[1]), w=float(pg[2]), w_se=float(seg[2])),
        fixed_hazard=dict(M=KM_MEDIAN[typ], tstar=float(pf[1]), tstar_se=float(sef[1]), w=float(pf[2]), w_se=float(sef[2])),
        tail_share_discounted_fitted=tails, tail_share_raw_means=raw_tail,
        mean_spend_per_brand_0_20=float(d[d.age <= 20]["sum"].sum() / d[d.age <= 20]["count"].sum() / 1e6),
        n_brand_years=int(d["count"].sum()), n_brands=int(m[m.type == typ].key.nunique()))
res["lambda_level_ratio"] = res["BLA"]["mean_spend_per_brand_0_20"] / res["NDA"]["mean_spend_per_brand_0_20"]
json.dump(res, open(OUT / "public_check.json", "w"), indent=2)
print(json.dumps(res, indent=2))
