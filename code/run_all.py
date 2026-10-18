"""
Reproduce all tables and figures in the paper.

    python code/public_check.py      # builds output/public_check.json (needs data/raw)
    python code/run_all.py           # builds output/*.csv, output/*.tex, output/fig*.pdf, output/results.json

Runtime: about ten seconds (2,000 bootstrap draws).
"""
import json, pathlib, sys
from dataclasses import replace

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import (BASE, POLICIES, MODALITIES, Primitives, evaluate, evaluate_all,
                   tail_share, flow_density, eta_breakeven_fast, eta_sign_flip_fast,
                   iso_savings_frontier, bootstrap_fast, annuity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
pub = json.load(open(OUT / "public_check.json"))

# Baseline: Kythera shapes, public-data level ratio
LAM = round(pub["lambda_level_ratio"], 1)
P = replace(BASE, lam=LAM)
R = {}

# ---------------------------------------------------------------- Table: tails
tails = {}
for m in MODALITIES:
    tails[m] = {c: tail_share(P, m, c) for c in (9.0, 11.0, 13.0)}
R["tail_shares_kythera"] = tails
R["lambda"] = LAM

# ---------------------------------------------------------------- Table III: policies
res = evaluate_all(P)
rows = []
for k, o in res.items():
    rows.append(dict(policy=k, tau_sm=POLICIES[k]["tau"]["SM"], tau_bio=POLICIES[k]["tau"]["BIO"],
                     missing=o.missing_nmes, missing_sm=o.missing_by_mod["SM"], missing_bio=o.missing_by_mod["BIO"],
                     savings=o.savings, share_sm=o.savings_share_by_mod["SM"], innovation_loss=o.innovation_loss,
                     patient_loss=o.patient_loss, net=o.net_welfare, nmes_per_100b=o.nmes_per_100b))
T3 = pd.DataFrame(rows); T3.to_csv(OUT / "table_policies.csv", index=False)
R["policies"] = T3.set_index("policy").to_dict("index")
R["eta_breakeven_cl_epic"] = eta_breakeven_fast(res, P)
R["eta_sign_flip_cl"] = eta_sign_flip_fast(res, P)

# ---------------------------------------------------------------- Innovation price of savings by modality
# NMEs forgone per $100B extracted from each modality under current law
o = res["Current law"]
price = {}
for m in MODALITIES:
    dollars = o.savings * o.savings_share_by_mod[m]
    price[m] = 100 * o.missing_by_mod[m] / dollars
R["innovation_price_by_modality"] = price

# ---------------------------------------------------------------- Iso-savings frontier
front, best = iso_savings_frontier(P, res["Current law"].savings)
pd.DataFrame(front, columns=["tau_sm", "tau_bio", "missing", "net"]).to_csv(OUT / "iso_savings_frontier.csv", index=False)
R["iso_savings_best"] = dict(tau_sm=best[0], tau_bio=best[1], missing=best[2], net=best[3])
# the same for a frontier restricted to tau_SM <= 13
sub = front[front[:, 0] <= 13.0]
b13 = sub[np.argmin(sub[:, 2])]
R["iso_savings_best_tau_sm_le_13"] = dict(tau_sm=b13[0], tau_bio=b13[1], missing=b13[2], net=b13[3])

# ---------------------------------------------------------------- Sensitivity grid
def run(p, label):
    r = evaluate_all(p)
    cl, c11, rev, epic = r["Current law"], r["Converge at year 11"], r["Reversed clocks (13, 9)"], r["EPIC (both year 13)"]
    return dict(case=label, cl_missing=cl.missing_nmes, cl_savings=cl.savings, cl_net=cl.net_welfare,
                c11_missing=c11.missing_nmes, c11_savings=c11.savings, c11_net=c11.net_welfare,
                rev_missing=rev.missing_nmes, rev_savings=rev.savings, rev_net=rev.net_welfare,
                epic_net=epic.net_welfare,
                cl_dominated=bool(rev.savings >= cl.savings and rev.missing_nmes <= cl.missing_nmes),
                eta_star=eta_breakeven_fast(r, p), eta_flip=eta_sign_flip_fast(r, p))

pub_shapes = dict(tstar={"SM": pub["NDA"]["fixed_hazard"]["tstar"], "BIO": pub["BLA"]["fixed_hazard"]["tstar"]},
                  width={"SM": pub["NDA"]["fixed_hazard"]["w"], "BIO": pub["BLA"]["fixed_hazard"]["w"]})
cases = [
    (P, "Baseline"),
    (replace(P, eta=0.10), "Elasticity eta = 0.10"),
    (replace(P, eta=0.50), "Elasticity eta = 0.50"),
    (replace(P, lam=1.0), "Equal per-drug revenue (lambda = 1)"),
    (replace(P, lam=2.0), "lambda = 2.0"),
    (replace(P, omega=1.0), "Producer surplus weighted equally (omega = 1)"),
    (replace(P, r_social=0.02), "Social discount rate 2 percent"),
    (replace(P, r_social=0.05), "Social discount rate 5 percent"),
    (replace(P, rho=0.05), "Private discount rate 5 percent"),
    (replace(P, rho=0.11), "Private discount rate 11 percent"),
    (replace(P, value_per_nme=5.0), "Value per NME $5B"),
    (replace(P, value_per_nme=15.0), "Value per NME $15B"),
    (replace(P, phi=0.22), "Negotiated discount phi = 0.22 (cycle 1 net)"),
    (replace(P, phi=0.50), "phi = 0.50"),
    (replace(P, theta=0.25), "Medicare share theta = 0.25"),
    (replace(P, theta=0.40), "theta = 0.40"),
    (replace(P, horizon=20), "Twenty-year horizon"),
    (replace(P, **pub_shapes), "Revenue shapes from CMS public files"),
    (replace(P, patient_loss_cl=18.7), "Patient loss at IV estimate with weight 2"),
    (replace(P, value_ratio_bio=LAM), "Value per NME proportional to per-drug revenue"),
    (replace(P, value_ratio_bio=1.0 / LAM), "Value per NME inversely proportional to per-drug revenue"),
]
S = pd.DataFrame([run(p, l) for p, l in cases]); S.to_csv(OUT / "table_sensitivity.csv", index=False)
R["sensitivity"] = S.to_dict("records")

# ---------------------------------------------------------------- Bootstrap
boot, etas, flips = bootstrap_fast(P, n=2000)
B = {}
for k, arr in boot.items():
    if k == "Repeal":
        continue
    B[k] = dict(missing=(arr[:, 0].mean(), arr[:, 0].std(), *np.percentile(arr[:, 0], [2.5, 97.5])),
                savings=(arr[:, 1].mean(), arr[:, 1].std(), *np.percentile(arr[:, 1], [2.5, 97.5])),
                net=(arr[:, 3].mean(), arr[:, 3].std(), *np.percentile(arr[:, 3], [2.5, 97.5])))
R["bootstrap"] = B
R["eta_star_boot"] = dict(mean=float(etas.mean()), sd=float(etas.std()), lo=float(np.percentile(etas, 2.5)), hi=float(np.percentile(etas, 97.5)))
R["eta_flip_boot"] = dict(mean=float(flips.mean()), sd=float(flips.std()), lo=float(np.percentile(flips, 2.5)), hi=float(np.percentile(flips, 97.5)))
# probability that reversed clocks dominate current law across draws
rev, cl = boot["Reversed clocks (13, 9)"], boot["Current law"]
R["prob_reversed_dominates"] = float(np.mean((rev[:, 1] >= cl[:, 1]) & (rev[:, 0] <= cl[:, 0])))
c11 = boot["Converge at year 11"]
R["prob_c11_higher_welfare"] = float(np.mean(c11[:, 3] > cl[:, 3]))
pd.DataFrame({k: v[:, 3] for k, v in boot.items()}).to_csv(OUT / "bootstrap_net_welfare.csv", index=False)

# ---------------------------------------------------------------- Figures
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
# Figure 1: discounted revenue densities and clocks, model vs public data
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
prof = pd.read_csv(OUT / "public_profiles.csv")
t = np.linspace(0, 30, 601)
for ax, m, typ, clock, title in [(axes[0], "SM", "NDA", 9, "Small molecules (NDA)"), (axes[1], "BIO", "BLA", 13, "Biologics (BLA)")]:
    f = np.array([flow_density(P, m, x, discount=0.0) for x in t]); f = f / f.max()
    d = prof[(prof.type == typ) & (prof.age <= 25)]
    ax.plot(t, f, "k-", label="Model revenue shape (Kythera primitives)")
    ax.plot(d.age, d["mean"] / d["mean"].max(), "o", mfc="none", color="gray", label="CMS Medicare spending per brand, 2020-2024")
    ax.axvline(clock, color="k", ls="--", lw=0.8); ax.text(clock + 0.3, 0.6 if m == "SM" else 0.95, f"statutory clock = {clock}", fontsize=8)
    ax.axvline(13 if m == "SM" else 9, color="gray", ls=":", lw=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Years since FDA approval"); ax.set_ylim(0, 1.08)
axes[0].set_ylabel("Revenue, share of peak")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT / "fig1_revenue_shapes.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig1_revenue_shapes.png", dpi=200); plt.close(fig)

# Figure 2: savings vs missing NMEs for all clock pairs, with named policies
fig, ax = plt.subplots(figsize=(6, 4.2))
grid = np.arange(5, 20.1, 1.0)
pts = []
for a in grid:
    for b in grid:
        oo = evaluate(P, "x", dict(tau={"SM": float(a), "BIO": float(b)}))
        pts.append((a, b, oo.savings, oo.missing_nmes))
pts = np.array(pts)
sc = ax.scatter(pts[:, 2], pts[:, 3], c=pts[:, 0] - pts[:, 1], cmap="coolwarm", s=12, alpha=0.8)
cb = plt.colorbar(sc, ax=ax); cb.set_label(r"$\tau_{SM} - \tau_{BIO}$ (years)")
for k, mk, off in [("Current law", "s", (-62, 10)), ("EPIC (both year 13)", "^", (8, -10)), ("Converge at year 11", "D", (-95, -16)), ("Reversed clocks (13, 9)", "o", (8, -4))]:
    oo = res[k]; ax.plot(oo.savings, oo.missing_nmes, mk, color="k", ms=8, mfc="white"); ax.annotate(k, (oo.savings, oo.missing_nmes), textcoords="offset points", xytext=off, fontsize=8)
ax.set_xlim(-10, 900); ax.set_ylim(-0.5, 26)
ax.set_xlabel("Present value of Medicare savings, $ billions (3 percent, 30 years)")
ax.set_ylabel("Forgone new molecular entities over 30 years")
fig.tight_layout(); fig.savefig(OUT / "fig2_savings_vs_innovation.pdf"); fig.savefig(OUT / "fig2_savings_vs_innovation.png", dpi=200); plt.close(fig)

# Figure 3: bootstrap distribution of eta*
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.hist(etas, bins=50, color="lightgray", edgecolor="gray")
ax.axvline(P.eta, color="k", ls="--"); ax.text(P.eta + 0.01, ax.get_ylim()[1] * 0.9, r"$\hat\eta=0.25$", fontsize=9)
ax.axvline(0.50, color="k", ls=":"); ax.text(0.51, ax.get_ylim()[1] * 0.8, r"$\eta=0.50$", fontsize=9)
ax.set_xlabel(r"Break-even elasticity $\eta^*$ (current law vs. EPIC)"); ax.set_ylabel("Bootstrap draws")
fig.tight_layout(); fig.savefig(OUT / "fig3_eta_star.pdf"); fig.savefig(OUT / "fig3_eta_star.png", dpi=200); plt.close(fig)

json.dump(R, open(OUT / "results.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("sensitivity", "bootstrap", "policies")}, indent=2, default=float))
print(T3.round(2).to_string(index=False))
print(S.round(2).to_string(index=False))
