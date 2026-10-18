# Medicare Drug Price Negotiation and Pharmaceutical Innovation

Replication package for Onur Baser, "Medicare Drug Price Negotiation and
Pharmaceutical Innovation" (Department of Economics, Boğaziçi University,
October 2026). The paper studies the Inflation Reduction Act's modality-specific
negotiation clocks (nine years for small molecules, thirteen for biologics) as a
tax on tail revenue, and shows that the current assignment buys Medicare savings
with more forgone innovation than a budget-equivalent reassignment would.

The manuscript itself is available from the author (onur.baser@bogazici.edu.tr).

## What is reproducible here

Every number in Tables I to V and Figures 1 to 3 that is a function of the model
primitives is produced by `code/run_all.py`. The public-data check (Table II,
the level ratio λ, and the circles in Figure 1) is produced by
`code/public_check.py` from CMS and FDA files that `code/download_public.sh`
downloads (copies are included under `data/raw/`).

The primitives estimated on the restricted Kythera Medicare closed-claims extract
(peak years, widths, exclusivity medians, Medicare share, negotiated price
reduction, the SMM elasticity, and the patient-access loss) enter as numbers
with their reported standard errors in `code/model.py` (`Primitives` and `SE`).
They cannot be re-estimated from this repository. See `online_appendix/README.md`.

## Layout

```
code/
  model.py            life-cycle model, policy evaluation, break-even
                      elasticities, iso-savings design, parametric bootstrap
  public_check.py     CMS Part B/D spending by drug + Drugs@FDA -> Table II
  run_all.py          all tables, figures and output/results.json
  download_public.sh  fetches the public inputs into data/raw/
data/
  raw/                CMS spending files (2020-2024) and openFDA Drugs@FDA
  public_panel.csv    brand x year Medicare spending with modality and age
output/
  table_policies.csv, table_sensitivity.csv, bootstrap_net_welfare.csv,
  iso_savings_frontier.csv, public_profiles.csv, public_check.json,
  results.json, fig1_revenue_shapes.*, fig2_savings_vs_innovation.*, fig3_eta_star.*
online_appendix/
```

## Run

```
pip install -r requirements.txt
bash code/download_public.sh        # optional; copies are already in data/raw
python code/public_check.py
python code/run_all.py              # about 10 seconds; 2,000 bootstrap draws
```

`run_all.py` prints the headline objects and writes `output/results.json`.
Random seed is fixed (20260818).

## Model in one paragraph

Revenue of a modality-m drug at age t is a Gaussian envelope times an
exponential survival to generic or biosimilar entry. A clock τ_m with price cut
φ on Medicare share θ removes the fraction φθ·T_m(τ_m) of discounted lifetime
revenue, where T_m is the share of discounted revenue after the clock (the tail
share). New-drug counts respond to expected revenue with elasticity η
(N_m = A_m V_m^η). Savings are anchored to CBO's 2031 estimate under current law
and scale with dollar-weighted revenue extracted, which is where the public-data
level ratio λ = R_BIO / R_SM enters. Welfare is (1-ω)·savings minus the value of
forgone drugs minus a patient-access loss, with ω the weight on producer surplus.

## Data sources

- CMS, Medicare Part D Spending by Drug and Medicare Part B Spending by Drug,
  data year 2024 releases (https://data.cms.gov).
- FDA, Drugs@FDA via openFDA (https://open.fda.gov/apis/drug/drugsfda/),
  export dated 2026-10-16.
- Congressional Budget Office (2023), Report 58850.

## License

MIT for code. CMS and FDA data are public domain.
