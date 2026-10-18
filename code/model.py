"""
Core model for "Medicare Drug Price Negotiation and Pharmaceutical Innovation".

Everything in the paper's welfare tables is a deterministic function of the
primitives in `BASE` below.  The primitives estimated on the restricted
Kythera claims extract (peak year, half-width, median exclusivity, Medicare
share, haircut, elasticity, patient-access loss) are entered as numbers with
their reported standard errors; the public-data check in `public_check.py`
re-estimates the revenue-curve shape from CMS spending files.

Units: dollars in billions of 2024 dollars; time in years since FDA approval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import numpy as np
from scipy import integrate, optimize

# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

MODALITIES = ("SM", "BIO")


@dataclass
class Primitives:
    # Revenue-curve shape by modality (Kythera NLS / Kaplan-Meier)
    tstar: Dict[str, float] = field(default_factory=lambda: {"SM": 8.5, "BIO": 11.0})
    width: Dict[str, float] = field(default_factory=lambda: {"SM": 3.35, "BIO": 4.20})
    median_excl: Dict[str, float] = field(default_factory=lambda: {"SM": 14.4, "BIO": 21.5})
    # Per-drug revenue level of biologics relative to small molecules
    # (CMS public spending files, see public_check.py)
    lam: float = 1.0
    # Policy exposure
    theta: float = 0.33      # Medicare share of lifetime US revenue
    phi: float = 0.40        # negotiated discount relative to net price
    # Innovation response
    eta: float = 0.25        # elasticity of NME count w.r.t. expected NPV
    # Discounting
    rho: float = 0.08        # private real discount rate (firm NPV)
    r_social: float = 0.03   # social real discount rate (welfare aggregator)
    horizon: int = 30        # years of the welfare aggregator
    # Aggregate anchors
    nme_total: float = 1300.0      # CBO (2023) baseline approvals over 30 years
    sm_share: float = 0.66         # small-molecule share of approvals
    cbo_savings_2031: float = 25.0  # CBO (2023) $B federal savings in 2031
    cbo_phi: float = 0.50          # discount implicit in the CBO estimate
    value_per_nme: float = 10.0    # $B social value of a marginal NME (pipeline average)
    value_ratio_bio: float = 1.0   # social value of a biologic NME relative to a small molecule
    patient_loss_cl: float = 8.5   # $B, Kythera abandonment channel, current law
    # Welfare weight on producer surplus (0 = domestic payer perspective)
    omega: float = 0.0

    def n0(self, m: str) -> float:
        return self.nme_total * (self.sm_share if m == "SM" else 1.0 - self.sm_share)

    def level(self, m: str) -> float:
        return 1.0 if m == "SM" else self.lam


BASE = Primitives()

# Standard errors reported for the Kythera-estimated primitives (Table I)
SE = {
    ("tstar", "SM"): 0.31, ("tstar", "BIO"): 0.42,
    ("width", "SM"): 0.18, ("width", "BIO"): 0.24,
    ("median_excl", "SM"): 0.60, ("median_excl", "BIO"): 0.90,
    ("theta", None): 0.021, ("phi", None): 0.030, ("eta", None): 0.060,
}

# --------------------------------------------------------------------------
# Policies: (tau_SM, tau_BIO, options)
# --------------------------------------------------------------------------

POLICIES = {
    "Current law": dict(tau={"SM": 9.0, "BIO": 13.0}),
    "EPIC (both year 13)": dict(tau={"SM": 13.0, "BIO": 13.0}),
    "Converge at year 11": dict(tau={"SM": 11.0, "BIO": 11.0}),
    "Reversed clocks (13, 9)": dict(tau={"SM": 13.0, "BIO": 9.0}),
    "Orphan carve-out": dict(tau={"SM": 9.0, "BIO": 13.0}, exempt={"SM": 0.20, "BIO": 0.0}),
    "Repeal": dict(tau={"SM": math.inf, "BIO": math.inf}),
}

# --------------------------------------------------------------------------
# Revenue curve, NPV and tail share
# --------------------------------------------------------------------------


def hazard(p: Primitives, m: str) -> float:
    return math.log(2.0) / p.median_excl[m]


def flow_density(p: Primitives, m: str, t: float, discount: float | None = None) -> float:
    """Discounted revenue flow f_m(t) = g_m(t) S_m(t) e^{-rho t} (shape only)."""
    rho = p.rho if discount is None else discount
    g = math.exp(-0.5 * ((t - p.tstar[m]) / p.width[m]) ** 2)
    s = math.exp(-hazard(p, m) * t)
    return g * s * math.exp(-rho * t)


def _integral(p: Primitives, m: str, lo: float, hi: float, discount=None) -> float:
    if lo >= hi:
        return 0.0
    val, _ = integrate.quad(lambda t: flow_density(p, m, t, discount), lo, hi, limit=200)
    return val


T_MAX = 80.0


def npv_shape(p: Primitives, m: str, tau: float, phi: float | None = None) -> float:
    """NPV of the shape-normalised revenue curve under a clock tau (no MFP if tau = inf)."""
    phi = p.phi if phi is None else phi
    if math.isinf(tau):
        return _integral(p, m, 0.0, T_MAX)
    pre = _integral(p, m, 0.0, min(tau, T_MAX))
    post = _integral(p, m, min(tau, T_MAX), T_MAX)
    return pre + (1.0 - phi * p.theta) * post


def tail_share(p: Primitives, m: str, tau: float) -> float:
    """Share of discounted lifetime revenue earned after the clock expires."""
    if math.isinf(tau):
        return 0.0
    return _integral(p, m, tau, T_MAX) / _integral(p, m, 0.0, T_MAX)


# --------------------------------------------------------------------------
# Policy evaluation
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    policy: str
    missing_nmes: float
    missing_by_mod: Dict[str, float]
    savings: float                 # $B PV at social rate (consumer/taxpayer gain)
    savings_share_by_mod: Dict[str, float]
    innovation_loss: float         # $B PV
    patient_loss: float            # $B PV
    net_welfare: float             # $B PV at the chosen omega
    nmes_per_100b: float           # NMEs forgone per $100B of savings
    v_ratio: Dict[str, float]


def annuity(r: float, n: int) -> float:
    return (1.0 - (1.0 + r) ** (-n)) / r


def evaluate(p: Primitives, name: str, spec: dict) -> Outcome:
    tau = spec["tau"]
    exempt = spec.get("exempt", {"SM": 0.0, "BIO": 0.0})

    v0 = {m: npv_shape(p, m, math.inf) for m in MODALITIES}
    v1 = {m: npv_shape(p, m, tau[m]) for m in MODALITIES}

    # Dollar NPV extracted per modality (relative units: N0 * level * delta V)
    extracted = {m: p.n0(m) * p.level(m) * (1.0 - exempt[m]) * (v0[m] - v1[m]) for m in MODALITIES}
    # Reference extraction under current law pins the CBO anchor
    cl = POLICIES["Current law"]["tau"]
    extracted_cl = sum(p.n0(m) * p.level(m) * (v0[m] - npv_shape(p, m, cl[m])) for m in MODALITIES)
    scale = (p.cbo_savings_2031 * (p.phi / p.cbo_phi) * annuity(p.r_social, p.horizon)) / extracted_cl

    total_extracted = sum(extracted.values())
    savings = scale * total_extracted
    share = {m: (extracted[m] / total_extracted if total_extracted > 0 else 0.0) for m in MODALITIES}

    # Innovation response: N_m = A_m V_m^eta, applied to the non-exempt pipeline.
    missing = {}
    vr = {}
    for m in MODALITIES:
        ratio = v1[m] / v0[m]
        vr[m] = ratio
        missing[m] = p.n0(m) * (1.0 - exempt[m]) * (1.0 - ratio ** p.eta)
    missing_total = sum(missing.values())
    v_sm = p.value_per_nme * p.nme_total / (p.n0("SM") + p.n0("BIO") * p.value_ratio_bio)
    value = {"SM": v_sm, "BIO": v_sm * p.value_ratio_bio}
    innovation_loss = sum(missing[m] * value[m] for m in MODALITIES)

    # Patient access loss scales with the share of SM extraction exposed
    # (the channel runs through oral cardiometabolic therapies).
    sm_ratio_cl = (v0["SM"] - npv_shape(p, "SM", cl["SM"])) / v0["SM"]
    sm_ratio = (1.0 - exempt["SM"]) * (v0["SM"] - v1["SM"]) / v0["SM"]
    patient_loss = p.patient_loss_cl * (sm_ratio / sm_ratio_cl if sm_ratio_cl > 0 else 0.0)
    if math.isinf(tau["SM"]) and math.isinf(tau["BIO"]):
        patient_loss = 0.0

    net = (1.0 - p.omega) * savings - innovation_loss - patient_loss
    per100 = 100.0 * missing_total / savings if savings > 0 else float("nan")
    return Outcome(name, missing_total, missing, savings, share, innovation_loss,
                   patient_loss, net, per100, vr)


def evaluate_all(p: Primitives) -> Dict[str, Outcome]:
    return {k: evaluate(p, k, v) for k, v in POLICIES.items()}


# --------------------------------------------------------------------------
# Break-even elasticity and iso-savings design
# --------------------------------------------------------------------------


def eta_breakeven(p: Primitives, a: str = "Current law", b: str = "EPIC (both year 13)") -> float:
    """Elasticity at which net welfare under policy a equals policy b."""
    def gap(eta):
        q = replace(p, eta=eta)
        return evaluate(q, a, POLICIES[a]).net_welfare - evaluate(q, b, POLICIES[b]).net_welfare
    return optimize.brentq(gap, 1e-4, 5.0)


def eta_sign_flip(p: Primitives, a: str = "Current law") -> float:
    """Elasticity at which policy a stops improving on repeal."""
    def w(eta):
        q = replace(p, eta=eta)
        return evaluate(q, a, POLICIES[a]).net_welfare
    return optimize.brentq(w, 1e-4, 5.0)


def iso_savings_frontier(p: Primitives, target_savings: float,
                         grid=np.arange(5.0, 20.01, 0.25)) -> Tuple[np.ndarray, list]:
    """For each tau_SM on the grid, find tau_BIO delivering target savings; return
    the pair that minimises missing NMEs."""
    rows = []
    for tsm in grid:
        def f(tb):
            o = evaluate(p, "x", dict(tau={"SM": float(tsm), "BIO": float(tb)}))
            return o.savings - target_savings
        lo, hi = 2.0, 40.0
        if f(lo) < 0 or f(hi) > 0:
            continue
        tb = optimize.brentq(f, lo, hi)
        o = evaluate(p, "x", dict(tau={"SM": float(tsm), "BIO": float(tb)}))
        rows.append((float(tsm), float(tb), o.missing_nmes, o.net_welfare))
    best = min(rows, key=lambda r: r[2]) if rows else None
    return np.array(rows), best


# --------------------------------------------------------------------------
# Parametric bootstrap from reported standard errors
# --------------------------------------------------------------------------


def bootstrap(p: Primitives, n: int = 2000, seed: int = 20260818):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in POLICIES}
    etas, flips = [], []
    for _ in range(n):
        q = replace(
            p,
            tstar={m: rng.normal(p.tstar[m], SE[("tstar", m)]) for m in MODALITIES},
            width={m: max(0.5, rng.normal(p.width[m], SE[("width", m)])) for m in MODALITIES},
            median_excl={m: max(3.0, rng.normal(p.median_excl[m], SE[("median_excl", m)])) for m in MODALITIES},
            theta=float(np.clip(rng.normal(p.theta, SE[("theta", None)]), 0.05, 0.9)),
            phi=float(np.clip(rng.normal(p.phi, SE[("phi", None)]), 0.05, 0.9)),
            eta=float(max(0.01, rng.normal(p.eta, SE[("eta", None)]))),
        )
        res = evaluate_all(q)
        for k, o in res.items():
            out[k].append((o.missing_nmes, o.savings, o.innovation_loss, o.net_welfare))
        etas.append(eta_breakeven(q))
        flips.append(eta_sign_flip(q))
    return {k: np.array(v) for k, v in out.items()}, np.array(etas), np.array(flips)


# --------------------------------------------------------------------------
# Fast re-evaluation at alternative elasticities (integrals do not depend on eta)
# --------------------------------------------------------------------------


def net_at_eta(o: Outcome, p: Primitives, eta: float, spec: dict) -> float:
    exempt = spec.get("exempt", {"SM": 0.0, "BIO": 0.0})
    v_sm = p.value_per_nme * p.nme_total / (p.n0("SM") + p.n0("BIO") * p.value_ratio_bio)
    value = {"SM": v_sm, "BIO": v_sm * p.value_ratio_bio}
    loss = sum(p.n0(m) * (1.0 - exempt[m]) * (1.0 - o.v_ratio[m] ** eta) * value[m] for m in MODALITIES)
    return (1.0 - p.omega) * o.savings - loss - o.patient_loss


def eta_breakeven_fast(res: Dict[str, Outcome], p: Primitives,
                       a="Current law", b="EPIC (both year 13)") -> float:
    f = lambda e: net_at_eta(res[a], p, e, POLICIES[a]) - net_at_eta(res[b], p, e, POLICIES[b])
    if f(1e-4) * f(10.0) > 0:
        return float("nan")
    return optimize.brentq(f, 1e-4, 10.0)


def eta_sign_flip_fast(res: Dict[str, Outcome], p: Primitives, a="Current law") -> float:
    f = lambda e: net_at_eta(res[a], p, e, POLICIES[a])
    if f(1e-4) * f(10.0) > 0:
        return float("nan")
    return optimize.brentq(f, 1e-4, 10.0)


def bootstrap_fast(p: Primitives, n: int = 2000, seed: int = 20260818):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in POLICIES}
    etas, flips = [], []
    for _ in range(n):
        q = replace(
            p,
            tstar={m: rng.normal(p.tstar[m], SE[("tstar", m)]) for m in MODALITIES},
            width={m: max(0.5, rng.normal(p.width[m], SE[("width", m)])) for m in MODALITIES},
            median_excl={m: max(3.0, rng.normal(p.median_excl[m], SE[("median_excl", m)])) for m in MODALITIES},
            theta=float(np.clip(rng.normal(p.theta, SE[("theta", None)]), 0.05, 0.9)),
            phi=float(np.clip(rng.normal(p.phi, SE[("phi", None)]), 0.05, 0.9)),
            eta=float(max(0.01, rng.normal(p.eta, SE[("eta", None)]))),
        )
        res = evaluate_all(q)
        for k, o in res.items():
            out[k].append((o.missing_nmes, o.savings, o.innovation_loss, o.net_welfare))
        etas.append(eta_breakeven_fast(res, q))
        flips.append(eta_sign_flip_fast(res, q))
    return {k: np.array(v) for k, v in out.items()}, np.array(etas), np.array(flips)
