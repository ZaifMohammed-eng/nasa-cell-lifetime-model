"""Production forecaster built on the research model in src/lifetime.py.

Differences from the research script, all made for serving:
  * the cross-cell prior, noise level and training envelope are fit once and
    stored as a small JSON artifact (no refitting other cells at request time);
  * the noise level comes from training cells only (the research script pooled
    all four cells, including the held-out one, when estimating it);
  * uncertainty uses a Laplace approximation around the MAP fit plus an AR(1)
    inflation instead of a 6000-step emcee run, so a request takes milliseconds;
  * end of life for the power-law trend is solved analytically.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.lifetime import EOL_FRAC, full_model, power_law, true_eol

P0_FULL = [np.log(1e-3), 1.0, np.log(0.05), np.log(24), np.log(5)]
PRIOR_FLOOR = [0.05, 0.005, 0.1, 0.3, 0.1]
N_MAX = 3000


@dataclass
class HParams:
    prior_scale: float = 4.0      # inflation of the cross-cell covariance
    long_rest_h: float = 12.0     # rests longer than this count as recovery events
    noise_mult: float = 1.0       # multiplier on the pooled residual SD
    cov_scale: float = 1.0        # inflation of the Laplace posterior covariance
    min_cycles: int = 30          # below this the service abstains
    max_rel_width: float = 1.0    # (p95 - p05) / p50 above this -> abstain
    n_draws: int = 400


# ------------------------------------------------------------------ helpers
def cell_arrays(d: pd.DataFrame, long_rest_h: float):
    d = d.sort_values("cycle")
    n = d.cycle.values.astype(float)
    q = d.capacity_Ah.values.astype(float)
    rest = d.rest_before_h.values.astype(float)
    ev = [(float(nn), float(r)) for nn, r in zip(n, rest) if np.isfinite(r) and r > long_rest_h]
    return n, q, float(np.mean(q[:3])), ev


def fit_map(n, q, q0, ev, p0, prior=None, noise=0.02):
    def resid(p):
        r = (full_model(p, n, q0, ev) - q) / noise
        if prior is not None:
            mu, L = prior
            r = np.concatenate([r, L @ (p - mu)])
        return r
    return least_squares(resid, p0, max_nfev=20000)


def analytic_eol(log_a, z):
    """First integer cycle where q0*(1 - a n^z) < EOL_FRAC*q0, capped at N_MAX."""
    log_a, z = np.asarray(log_a, float), np.asarray(z, float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        n = np.exp((np.log(1 - EOL_FRAC) - log_a) / z)
    n = np.where((z > 0) & np.isfinite(n), np.floor(n) + 1, N_MAX)
    return np.clip(n, 1, N_MAX)


_FULL_CACHE: dict = {}


def full_life_fit(cell: str, d: pd.DataFrame, long_rest_h: float):
    key = (cell, round(long_rest_h, 3), len(d))
    if key not in _FULL_CACHE:
        n, q, q0, ev = cell_arrays(d, long_rest_h)
        r = fit_map(n, q, q0, ev, np.array(P0_FULL))
        _FULL_CACHE[key] = (r.x, float(np.std(full_model(r.x, n, q0, ev) - q)))
    return _FULL_CACHE[key]


# ------------------------------------------------------------------ model
class CellEOLForecaster:
    """Forecasts cycle-to-80 %-capacity for one cell from its cycle history."""

    def __init__(self, hp: HParams, mu, cov, noise, envelope, train_cells):
        self.hp, self.mu, self.cov = hp, np.asarray(mu), np.asarray(cov)
        self.noise, self.envelope, self.train_cells = float(noise), envelope, list(train_cells)
        self.L = np.linalg.inv(np.linalg.cholesky(self.cov))

    # ---------- training
    @classmethod
    def fit(cls, df: pd.DataFrame, hp: HParams | None = None) -> "CellEOLForecaster":
        hp = hp or HParams()
        cells = sorted(df.cell.unique())
        th, sd = zip(*[full_life_fit(c, df[df.cell == c], hp.long_rest_h) for c in cells])
        th = np.array(th)
        cov = (np.cov(th.T) if len(cells) > 1 else np.zeros((5, 5))) * hp.prior_scale + np.diag(PRIOR_FLOOR)
        noise = float(np.mean(sd)) * hp.noise_mult
        q0s = [cell_arrays(df[df.cell == c], hp.long_rest_h)[2] for c in cells]
        env = {
            "q0_Ah": [float(min(q0s)), float(max(q0s))],
            "T_max_C": [float(df.T_max_C.quantile(0.01)), float(df.T_max_C.quantile(0.99))],
            "capacity_Ah": [float(df.capacity_Ah.min()), float(df.capacity_Ah.max())],
        }
        return cls(hp, th.mean(0), cov, noise, env, cells)

    # ---------- inference
    def predict(self, hist: pd.DataFrame) -> dict:
        hp = self.hp
        reasons = self.validate(hist)
        hard = [r for r in reasons if r.startswith("invalid")]
        out = {"n_cycles_seen": int(len(hist)), "reasons": reasons, "model_train_cells": self.train_cells}
        if hard:
            return {**out, "decision": "reject", "eol_p50": None, "eol_p05": None, "eol_p95": None}
        n, q, q0, ev = cell_arrays(hist, hp.long_rest_h)
        seen = true_eol(q, q0)
        if np.isfinite(seen) and seen < len(q) - 2:
            # An 80 % crossing that coincides with an anomaly (e.g. a capacity jump) is
            # not reported as end of life: a sensor glitch must not retire a cell.
            if any(r.startswith("ood:") for r in reasons):
                reasons.append("eol_crossing_with_anomaly")
                return {**out, "decision": "route_to_engineer", "eol_p50": float(seen),
                        "eol_p05": float(seen), "eol_p95": float(seen)}
            return {**out, "decision": "eol_already_reached", "eol_p50": float(seen),
                    "eol_p05": float(seen), "eol_p95": float(seen)}
        r = fit_map(n, q, q0, ev, self.mu.copy(), prior=(self.mu, self.L), noise=self.noise)
        J = r.jac
        post = np.linalg.pinv(J.T @ J)
        res = full_model(r.x, n, q0, ev) - q
        phi = float(np.clip(np.corrcoef(res[1:], res[:-1])[0, 1], 0.0, 0.95)) if len(res) > 3 else 0.0
        post = post * hp.cov_scale * (1 + phi) / (1 - phi)
        rng = np.random.default_rng(0)
        draws = rng.multivariate_normal(r.x, (post + post.T) / 2, size=hp.n_draws, method="svd")
        eols = analytic_eol(draws[:, 0], draws[:, 1])
        p05, p50, p95 = np.percentile(eols, [5, 50, 95])
        map_eol = float(analytic_eol(r.x[0], r.x[1]))
        if len(hist) < hp.min_cycles:
            reasons.append(f"too_few_cycles:{len(hist)}<{hp.min_cycles}")
        if (p95 - p05) / max(p50, 1) > hp.max_rel_width:
            reasons.append(f"interval_too_wide:{(p95 - p05) / max(p50, 1):.2f}")
        decision = "forecast" if not reasons else "route_to_engineer"
        return {**out, "decision": decision, "eol_map": map_eol, "eol_p50": float(p50),
                "eol_p05": float(p05), "eol_p95": float(p95), "residual_ar1_phi": round(phi, 3)}

    def validate(self, hist: pd.DataFrame) -> list[str]:
        reasons = []
        need = {"cycle", "capacity_Ah", "rest_before_h"}
        if not need.issubset(hist.columns):
            return [f"invalid:missing_columns:{sorted(need - set(hist.columns))}"]
        if len(hist) < 5:
            return ["invalid:fewer_than_5_cycles"]
        c = hist.sort_values("cycle")
        if c.cycle.duplicated().any() or not np.all(np.diff(c.cycle.values) > 0):
            reasons.append("invalid:cycle_index_not_increasing")
        cap = c.capacity_Ah.values
        if not np.all(np.isfinite(cap)) or np.any(cap <= 0.05):
            reasons.append("invalid:nonpositive_or_missing_capacity")
        lo, hi = self.envelope["q0_Ah"]
        q0 = float(np.mean(cap[:3]))
        if not (0.9 * lo <= q0 <= 1.1 * hi):
            reasons.append(f"ood:initial_capacity:{q0:.3f}Ah_outside_[{0.9*lo:.2f},{1.1*hi:.2f}]")
        if "T_max_C" in c.columns and c.T_max_C.notna().any():
            tlo, thi = self.envelope["T_max_C"]
            tm = float(c.T_max_C.median())
            if not (tlo - 3 <= tm <= thi + 3):
                reasons.append(f"ood:temperature:{tm:.1f}C_outside_[{tlo-3:.1f},{thi+3:.1f}]")
        jumps = np.abs(np.diff(cap)) / max(q0, 1e-6)
        if np.any(jumps > 0.15):
            reasons.append("ood:capacity_jump_gt_15pct")
        return reasons

    # ---------- persistence
    def to_json(self) -> str:
        return json.dumps({"hp": asdict(self.hp), "mu": self.mu.tolist(), "cov": self.cov.tolist(),
                           "noise": self.noise, "envelope": self.envelope, "train_cells": self.train_cells})

    @classmethod
    def from_json(cls, s: str) -> "CellEOLForecaster":
        d = json.loads(s)
        return cls(HParams(**d["hp"]), d["mu"], d["cov"], d["noise"], d["envelope"], d["train_cells"])


# ------------------------------------------------------------------ evaluation
def interval_score(lo, hi, y, alpha=0.1):
    return (hi - lo) + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)


def loco_eval(df: pd.DataFrame, hp: HParams, ks=(30, 40, 50), cells=None) -> dict:
    """Leave-one-cell-out over `cells`: fit on the rest, forecast the held-out cell from k cycles."""
    cells = cells or sorted(df.cell.unique())
    rows = []
    for c in cells:
        m = CellEOLForecaster.fit(df[df.cell.isin([o for o in cells if o != c])], hp)
        d = df[df.cell == c].sort_values("cycle")
        q = d.capacity_Ah.values
        te = true_eol(q, float(np.mean(q[:3])))
        for k in ks:
            p = m.predict(d.iloc[:k])
            if p["eol_p50"] is None:
                continue
            rows.append(dict(cell=c, k=k, true_eol=te, p50=p["eol_p50"], map=p.get("eol_map", p["eol_p50"]),
                             p05=p["eol_p05"], p95=p["eol_p95"], decision=p["decision"],
                             abs_err=abs(p["eol_p50"] - te), abs_err_map=abs(p.get("eol_map", p["eol_p50"]) - te),
                             covered=bool(p["eol_p05"] <= te <= p["eol_p95"]),
                             iscore=interval_score(p["eol_p05"], p["eol_p95"], te)))
    r = pd.DataFrame(rows)
    return {"rows": r, "mae": float(r.abs_err.mean()), "mae_map": float(r.abs_err_map.mean()),
            "coverage90": float(r.covered.mean()), "interval_score": float(r.iscore.mean()),
            "mean_width": float((r.p95 - r.p05).mean())}


def fleet_mean_baseline(df: pd.DataFrame, ks=(30, 40, 50), cells=None) -> float:
    """Baseline: predict the mean observed EOL of the training cells, ignoring the new cell's data."""
    cells = cells or sorted(df.cell.unique())
    eol = {c: true_eol(df[df.cell == c].sort_values("cycle").capacity_Ah.values,
                       float(df[df.cell == c].sort_values("cycle").capacity_Ah.values[:3].mean())) for c in cells}
    errs = [abs(np.mean([eol[o] for o in cells if o != c]) - eol[c]) for c in cells for _ in ks]
    return float(np.mean(errs))
