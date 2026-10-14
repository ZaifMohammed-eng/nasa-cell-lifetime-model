"""Capacity-fade lifetime models for the NASA PCoE 18650 cells (B0005/6/7/18).

Measured capacity = irreversible fade + recoverable capacity.

    Q(n) = q0 * (1 - a * n**z)                                   irreversible (power law)
         + sum_j  g * (1 - exp(-rest_j / tau)) * exp(-(n - n_j) / lam) * [n >= n_j]
                                                                 recovery after long rests

Steps (all numbers -> results/metrics.json)
1. Fade-law comparison: power law, sqrt+linear, double exponential, each fit
   on the first 40 cycles (before every cell's end of life) and scored on the rest.
2. Leave-one-cell-out forecasting: priors learned from the other three cells,
   fit on the first k cycles of the held-out cell, with and without the
   recovery term.
3. Bayesian end-of-life posterior (emcee) at k = 40: i.i.d. vs AR(1) residuals,
   90 % interval coverage.
4. Regeneration statistics: capacity jump vs preceding rest time.
"""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
import emcee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOL_FRAC = 0.80        # end of life at 80 % of initial capacity
LONG_REST_H = 12.0     # rests longer than this are recovery events
K_COMPARE = 40         # fit window for the fade-law comparison (< every cell's EOL)
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ fade laws
def power_law(p, n, q0):                      # p = [log a, z]
    return q0 * (1.0 - np.exp(p[0]) * n ** p[1])

def sqrt_linear(p, n, q0):                    # SEI sqrt(n) + linear (LAM-like)
    return q0 - p[0] * np.sqrt(n) - p[1] * n

def double_exp(p, n, q0):                     # empirical double exponential
    return p[0] * np.exp(p[1] * n) + p[2] * np.exp(p[3] * n)

LAWS = {"power_law": power_law, "sqrt_linear": sqrt_linear, "double_exp": double_exp}

def law_p0(name, q0):
    return {"power_law": [np.log(1e-3), 1.0], "sqrt_linear": [1e-3, 1e-3],
            "double_exp": [q0 * 0.99, -2e-3, q0 * 0.01, 1e-2]}[name]


# ------------------------------------------------------ recovery (regeneration)
def recovery(r, n, events):
    """r = [log g, log tau, log lam]; events = list of (cycle, rest_h)."""
    g, tau, lam = np.exp(r)
    out = np.zeros_like(n, dtype=float)
    for nj, rest in events:
        m = n >= nj
        out[m] += g * (1 - np.exp(-rest / tau)) * np.exp(-(n[m] - nj) / lam)
    return out

def full_model(th, n, q0, events):           # th = [log a, z, log g, log tau, log lam]
    return power_law(th[:2], n, q0) + recovery(th[2:], n, events)


def fit_ls(fun, p0, n, q, prior=None, noise=0.02):
    def resid(p):
        r = (fun(p, n) - q) / noise
        if prior is not None:
            mu, L = prior
            r = np.concatenate([r, L @ (p - mu)])
        return r
    return least_squares(resid, p0, max_nfev=20000).x


def eol_cycle(trend, q0, n_max=3000):
    n = np.arange(1, n_max + 1, dtype=float)
    idx = np.where(trend(n) < EOL_FRAC * q0)[0]
    return float(n[idx[0]]) if idx.size else np.inf


def true_eol(q, q0):
    """First cycle where the 5-cycle running median of measured capacity drops below 80 % of q0."""
    med = pd.Series(q).rolling(5, center=True, min_periods=1).median().values
    idx = np.where(med < EOL_FRAC * q0)[0]
    return float(idx[0] + 1) if idx.size else np.nan


def prior_from(params, scale=4.0, floor=None):
    mu = params.mean(0)
    cov = np.cov(params.T) * scale + np.diag(floor)
    return mu, np.linalg.inv(np.linalg.cholesky(cov)), np.linalg.inv(cov)


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "capacity.csv"))
    cells = sorted(df.cell.unique())
    D = {}
    for c in cells:
        d = df[df.cell == c].sort_values("cycle")
        q = d.capacity_Ah.values
        n = d.cycle.values.astype(float)
        ev = [(float(nn), float(r)) for nn, r in zip(n, d.rest_before_h.values) if r > LONG_REST_H]
        D[c] = dict(n=n, q=q, q0=float(np.mean(q[:3])), ev=ev, rest=d.rest_before_h.values)
    out = {"settings": {"eol_fraction": EOL_FRAC, "long_rest_h": LONG_REST_H, "k_compare": K_COMPARE},
           "cells": {}}

    # ---------------- 1. fade-law comparison on the first K_COMPARE cycles
    for c in cells:
        n, q, q0 = D[c]["n"], D[c]["q"], D[c]["q0"]
        k, te = K_COMPARE, true_eol(q, q0)
        rec = {"n_cycles": int(len(n)), "q0_Ah": round(q0, 4), "fade_pct": round(100 * (1 - q[-1] / q0), 1),
               "true_eol_cycle": te, "n_long_rests": len(D[c]["ev"]), "laws": {}}
        for name, f in LAWS.items():
            p = fit_ls(lambda p, x: f(p, x, q0), law_p0(name, q0), n[:k], q[:k])
            rmse = float(np.sqrt(np.mean((f(p, n[k:], q0) - q[k:]) ** 2)))
            pe = eol_cycle(lambda x: f(p, x, q0), q0)
            rec["laws"][name] = {"heldout_rmse_mAh": round(1000 * rmse, 1), "pred_eol": pe,
                                 "eol_abs_err": abs(pe - te) if np.isfinite(pe) else None}
        out["cells"][c] = rec
    out["law_summary"] = {}
    for name in LAWS:
        r = [out["cells"][c]["laws"][name]["heldout_rmse_mAh"] for c in cells]
        e = [out["cells"][c]["laws"][name]["eol_abs_err"] for c in cells]
        ef = [x for x in e if x is not None]
        out["law_summary"][name] = {"mean_rmse_mAh": round(float(np.mean(r)), 1),
                                    "mean_eol_abs_err": round(float(np.mean(ef)), 1) if ef else None,
                                    "n_never_reach_eol": len(e) - len(ef)}

    # ---------------- full-life fits (for priors)
    full_trend, full_rec, res_sd = {}, {}, {}
    for c in cells:
        n, q, q0, ev = D[c]["n"], D[c]["q"], D[c]["q0"], D[c]["ev"]
        th = fit_ls(lambda p, x: full_model(p, x, q0, ev), [np.log(1e-3), 1.0, np.log(0.05), np.log(24), np.log(5)], n, q)
        full_rec[c] = th
        full_trend[c] = fit_ls(lambda p, x: power_law(p, x, q0), [np.log(1e-3), 1.0], n, q)
        res_sd[c] = (float(np.std(power_law(full_trend[c], n, q0) - q)),
                     float(np.std(full_model(th, n, q0, ev) - q)))
    out["full_fit"] = {c: {"a": float(np.exp(full_rec[c][0])), "z": round(float(full_rec[c][1]), 3),
                           "g_mAh": round(1000 * float(np.exp(full_rec[c][2])), 1),
                           "tau_h": round(float(np.exp(full_rec[c][3])), 1),
                           "lam_cycles": round(float(np.exp(full_rec[c][4])), 1),
                           "resid_sd_trend_only_mAh": round(1000 * res_sd[c][0], 1),
                           "resid_sd_with_recovery_mAh": round(1000 * res_sd[c][1], 1)} for c in cells}
    noise = float(np.mean([v[1] for v in res_sd.values()]))

    # ---------------- 2. leave-one-cell-out forecasting
    ks = [20, 30, 40, 50]
    loco = {}
    for c in cells:
        oth = [o for o in cells if o != c]
        mu_t, L_t, _ = prior_from(np.array([full_trend[o] for o in oth]), floor=[0.05, 0.005])
        mu_r, L_r, _ = prior_from(np.array([full_rec[o] for o in oth]), floor=[0.05, 0.005, 0.1, 0.3, 0.1])
        n, q, q0, ev = D[c]["n"], D[c]["q"], D[c]["q0"], D[c]["ev"]
        te = true_eol(q, q0)
        loco[c] = {"true_eol": te}
        for k in ks:
            evk = [e for e in ev if e[0] <= n[k - 1]]
            p_np = fit_ls(lambda p, x: power_law(p, x, q0), [np.log(1e-3), 1.0], n[:k], q[:k])
            p_pr = fit_ls(lambda p, x: power_law(p, x, q0), mu_t.copy(), n[:k], q[:k], prior=(mu_t, L_t), noise=noise)
            th = fit_ls(lambda p, x: full_model(p, x, q0, evk), mu_r.copy(), n[:k], q[:k], prior=(mu_r, L_r), noise=noise)
            r = {}
            for tag, tr in [("trend_no_prior", lambda x, p=p_np: power_law(p, x, q0)),
                            ("trend_prior", lambda x, p=p_pr: power_law(p, x, q0)),
                            ("trend_recovery_prior", lambda x, t=th: power_law(t[:2], x, q0))]:
                pe = eol_cycle(tr, q0)
                r[tag] = {"eol": pe, "abs_err": abs(pe - te) if np.isfinite(pe) else None}
            loco[c][f"k{k}"] = r
    out["loco"] = loco
    out["loco_summary"] = {}
    for k in ks:
        for tag in ["trend_no_prior", "trend_prior", "trend_recovery_prior"]:
            e = [loco[c][f"k{k}"][tag]["abs_err"] for c in cells]
            ef = [x for x in e if x is not None]
            te = [loco[c]["true_eol"] for c in cells]
            rel = [x / t for x, t in zip(e, te) if x is not None]
            out["loco_summary"][f"k{k}_{tag}"] = {"mean_abs_err_cycles": round(float(np.mean(ef)), 1) if ef else None,
                                                  "mean_rel_err_pct": round(100 * float(np.mean(rel)), 1) if rel else None,
                                                  "n_never_reach_eol": len(e) - len(ef)}

    # ---------------- 3. Bayesian EOL at k = 40: iid vs AR(1) residuals
    k = 40
    bayes = {"iid": {}, "ar1": {}}
    for c in cells:
        oth = [o for o in cells if o != c]
        mu_r, L_r, icov_r = prior_from(np.array([full_rec[o] for o in oth]), floor=[0.05, 0.005, 0.1, 0.3, 0.1])
        n, q, q0, ev = D[c]["n"], D[c]["q"], D[c]["q0"], D[c]["ev"]
        nk, qk = n[:k], q[:k]
        evk = [e for e in ev if e[0] <= nk[-1]]
        te = true_eol(q, q0)
        th0 = fit_ls(lambda p, x: full_model(p, x, q0, evk), mu_r.copy(), nk, qk, prior=(mu_r, L_r), noise=noise)
        for mode in ["iid", "ar1"]:
            ndim = 5 + 1 + (mode == "ar1")
            def logp(t):
                th, log_s = t[:5], t[5]
                phi = t[6] if mode == "ar1" else 0.0
                if not (-14 < th[0] < 0 and 0.2 < th[1] < 3 and -8 < log_s < 0 and -0.95 < phi < 0.95):
                    return -np.inf
                s = np.exp(log_s)
                r = full_model(th, nk, q0, evk) - qk
                e = r[1:] - phi * r[:-1]
                ll = -0.5 * (r[0] ** 2 * (1 - phi ** 2) + np.sum(e ** 2)) / s ** 2 - k * log_s + 0.5 * np.log(1 - phi ** 2)
                d = th - mu_r
                return ll - 0.5 * d @ icov_r @ d
            start = np.r_[th0, np.log(noise)] if mode == "iid" else np.r_[th0, np.log(noise), 0.5]
            nw = 32
            p0 = start + 1e-3 * RNG.standard_normal((nw, ndim))
            S = emcee.EnsembleSampler(nw, ndim, logp)
            S.run_mcmc(p0, 6000, progress=False)
            ch = S.get_chain(discard=2000, thin=10, flat=True)
            eols = np.array([eol_cycle(lambda x, t=t: power_law(t[:2], x, q0), q0, 1500) for t in ch])
            eols = np.where(np.isfinite(eols), eols, 1500)
            lo, med, hi = np.percentile(eols, [5, 50, 95])
            bayes[mode][c] = {"p05": float(lo), "p50": float(med), "p95": float(hi), "true_eol": te,
                              "covered_90": bool(lo <= te <= hi), "width": float(hi - lo),
                              "phi_median": float(np.median(ch[:, 6])) if mode == "ar1" else None,
                              "acceptance": round(float(np.mean(S.acceptance_fraction)), 2)}
    out["bayes_k40"] = bayes
    out["bayes_k40_coverage"] = {m: f"{sum(v['covered_90'] for v in bayes[m].values())}/{len(cells)}" for m in bayes}

    # ---------------- 4. regeneration statistics
    rows = []
    for c in cells:
        q, rest = D[c]["q"], D[c]["rest"]
        for i in range(1, len(q)):
            rows.append((c, rest[i], q[i] - q[i - 1]))
    reg = pd.DataFrame(rows, columns=["cell", "rest_h", "dq_Ah"]).dropna()
    lg, sh = reg[reg.rest_h > LONG_REST_H], reg[reg.rest_h <= LONG_REST_H]
    rho, pv = spearmanr(reg.rest_h, reg.dq_Ah)
    out["regeneration"] = {"n_long_rest_events": int(len(lg)),
                           "mean_jump_after_long_rest_mAh": round(1000 * float(lg.dq_Ah.mean()), 1),
                           "mean_change_after_short_rest_mAh": round(1000 * float(sh.dq_Ah.mean()), 1),
                           "spearman_rho": round(float(rho), 2), "spearman_p": float(pv)}
    reg.to_csv(os.path.join(ROOT, "results", "regeneration.csv"), index=False)

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list): return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o): return None
        return o
    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as fh:
        json.dump(clean(out), fh, indent=2)

    print("law_summary", json.dumps(out["law_summary"]))
    print("full_fit", json.dumps(out["full_fit"], indent=0))
    for kk, v in out["loco_summary"].items():
        print(kk, v)
    for m in bayes:
        for c, v in bayes[m].items():
            print(m, c, {kk: (round(x, 2) if isinstance(x, float) else x) for kk, x in v.items()})
    print("coverage", out["bayes_k40_coverage"], "regen", out["regeneration"])


if __name__ == "__main__":
    main()
