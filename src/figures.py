"""Figures for the README (static PNG, light theme, validated categorical slots)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifetime import ROOT, full_model, power_law, LONG_REST_H, fit_ls, EOL_FRAC

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "capacity.csv"))
    m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
    cells = sorted(df.cell.unique())
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

    # 1. capacity with irreversible trend + recovery model
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for ax, c, col in zip(axes.flat, cells, SERIES):
        d = df[df.cell == c].sort_values("cycle")
        n, q = d.cycle.values.astype(float), d.capacity_Ah.values
        q0 = float(np.mean(q[:3]))
        ev = [(float(a), float(b)) for a, b in zip(n, d.rest_before_h.values) if b > LONG_REST_H]
        f = m["full_fit"][c]
        th = np.array([np.log(f["a"]), f["z"], np.log(f["g_mAh"] / 1000), np.log(f["tau_h"]), np.log(f["lam_cycles"])])
        ax.plot(n, q, "o", ms=3.5, color=col, alpha=0.55, label="measured")
        ax.plot(n, full_model(th, n, q0, ev), color=INK, lw=1.6, label="irreversible + recovery")
        ax.plot(n, power_law(th[:2], n, q0), color=INK2, lw=1.4, ls="--", label="irreversible only")
        ax.axhline(EOL_FRAC * q0, color=INK2, lw=0.9, ls=":")
        for e in ev:
            ax.axvline(e[0], color=GRID, lw=1.0, zorder=0)
        ax.set_title(f"{c}  (resid. SD {f['resid_sd_trend_only_mAh']:.0f} → {f['resid_sd_with_recovery_mAh']:.0f} mAh)",
                     fontsize=10, loc="left", color=INK)
        ax.set_ylabel("Capacity (Ah)")
    for ax in axes[1]:
        ax.set_xlabel("Cycle")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Capacity = irreversible power-law fade + recovery after long rests (gray lines)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fit_recovery.png"), dpi=160)
    plt.close(fig)

    # 2. leave-one-cell-out EOL error vs. cycles observed
    ks = [20, 30, 40, 50]
    tags = [("trend_no_prior", "power law, no prior", SERIES[1]),
            ("trend_prior", "power law + cross-cell prior", SERIES[0]),
            ("trend_recovery_prior", "+ recovery term", SERIES[2])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, lab, col in tags:
        y = [m["loco_summary"][f"k{k}_{tag}"]["mean_abs_err_cycles"] for k in ks]
        ax.plot(ks, y, "-o", color=col, lw=2, ms=8, mec=SURF, mew=2, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlim(17, 53)
    ax.set_xlabel("Cycles observed on the held-out cell")
    ax.set_ylabel("Mean |EOL error| (cycles, log scale)")
    ax.set_title("Leave-one-cell-out end-of-life forecast error", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "loco_error.png"), dpi=160)
    plt.close(fig)

    # 3. Bayesian 90 % EOL intervals, iid vs AR(1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, c in enumerate(cells):
        for j, (mode, col, lab) in enumerate([("iid", SERIES[1], "i.i.d. residuals"), ("ar1", SERIES[0], "AR(1) residuals")]):
            b = m["bayes_k40"][mode][c]
            y = i + (j - 0.5) * 0.28
            ax.plot([b["p05"], b["p95"]], [y, y], color=col, lw=2.5, solid_capstyle="round", label=lab if i == 0 else None)
            ax.plot(b["p50"], y, "o", color=col, ms=8, mec=SURF, mew=2)
        ax.plot(m["bayes_k40"]["ar1"][c]["true_eol"], i, marker="|", ms=22, mew=2.2, color=INK,
                label="observed EOL" if i == 0 else None)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells)
    ax.invert_yaxis()
    ax.set_xlabel("End-of-life cycle (90 % posterior interval, 40 cycles observed)")
    cov = m["bayes_k40_coverage"]
    ax.set_title(f"Interval coverage: i.i.d. {cov['iid']}, AR(1) {cov['ar1']}", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "eol_intervals.png"), dpi=160)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
