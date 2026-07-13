#!/usr/bin/env python3
"""L4 analysis: does confidence weighting recover pose under injected corruption?

Consumes icp_corruption.csv from the IcpConfidenceCorruption tool and produces the
mechanism-ceiling figure: pose error vs corruption fraction, one line per weight
regime, split by whether the ICP robustness stack (reciprocal + adaptive gate +
trim + Cauchy) is on.

Reading it:
  * oracle vs uniform gap = the MECHANISM CEILING (best case for perfect confidence)
  * real / depth vs uniform = what the actual stereo signal buys
  * if oracle helps but real/depth track uniform -> mechanism fine, signal is the
    problem (the definitive Track-C statement)
  * stack=on vs off = whether weighting adds anything on top of trim/Cauchy

numpy/scipy/matplotlib only (no pandas).

Usage: python3 scripts/analyze_corruption.py --in results/corruption
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Okabe-Ito, fixed roles: oracle = "good" green, uniform = neutral, real/depth = hues.
STYLE = {
    "uniform": dict(color="#555555", ls="--", marker="s", label="uniform (no weight)"),
    "oracle":  dict(color="#009E73", ls="-",  marker="o", label="oracle (w=0 on corrupted)"),
    "real":    dict(color="#D55E00", ls="-",  marker="^", label="real (full product w)"),
    "depth":   dict(color="#0072B2", ls="-",  marker="v", label="depth-only (c_depth)"),
}
ORDER = ["oracle", "uniform", "real", "depth"]


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def mean_ci(vals, conf=0.95):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan
    m = vals.mean()
    if n < 2:
        return m, 0.0
    sem = vals.std(ddof=1) / np.sqrt(n)
    h = sem * stats.t.ppf(0.5 + conf / 2, n - 1)
    return m, h


def curve(rows, stack, regime, metric):
    m = (rows["stack"] == stack) & (rows["regime"] == regime)
    r = rows[m]
    fracs = np.unique(r["corrupt_frac"])
    xs, ys, es = [], [], []
    for f in fracs:
        v = r[metric][r["corrupt_frac"] == f]
        mu, h = mean_ci(v)
        xs.append(f); ys.append(mu); es.append(h)
    return np.array(xs), np.array(ys), np.array(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.indir
    style()

    rows = np.genfromtxt(os.path.join(args.indir, "icp_corruption.csv"),
                         delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    n_cells = len(np.unique(rows["pair"])) * len(np.unique(rows["seed"]))
    mode = str(rows["mode"][0])
    omag = float(rows["outlier_mag"][0])
    print(f"{len(rows)} runs; {n_cells} (pair x seed) reps/cell; "
          f"corruption mode={mode}, magnitude={omag} (normalized).")

    metrics = [("rot_err_deg", "rotation error [deg]"),
               ("trans_err_norm", "translation error [normalized]")]
    stacks = [("off", "robustness stack OFF (weight = only defense)"),
              ("on", "robustness stack ON (reciprocal+gate+trim+Cauchy)")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row_i, (metric, ylab) in enumerate(metrics):
        for col_i, (stack, stitle) in enumerate(stacks):
            ax = axes[row_i, col_i]
            for regime in ORDER:
                x, y, e = curve(rows, stack, regime, metric)
                if len(x) == 0:
                    continue
                st = STYLE[regime]
                ax.errorbar(x * 100, y, yerr=e, capsize=3, markersize=5, linewidth=2,
                            color=st["color"], ls=st["ls"], marker=st["marker"],
                            label=st["label"])
            ax.set_xlabel("corrupted fraction [%]")
            ax.set_ylabel(ylab)
            if row_i == 0:
                ax.set_title(stitle, fontsize=10)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle(f"L4 mechanism test: pose recovery vs injected corruption "
                 f"({mode}, mag={omag}, {n_cells} reps, 95% CI)\n"
                 "oracle–uniform gap = mechanism ceiling; real/depth tracking uniform "
                 "= signal captures none of it", fontsize=12, fontweight="bold")
    p = os.path.join(outdir, "L4_corruption_recovery.png")
    fig.savefig(p)
    plt.close(fig)

    # ---- headline table ----
    print("\nMechanism ceiling and realized benefit (rotation error, deg; mean over reps)")
    for stack, stitle in stacks:
        print(f"\n  stack={stack}")
        fr = np.unique(rows["corrupt_frac"])
        vals = {reg: {f: mean_ci(rows["rot_err_deg"][
            (rows["stack"] == stack) & (rows["regime"] == reg) & (rows["corrupt_frac"] == f)])[0]
            for f in fr} for reg in ORDER}
        hdr = "    frac   " + "".join(f"{f*100:6.0f}%" for f in fr)
        print(hdr)
        for reg in ORDER:
            print(f"    {reg:<7}" + "".join(f"{vals[reg][f]:7.2f}" for f in fr))
        # gaps at the largest fraction
        fmax = fr.max()
        ceil = vals["uniform"][fmax] - vals["oracle"][fmax]
        realized = vals["uniform"][fmax] - vals["depth"][fmax]
        print(f"    -> at {fmax*100:.0f}% corruption: ceiling(uniform-oracle)={ceil:+.2f} deg, "
              f"depth realized={realized:+.2f} deg "
              f"({100*realized/ceil:.0f}% of ceiling)" if abs(ceil) > 1e-6 else
              f"    -> at {fmax*100:.0f}% corruption: ceiling ~ 0 (no room)")

    # ---- machine-readable summary ----
    summ = os.path.join(outdir, "corruption_summary.csv")
    with open(summ, "w") as fh:
        fh.write("stack,regime,corrupt_frac,rot_err_deg_mean,rot_err_deg_ci,"
                 "trans_err_norm_mean,trans_err_norm_ci\n")
        for stack, _ in stacks:
            for reg in ORDER:
                for f in np.unique(rows["corrupt_frac"]):
                    m = (rows["stack"] == stack) & (rows["regime"] == reg) & (rows["corrupt_frac"] == f)
                    rm, rh = mean_ci(rows["rot_err_deg"][m])
                    tm, th = mean_ci(rows["trans_err_norm"][m])
                    fh.write(f"{stack},{reg},{f},{rm:.4f},{rh:.4f},{tm:.6f},{th:.6f}\n")

    print(f"\nWrote {p} and {summ}")


if __name__ == "__main__":
    main()
