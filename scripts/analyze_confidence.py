#!/usr/bin/env python3
"""Confidence-ICP analysis, diagnostic core (L0 + L1) + candidate-cue validation.

Consumes the CSV written by the ConfidenceAnalysis C++ tool and, for every
confidence signal present, measures whether it ranks points by their TRUE error
against the DTU ground-truth scan -- the Hu-Mordohai / Poggi ROC/AUC protocol:

  L0  reliability diagram (binned predicted reliability -> empirical error),
      Spearman rho(signal, error) [want << 0], outlier-detection AUC [want >> 0.5].
  L1  distribution / dynamic range / concentration; spatial leverage (for w).

Signals analysed (whichever columns are present):
  production weight/factors : w, c_depth, c_edge, c_stereo
  candidate cost-curve cues : c_curv (curvature), c_pkr (peak ratio), c_tex
                              (image texture), w_new (curvature-propagated 1/sigma_Z^2)

numpy/scipy/matplotlib only (no pandas).

Usage: python3 scripts/analyze_confidence.py --in results/confidence [--out DIR]
"""
import argparse
import math
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Okabe-Ito + a couple of extra CVD-safe hues, one per signal (fixed order).
LABEL = {
    "w": "w (product)", "c_depth": "c_depth", "c_edge": "c_edge", "c_stereo": "c_stereo",
    "c_curv": "c_curv (curvature)", "c_pkr": "c_pkr (peak ratio)", "c_tex": "c_tex (texture)",
    "w_new": "w_new (curv-propagated)",
}
COLOR = {
    "w": "#000000", "c_depth": "#0072B2", "c_edge": "#D55E00", "c_stereo": "#009E73",
    "c_curv": "#CC79A7", "c_pkr": "#E69F00", "c_tex": "#56B4E9", "w_new": "#117733",
}
CANDIDATE_ORDER = ["w", "w_new", "c_depth", "c_curv", "c_pkr", "c_tex", "c_edge", "c_stereo"]
ACCENT, MUTED = "#E69F00", "#999999"


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    })


def load(indir):
    pts = np.genfromtxt(os.path.join(indir, "confidence_points.csv"), delimiter=",", names=True)
    meta = np.atleast_1d(np.genfromtxt(os.path.join(indir, "confidence_pairs.csv"),
                                       delimiter=",", names=True))
    if pts.size == 0:
        sys.exit("No points loaded -- did ConfidenceAnalysis run?")
    return pts, meta


def signals_present(pts):
    return [s for s in CANDIDATE_ORDER if s in pts.dtype.names]


# ------------------------------- metrics ----------------------------------- #
def auc_reliability(score, error, err_threshold):
    """AUC that 'score' (higher = more reliable) separates good (err<=thr) from bad."""
    m = np.isfinite(score) & np.isfinite(error)
    score, error = score[m], error[m]
    good = error <= err_threshold
    n_pos, n_neg = int(good.sum()), int((~good).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[good].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def decile_curve(score, error, nbins=10):
    m = np.isfinite(score) & np.isfinite(error)
    score, error = score[m], error[m]
    order = np.argsort(score, kind="mergesort")
    x, med, q1, q3 = [], [], [], []
    for k, idx in enumerate(np.array_split(order, nbins)):
        if len(idx):
            e = error[idx]
            x.append(k + 1); med.append(np.median(e))
            q1.append(np.percentile(e, 25)); q3.append(np.percentile(e, 75))
    return np.array(x), np.array(med), np.array(q1), np.array(q3)


def cov(a):
    a = a[np.isfinite(a)]; m = np.mean(a)
    return float(np.std(a) / m) if m != 0 else float("nan")


def dyn_range(a):
    a = a[np.isfinite(a)]; p1, p99 = np.percentile(a, [1, 99])
    return float(p99 / p1) if p1 > 0 else float("inf")


def top_mass(a, frac=0.10):
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return float("nan")
    s = np.sort(a)[::-1]; k = max(1, int(frac * len(s)))
    return float(s[:k].sum() / s.sum())


def spearman(score, error):
    m = np.isfinite(score) & np.isfinite(error)
    return spearmanr(score[m], error[m]).correlation


# ------------------------------- figures ----------------------------------- #
def grid(n):
    ncol = 3 if n > 4 else 2
    return math.ceil(n / ncol), ncol


def fig_reliability(pts, signals, outdir):
    err = pts["gt_err_mm"]
    nrow, ncol = grid(len(signals))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for s, ax in zip(signals, axes.ravel()):
        ax.set_visible(True)
        x, med, q1, q3 = decile_curve(pts[s], err)
        ax.fill_between(x, q1, q3, color=COLOR[s], alpha=0.15, linewidth=0)
        ax.plot(x, med, "-o", color=COLOR[s], markersize=4, linewidth=2)
        rho = spearman(pts[s], err)
        tag = "informative" if rho < -0.1 else ("inverted!" if rho > 0.1 else "flat")
        ax.set_title(f"{LABEL[s]}   rho={rho:+.3f} ({tag})")
        ax.set_xlabel("predicted-reliability decile (1=least)")
        ax.set_ylabel("true error to GT [mm]")
    fig.suptitle("L0 reliability: does higher predicted confidence => lower true error?\n"
                 "(useful signal slopes DOWN; production factors top row, candidate cues after)",
                 fontsize=12, fontweight="bold")
    p = os.path.join(outdir, "L0_reliability_diagram.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_auc(pts, signals, outdir, thresholds):
    err = pts["gt_err_mm"]
    table = {s: [auc_reliability(pts[s], err, t) for t in thresholds] for s in signals}
    fig, ax = plt.subplots(figsize=(1.3 * len(signals) + 2, 4.8), constrained_layout=True)
    x = np.arange(len(signals)); width = 0.8 / len(thresholds)
    for ti, t in enumerate(thresholds):
        ax.bar(x + ti * width, [table[s][ti] for s in signals], width,
               label=f"err > {t:g} mm", color=plt.cm.Greys(0.4 + 0.5 * ti / len(thresholds)))
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([LABEL[s] for s in signals], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC (confidence separates good/bad)")
    ax.set_ylim(0.3, 1.0); ax.legend(fontsize=8)
    ax.set_title("L0 outlier-detection power per signal (0.5 = useless)")
    p = os.path.join(outdir, "L0_outlier_auc.png")
    fig.savefig(p); plt.close(fig)
    return p, table


def fig_distributions(pts, signals, outdir):
    nrow, ncol = grid(len(signals))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.0 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for s, ax in zip(signals, axes.ravel()):
        ax.set_visible(True)
        a = pts[s]; a = a[np.isfinite(a)]
        logx = np.all(a > 0) and a.max() / max(a.min(), 1e-9) > 50
        if logx:
            bins = np.logspace(np.log10(max(a.min(), 1e-6)), np.log10(a.max()), 60); ax.set_xscale("log")
        else:
            bins = 60
        ax.hist(a, bins=bins, color=COLOR[s], alpha=0.85)
        ax.axvline(np.median(a), color=ACCENT, ls="--", lw=1.3)
        ax.set_title(f"{LABEL[s]}  CoV={cov(a):.2f} p99/p1={dyn_range(a):.1f}")
        ax.set_xlabel(LABEL[s]); ax.set_ylabel("count")
    fig.suptitle("L1 distributions (dashed = median)", fontsize=12, fontweight="bold")
    p = os.path.join(outdir, "L1_distributions.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_spatial_leverage(pts, meta, outdir):
    xyz = np.column_stack([pts["wx"], pts["wy"], pts["wz"]])
    rms_by_id = {int(r["pair_id"]): float(r["rms_radius_mm"]) for r in meta}
    seps, labels = [], []
    for pid in np.unique(pts["pair_id"]).astype(int):
        m = pts["pair_id"] == pid
        w, P = pts["w"][m], xyz[m]
        if len(w) < 100:
            continue
        lo, hi = w <= np.percentile(w, 10), w >= np.percentile(w, 90)
        rms = rms_by_id.get(pid, np.nan)
        seps.append(np.linalg.norm(P[hi].mean(0) - P[lo].mean(0)) / rms if rms > 0 else np.nan)
        labels.append(f"{int(pts['left'][m][0])}-{int(pts['right'][m][0])}")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.bar(np.arange(len(seps)), seps, color="#0072B2")
    ax.axhline(np.nanmean(seps), color=ACCENT, ls="--", lw=1.5, label=f"mean={np.nanmean(seps):.2f}")
    ax.set_xticks(np.arange(len(seps))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("centroid sep (top vs bottom decile w) / RMS radius")
    ax.set_title("L1 spatial leverage of w (small => co-located => low leverage)")
    ax.legend()
    p = os.path.join(outdir, "L1_spatial_leverage.png")
    fig.savefig(p); plt.close(fig)
    return p, float(np.nanmean(seps))


# --------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", default=None)
    ap.add_argument("--err-thresholds", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    args = ap.parse_args()
    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)
    style()

    pts, meta = load(args.indir)
    signals = signals_present(pts)
    err = pts["gt_err_mm"]
    print(f"\nLoaded {len(err):,} points across {len(np.unique(pts['pair_id']))} pairs.")
    print(f"Signals present: {', '.join(signals)}")
    print(f"True error to GT [mm]: median={np.median(err):.3f}, p90={np.percentile(err,90):.3f}\n")

    print("L0  Spearman rho(signal, true_error)  [want << 0]")
    rho = {}
    for s in signals:
        rho[s] = spearman(pts[s], err)
        print(f"     {LABEL[s]:<26} rho = {rho[s]:+.4f}")

    r1 = fig_reliability(pts, signals, outdir)
    r2, auc_table = fig_auc(pts, signals, outdir, args.err_thresholds)
    print("\nL0  outlier-detection AUC  [0.5 = useless]")
    print("     " + " " * 26 + "  ".join(f"{t:g}mm" for t in args.err_thresholds))
    for s in signals:
        print(f"     {LABEL[s]:<26}" + "  ".join(f"{a:.3f}" for a in auc_table[s]))

    d1 = fig_distributions(pts, signals, outdir)
    d2, mean_sep = fig_spatial_leverage(pts, meta, outdir)

    with open(os.path.join(outdir, "confidence_summary.csv"), "w") as fh:
        fh.write("signal,spearman_rho,cov,dyn_range_p99_p1,top10pct_mass,"
                 + ",".join(f"auc_gt{t:g}mm" for t in args.err_thresholds) + "\n")
        for s in signals:
            a = pts[s][np.isfinite(pts[s])]
            fh.write(f"{s},{rho[s]:.4f},{cov(a):.4f},{dyn_range(a):.4f},{top_mass(a):.4f},"
                     + ",".join(f"{v:.4f}" for v in auc_table[s]) + "\n")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    ranked = sorted(signals, key=lambda s: rho[s])  # most negative first
    print(" Signals ranked by rho(., error), most informative first:")
    for s in ranked:
        print(f"   {LABEL[s]:<26} rho={rho[s]:+.3f}  AUC@2mm={auc_table[s][min(1,len(auc_table[s])-1)]:.3f}")
    cand = [s for s in ("c_curv", "c_pkr", "w_new") if s in rho]
    if cand:
        best_new = min(cand, key=lambda s: rho[s])
        verdict = ("beat" if rho[best_new] < rho.get("c_depth", 0) - 0.02 else
                   "did NOT beat")
        print(f"\n Best candidate cue: {LABEL[best_new]} (rho={rho[best_new]:+.3f}) "
              f"-- {verdict} the incumbent c_depth (rho={rho.get('c_depth', float('nan')):+.3f}).")
    print(f" Figures + confidence_summary.csv in {outdir}")
    for p in (r1, r2, d1, d2):
        print("   -", p)


if __name__ == "__main__":
    main()
