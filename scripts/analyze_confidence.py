#!/usr/bin/env python3
"""Confidence-ICP analysis, diagnostic core (L0 + L1).

Consumes the CSVs written by the ConfidenceAnalysis C++ tool and answers the two
questions that gate the whole confidence-weighted-ICP investigation, WITHOUT
running ICP:

  L0  Is the confidence informative?  Does the weight w (and each factor) rank
      points by their true error against the DTU ground-truth scan?
        - reliability diagram (binned predicted-reliability -> empirical error)
        - Spearman rho(score, error)      (want strongly negative)
        - outlier-detection AUC            (want >> 0.5; 0.5 = useless)
        - depth-uncertainty calibration    (does predicted sigma_Z track reality?)

  L1  Does the weight have leverage?  Even an informative weight is inert for a
      single rigid registration if it barely varies or if high/low-weight points
      are co-located.
        - distribution / dynamic range / concentration of w and each factor
        - spatial separation of high- vs low-confidence points (per pair)

Only numpy/scipy/matplotlib are required (no pandas). Figures + a summary CSV are
written to --out (default: alongside the input CSVs).

Usage:
    python3 scripts/analyze_confidence.py --in results/confidence [--out DIR]
"""
import argparse
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Okabe-Ito colorblind-safe palette, used in a fixed order for the factor series.
CB = {
    "w":        "#000000",  # black  -- the product weight (the headline signal)
    "c_depth":  "#0072B2",  # blue
    "c_edge":   "#D55E00",  # vermillion
    "c_stereo": "#009E73",  # bluish green
    "accent":   "#E69F00",  # orange -- reference / annotation
    "muted":    "#999999",
}
FACTORS = ["w", "c_depth", "c_edge", "c_stereo"]
FACTOR_LABEL = {
    "w": "w (product)",
    "c_depth": "c_depth",
    "c_edge": "c_edge",
    "c_stereo": "c_stereo",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def load(indir):
    pts = np.genfromtxt(os.path.join(indir, "confidence_points.csv"),
                        delimiter=",", names=True)
    meta = np.genfromtxt(os.path.join(indir, "confidence_pairs.csv"),
                         delimiter=",", names=True)
    if pts.size == 0:
        sys.exit("No points loaded -- did ConfidenceAnalysis run?")
    meta = np.atleast_1d(meta)
    return pts, meta


# ----------------------------------------------------------------------------- #
# metrics
# ----------------------------------------------------------------------------- #
def auc_reliability(score, error, err_threshold):
    """AUC that 'score' (higher = more reliable) separates good (error<=thr) from
    bad (error>thr) points. 0.5 = no ranking ability, 1.0 = perfect. Rank-based
    Mann-Whitney form, robust to ties."""
    good = error <= err_threshold  # positive class = reliable point
    n_pos, n_neg = int(good.sum()), int((~good).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over tie groups
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    auc = (ranks[good].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc, n_pos, n_neg


def decile_curve(score, error, nbins=10):
    """Median (and IQR) error within equal-count bins of ascending score."""
    order = np.argsort(score, kind="mergesort")
    parts = np.array_split(order, nbins)
    x, med, q1, q3 = [], [], [], []
    for k, idx in enumerate(parts):
        if len(idx) == 0:
            continue
        e = error[idx]
        x.append(k + 1)
        med.append(np.median(e))
        q1.append(np.percentile(e, 25))
        q3.append(np.percentile(e, 75))
    return np.array(x), np.array(med), np.array(q1), np.array(q3)


def cov(a):
    m = np.mean(a)
    return float(np.std(a) / m) if m != 0 else float("nan")


def dyn_range(a):
    p1, p99 = np.percentile(a, [1, 99])
    return float(p99 / p1) if p1 > 0 else float("inf")


def top_mass(a, frac=0.10):
    """Fraction of total weight carried by the top-`frac` heaviest points."""
    s = np.sort(a)[::-1]
    k = max(1, int(frac * len(s)))
    return float(s[:k].sum() / s.sum()) if s.sum() > 0 else float("nan")


# ----------------------------------------------------------------------------- #
# L0 figures
# ----------------------------------------------------------------------------- #
def fig_reliability(pts, outdir):
    err = pts["gt_err_mm"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, f in zip(axes.ravel(), FACTORS):
        x, med, q1, q3 = decile_curve(pts[f], err)
        ax.fill_between(x, q1, q3, color=CB[f], alpha=0.15, linewidth=0)
        ax.plot(x, med, "-o", color=CB[f], markersize=5, linewidth=2,
                label="median error")
        ax.set_title(FACTOR_LABEL[f])
        ax.set_xlabel("predicted-reliability decile  (1 = least reliable)")
        ax.set_ylabel("true error to GT [mm]")
        # trend annotation
        rho, _ = spearmanr(pts[f], err)
        good = "informative" if rho < -0.1 else ("inverted!" if rho > 0.1 else "flat")
        ax.text(0.03, 0.95, f"Spearman rho = {rho:+.3f}  ({good})",
                transform=ax.transAxes, va="top", fontsize=9, color=CB[f])
    fig.suptitle("L0 reliability: does higher predicted confidence mean lower true error?\n"
                 "(a useful signal slopes DOWN to the right; IQR band shaded)",
                 fontsize=12, fontweight="bold")
    p = os.path.join(outdir, "L0_reliability_diagram.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_depth_calibration(pts, outdir):
    """Empirical error vs camera depth, against the sigma_Z ~ Z^2 model the depth
    confidence assumes. If the model is right, error rises ~quadratically with Z."""
    z = pts["cam_depth"]
    err = pts["gt_err_mm"]
    m = np.isfinite(z) & (z > 0) & np.isfinite(err)
    z, err = z[m], err[m]
    order = np.argsort(z)
    parts = np.array_split(order, 15)
    zc, ec = [], []
    for idx in parts:
        if len(idx):
            zc.append(np.median(z[idx]))
            ec.append(np.median(err[idx]))
    zc, ec = np.array(zc), np.array(ec)

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    ax.plot(zc, ec, "-o", color=CB["c_depth"], linewidth=2, markersize=5,
            label="empirical median error")
    # sigma_Z ~ Z^2 reference, scaled to the first bin
    if len(zc) and ec[0] > 0:
        ref = ec[0] * (zc / zc[0]) ** 2
        ax.plot(zc, ref, "--", color=CB["accent"], linewidth=1.8,
                label=r"$\sigma_Z \propto Z^2$ model (scaled)")
    ax.set_xlabel("camera-frame depth Z [mm]")
    ax.set_ylabel("true error to GT [mm]")
    ax.set_title("L0 depth-uncertainty calibration")
    ax.legend()
    p = os.path.join(outdir, "L0_depth_calibration.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_auc(pts, outdir, thresholds):
    err = pts["gt_err_mm"]
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    width = 0.8 / len(FACTORS)
    xbase = np.arange(len(thresholds))
    table = {}
    for fi, f in enumerate(FACTORS):
        aucs = [auc_reliability(pts[f], err, t)[0] for t in thresholds]
        table[f] = aucs
        ax.bar(xbase + fi * width, aucs, width, color=CB[f], label=FACTOR_LABEL[f])
    ax.axhline(0.5, color=CB["muted"], linestyle="--", linewidth=1)
    ax.text(len(thresholds) - 0.5, 0.505, "0.5 = useless", color=CB["muted"],
            fontsize=8, ha="right", va="bottom")
    ax.set_xticks(xbase + 0.4 - width / 2)
    ax.set_xticklabels([f"err > {t:g} mm" for t in thresholds])
    ax.set_ylabel("AUC: confidence separates good from bad points")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("L0 outlier-detection power of each confidence signal")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    p = os.path.join(outdir, "L0_outlier_auc.png")
    fig.savefig(p)
    plt.close(fig)
    return p, table


# ----------------------------------------------------------------------------- #
# L1 figures
# ----------------------------------------------------------------------------- #
def fig_distributions(pts, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, f in zip(axes.ravel(), FACTORS):
        a = pts[f]
        a = a[np.isfinite(a)]
        logx = f in ("w", "c_depth") and np.all(a > 0)
        if logx:
            bins = np.logspace(np.log10(max(a.min(), 1e-6)), np.log10(a.max()), 60)
            ax.set_xscale("log")
        else:
            bins = 60
        ax.hist(a, bins=bins, color=CB[f], alpha=0.85)
        ax.axvline(np.median(a), color=CB["accent"], linestyle="--", linewidth=1.5)
        ax.set_title(f"{FACTOR_LABEL[f]}   (CoV={cov(a):.2f}, p99/p1={dyn_range(a):.1f})")
        ax.set_xlabel(FACTOR_LABEL[f])
        ax.set_ylabel("point count")
    fig.suptitle("L1 effect size: how much does each confidence signal vary?\n"
                 "(low CoV / dynamic range near 1 => little leverage on a rigid fit; "
                 "dashed = median)", fontsize=12, fontweight="bold")
    p = os.path.join(outdir, "L1_distributions.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_spatial_leverage(pts, meta, outdir):
    """Per pair: distance between the centroid of the top-decile-weight points and
    the bottom-decile-weight points, normalized by the cloud's RMS radius. Small =>
    high/low confidence points are co-located => reweighting barely moves the rigid
    pose (link-2 leverage)."""
    xyz = np.column_stack([pts["wx"], pts["wy"], pts["wz"]])
    seps, labels = [], []
    rms_by_id = {int(r["pair_id"]): float(r["rms_radius_mm"]) for r in meta}
    for pid in np.unique(pts["pair_id"]).astype(int):
        m = pts["pair_id"] == pid
        w = pts["w"][m]
        P = xyz[m]
        if len(w) < 100:
            continue
        lo = w <= np.percentile(w, 10)
        hi = w >= np.percentile(w, 90)
        d = np.linalg.norm(P[hi].mean(0) - P[lo].mean(0))
        rms = rms_by_id.get(pid, np.nan)
        seps.append(d / rms if rms and np.isfinite(rms) and rms > 0 else np.nan)
        lo_r, hi_r = int(pts["left"][m][0]), int(pts["right"][m][0])
        labels.append(f"{lo_r}-{hi_r}")

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    x = np.arange(len(seps))
    ax.bar(x, seps, color=CB["c_depth"])
    ax.axhline(np.nanmean(seps), color=CB["accent"], linestyle="--", linewidth=1.5,
               label=f"mean = {np.nanmean(seps):.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("centroid separation / RMS radius")
    ax.set_title("L1 spatial leverage: are high- and low-confidence points co-located?\n"
                 "(small => low leverage on a rigid pose)")
    ax.legend()
    p = os.path.join(outdir, "L1_spatial_leverage.png")
    fig.savefig(p)
    plt.close(fig)
    return p, float(np.nanmean(seps))


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="dir with confidence_*.csv")
    ap.add_argument("--out", dest="outdir", default=None)
    ap.add_argument("--err-thresholds", type=float, nargs="+", default=[1.0, 2.0, 5.0],
                    help="mm thresholds defining an 'outlier' point for the AUC")
    args = ap.parse_args()
    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)
    style()

    pts, meta = load(args.indir)
    err = pts["gt_err_mm"]
    npts = len(err)

    # ---- headline numbers ----
    print(f"\nLoaded {npts:,} points across {len(np.unique(pts['pair_id']))} pairs.")
    print(f"Global confidence per cloud: {np.unique(meta['global_confidence'])}"
          "  (constant per cloud => structurally inert for single-pair ICP)")
    print(f"True error to GT [mm]: median={np.median(err):.3f}, "
          f"p90={np.percentile(err, 90):.3f}, max={err.max():.1f}\n")

    print("L0  Spearman rho(score, true_error)  [want strongly NEGATIVE]")
    rho_row = {}
    for f in FACTORS:
        rho, _ = spearmanr(pts[f], err)
        rho_row[f] = rho
        print(f"     {FACTOR_LABEL[f]:<14} rho = {rho:+.4f}")

    r1 = fig_reliability(pts, outdir)
    r2 = fig_depth_calibration(pts, outdir)
    r3, auc_table = fig_auc(pts, outdir, args.err_thresholds)
    print("\nL0  outlier-detection AUC  [0.5 = useless, ->1 = perfect]")
    hdr = "     thresholds:  " + "  ".join(f"{t:g}mm" for t in args.err_thresholds)
    print(hdr)
    for f in FACTORS:
        print(f"     {FACTOR_LABEL[f]:<14}" + "  ".join(f"{a:.3f}" for a in auc_table[f]))

    d1 = fig_distributions(pts, outdir)
    d2, mean_sep = fig_spatial_leverage(pts, meta, outdir)
    print("\nL1  effect size / leverage")
    for f in FACTORS:
        a = pts[f][np.isfinite(pts[f])]
        print(f"     {FACTOR_LABEL[f]:<14} CoV={cov(a):.3f}  p99/p1={dyn_range(a):8.2f}  "
              f"top10%_mass={top_mass(a):.3f}")
    print(f"     mean spatial separation (high vs low w) / RMS radius = {mean_sep:.3f}")

    # ---- machine-readable summary ----
    summ = os.path.join(outdir, "confidence_summary.csv")
    with open(summ, "w") as fh:
        fh.write("factor,spearman_rho,cov,dyn_range_p99_p1,top10pct_mass,"
                 + ",".join(f"auc_gt{t:g}mm" for t in args.err_thresholds) + "\n")
        for f in FACTORS:
            a = pts[f][np.isfinite(pts[f])]
            fh.write(f"{f},{rho_row[f]:.4f},{cov(a):.4f},{dyn_range(a):.4f},"
                     f"{top_mass(a):.4f},"
                     + ",".join(f"{v:.4f}" for v in auc_table[f]) + "\n")

    # ---- verdict ----
    print("\n=== VERDICT (diagnostic core) ===")
    best = min(FACTORS, key=lambda f: rho_row[f])  # most negative
    if rho_row["w"] > -0.1:
        print(" L0: w is NOT informative (|rho| < 0.1). Confidence-weighted ICP cannot")
        print("     help on this data for the RIGHT reason -- the weight does not rank")
        print("     points by error. This is itself the finding; the null benchmark is")
        print("     explained without invoking ICP.")
    else:
        print(f" L0: w ranks points by error (rho={rho_row['w']:+.3f}). Strongest single")
        print(f"     factor: {FACTOR_LABEL[best]} (rho={rho_row[best]:+.3f}).")
    print(f" Figures + confidence_summary.csv written to {outdir}")
    for p in (r1, r2, r3, d1, d2):
        print("   -", p)


if __name__ == "__main__":
    main()
