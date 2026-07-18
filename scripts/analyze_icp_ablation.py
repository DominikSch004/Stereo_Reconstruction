#!/usr/bin/env python3
"""Analyze and visualize the paired confidence-weighted ICP ablation.

Inputs are produced by the ConfidenceIcpAblation C++ target.  The script uses
only NumPy, SciPy, and Matplotlib and writes print-ready vector figures (PDF),
raster previews (PNG), LaTeX tables, and machine-readable CSV summaries.

Figures are sized for a two-column paper (columnwidth ~3.34 in, textwidth
~7.0 in) so they can be included at their native size without scaling.

Usage:
  python3 scripts/analyze_icp_ablation.py --in results/icp_weight_ablation
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from scipy.stats import rankdata, spearmanr, wilcoxon


# --------------------------------------------------------------------------
# Naming, grouping, and color assignments
# --------------------------------------------------------------------------

# Rows of the paired forest plot, top to bottom.  Negative controls are
# recipes that are mathematically identical to their reference (a constant
# weight cannot change the mean-normalized least-squares minimizer); they
# measure the pipeline's numerical noise floor, not a confidence effect.
FOREST_GROUPS = [
    ("Single cue  (vs. uniform)", ["depth_only", "edge_only", "stereo_only"]),
    ("All cues  (vs. uniform)", ["full"]),
    ("Leave one out  (vs. full)", ["no_depth", "no_edge", "no_stereo"]),
    ("Negative controls", ["global_only", "no_global"]),
]

CORE = ["uniform", "depth_only", "edge_only", "stereo_only", "full"]

ORDER = [
    "uniform",
    "global_only",
    "depth_only",
    "edge_only",
    "stereo_only",
    "full",
    "no_global",
    "no_depth",
    "no_edge",
    "no_stereo",
]

LABEL = {
    "uniform": "Uniform",
    "global_only": "Global only",
    "depth_only": "Depth only",
    "edge_only": "Edge only",
    "stereo_only": "Stereo only",
    "full": "Full product",
    "no_global": "Full − global",
    "no_depth": "Full − depth",
    "no_edge": "Full − edge",
    "no_stereo": "Full − stereo",
}

TEX_LABEL = {
    "uniform": "Uniform",
    "global_only": "Global only",
    "depth_only": "Depth only",
    "edge_only": "Edge only",
    "stereo_only": "Stereo only",
    "full": "Full product",
    "no_global": "Full $-$ global",
    "no_depth": "Full $-$ depth",
    "no_edge": "Full $-$ edge",
    "no_stereo": "Full $-$ stereo",
}

# Okabe-Ito colors keyed by the cue a row concerns; identity is always also
# carried by row labels, legends, or marker shape (never color alone).
COLOR = {
    "uniform": "#4D4D4D",
    "global_only": "#767676",
    "depth_only": "#0072B2",
    "edge_only": "#D55E00",
    "stereo_only": "#009E73",
    "full": "#CC79A7",
    "no_global": "#767676",
    "no_depth": "#0072B2",
    "no_edge": "#D55E00",
    "no_stereo": "#009E73",
}

MARKER = {
    "uniform": "o",
    "depth_only": "s",
    "edge_only": "^",
    "stereo_only": "D",
    "full": "v",
}

INK = "#1A1A1A"
MUTED = "#666666"
FAINT = "#AAAAAA"

CONTROLS = {"global_only", "no_global"}

METRICS = {
    "rot_err_deg": {"unit": "deg", "axis": "rotation error [deg]"},
    "trans_err_mm": {"unit": "mm", "axis": "translation error [mm]"},
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------
# Loading and statistics
# --------------------------------------------------------------------------


def load_rows(path: str) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(rows)
    if rows.size == 0:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def block_keys(rows: np.ndarray) -> list[tuple]:
    return [
        (int(r["case"]), int(r["trial"]), float(r["perturb_deg"]), float(r["perturb_trans_mm"]))
        for r in rows
    ]


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, repeats: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    if values.size == 1:
        return estimate, estimate, estimate
    draws = rng.integers(0, values.size, size=(repeats, values.size))
    estimates = np.median(values[draws], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def reference_for_variant(variant: str) -> str:
    """Uniform for additions; full product for leave-one-out effects."""
    return "full" if variant.startswith("no_") else "uniform"


def paired_deltas(rows: np.ndarray, variant: str, metric: str) -> np.ndarray:
    """variant - reference on identical (case, trial, perturbation) blocks."""
    reference = reference_for_variant(variant)
    ref_rows = rows[rows["variant"] == reference]
    var_rows = rows[rows["variant"] == variant]
    baseline = dict(zip(block_keys(ref_rows), np.asarray(ref_rows[metric], dtype=float)))
    deltas = [
        float(r[metric]) - baseline[key]
        for key, r in zip(block_keys(var_rows), var_rows)
        if key in baseline
    ]
    if len(deltas) != len(var_rows):
        raise RuntimeError(f"Unpaired blocks for {variant}; the CSV is incomplete.")
    return np.asarray(deltas, dtype=float)


def wilcoxon_p(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test; NaN when all deltas are zero."""
    nonzero = deltas[deltas != 0.0]
    if nonzero.size < 5:
        return float("nan")
    try:
        return float(wilcoxon(nonzero, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def available_variants(rows: np.ndarray) -> list[str]:
    present = set(np.asarray(rows["variant"], dtype=str).tolist())
    return [variant for variant in ORDER if variant in present]


def assemble_stats(
    rows: np.ndarray, rng: np.random.Generator, repeats: int
) -> dict[str, dict]:
    """One record per variant with absolute and paired statistics."""
    stats: dict[str, dict] = {}
    for variant in available_variants(rows):
        selected = rows[rows["variant"] == variant]
        record: dict = {
            "n": int(len(selected)),
            "reference": None if variant == "uniform" else reference_for_variant(variant),
            "ess": float(np.median(selected["weight_ess"])),
        }
        for metric in METRICS:
            med, lo, hi = bootstrap_ci(np.asarray(selected[metric], dtype=float), rng, repeats)
            record[metric] = {"med": med, "lo": lo, "hi": hi}
            if variant == "uniform":
                record[f"d_{metric}"] = None
            else:
                deltas = paired_deltas(rows, variant, metric)
                dmed, dlo, dhi = bootstrap_ci(deltas, rng, repeats)
                record[f"d_{metric}"] = {
                    "values": deltas,
                    "med": dmed,
                    "lo": dlo,
                    "hi": dhi,
                    "pct_worse": float(np.mean(deltas > 0.0) * 100.0),
                    "p": wilcoxon_p(deltas),
                    "n_zero": int(np.sum(deltas == 0.0)),
                }
        stats[variant] = record
    return stats


def perturbation_magnitudes(rows: np.ndarray) -> list[tuple[float, float]]:
    return sorted(
        set(zip(rows["perturb_deg"].astype(float), rows["perturb_trans_mm"].astype(float)))
    )


def select_magnitude(rows: np.ndarray, magnitude: tuple[float, float]) -> np.ndarray:
    rotation, translation = magnitude
    return rows[
        np.isclose(rows["perturb_deg"], rotation)
        & np.isclose(rows["perturb_trans_mm"], translation)
    ]


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------


def fmt(value: float, decimals: int, sign: bool = False) -> str:
    if not np.isfinite(value):
        return "--"
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    # Avoid the confusing "-0.000" / "+0.000" after rounding.
    if float(text) == 0.0:
        text = f"{0.0:+.{decimals}f}" if sign else f"{0.0:.{decimals}f}"
    return text


def fmt_ci(med: float, lo: float, hi: float, decimals: int, sign: bool = False) -> str:
    return (
        f"{fmt(med, decimals, sign)} "
        f"[{fmt(lo, decimals, sign)}, {fmt(hi, decimals, sign)}]"
    )


def tex_ci(med: float, lo: float, hi: float, decimals: int, sign: bool = False) -> str:
    """Math mode per number, text mode between: spaces survive LaTeX."""
    return (
        f"${fmt(med, decimals, sign)}$ "
        f"[${fmt(lo, decimals, sign)}$, ${fmt(hi, decimals, sign)}$]"
    )


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# --------------------------------------------------------------------------
# Figure 1: paired marginal effects (main result)
# --------------------------------------------------------------------------


def fig_forest(stats: dict[str, dict], output_dir: str, rng: np.random.Generator) -> None:
    layout: list[tuple[str, str | None]] = []  # (kind, payload) per y slot
    for header, members in FOREST_GROUPS:
        present = [m for m in members if m in stats]
        if not present:
            continue
        layout.append(("header", header))
        for member in present:
            layout.append(("row", member))

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 0.34 * len(layout) + 0.9), sharey=True, constrained_layout=True
    )
    panel_meta = [
        ("(a)", "d_rot_err_deg", "Δ rotation error vs. reference [deg]", 3),
        ("(b)", "d_trans_err_mm", "Δ translation error vs. reference [mm]", 3),
    ]

    for axis, (letter, key, xlabel, decimals) in zip(axes, panel_meta):
        # Symmetric data window from robust quantiles; annotation gutter right.
        all_deltas = np.concatenate(
            [stats[v][key]["values"] for _, v in layout if _ == "row" and stats[v][key]]
        )
        ci_extent = max(
            max(abs(stats[v][key]["hi"]), abs(stats[v][key]["lo"]))
            for kind, v in layout
            if kind == "row"
        )
        window = float(
            max(np.quantile(np.abs(all_deltas), 0.98) * 1.15, ci_extent * 1.25, 1e-3)
        )
        clipped_total = 0

        for position, (kind, payload) in enumerate(layout):
            if kind != "row":
                continue
            variant = payload
            record = stats[variant][key]
            color = COLOR[variant]
            control = variant in CONTROLS

            values = record["values"]
            inside = np.clip(values, -window, window)
            clipped = np.abs(values) > window
            clipped_total += int(np.sum(clipped))
            jitter = rng.uniform(-0.17, 0.17, size=values.size)
            axis.scatter(
                inside[~clipped],
                position + jitter[~clipped],
                s=5.5,
                color=color,
                alpha=0.30,
                linewidths=0,
                zorder=2,
            )
            for sign, marker in ((1.0, ">"), (-1.0, "<")):
                side = clipped & (np.sign(values) == sign)
                if np.any(side):
                    axis.scatter(
                        np.full(int(np.sum(side)), sign * window * 0.99),
                        position + jitter[side],
                        s=9,
                        marker=marker,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.6,
                        alpha=0.75,
                        zorder=2,
                    )

            axis.errorbar(
                record["med"],
                position,
                xerr=[[record["med"] - record["lo"]], [record["hi"] - record["med"]]],
                marker="D" if not control else "o",
                markersize=4.6 if not control else 3.8,
                markerfacecolor=color if not control else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                color=color,
                linewidth=1.3,
                capsize=2.0,
                zorder=4,
            )

            text = fmt_ci(record["med"], record["lo"], record["hi"], decimals, sign=True)
            axis.text(
                0.995,
                position,
                text,
                transform=blended_transform_factory(axis.transAxes, axis.transData),
                ha="right",
                va="center",
                fontsize=6.2,
                color=MUTED if control else INK,
                zorder=5,
            )

        axis.axvline(0.0, color=INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
        axis.set_xlim(-window, window * 2.75)
        # Keep ticks (and thus grid lines) out of the annotation gutter so the
        # text column cannot be misread as data space.
        axis.set_xticks([t for t in axis.get_xticks() if -window <= t <= window])
        axis.set_xlim(-window, window * 2.75)
        axis.set_ylim(len(layout) - 0.5, -0.7)
        axis.set_xlabel(xlabel)
        axis.set_title(letter, loc="left", fontweight="bold")
        axis.grid(axis="y", visible=False)
        axis.text(
            -window * 0.97,
            -0.62,
            "← improves",
            ha="left",
            va="center",
            fontsize=6.4,
            color=MUTED,
            fontstyle="italic",
        )
        axis.text(
            window * 0.97,
            -0.62,
            "degrades →",
            ha="right",
            va="center",
            fontsize=6.4,
            color=MUTED,
            fontstyle="italic",
        )
        if clipped_total:
            print(
                f"  forest panel {letter}: {clipped_total} of {len(all_deltas)} block deltas "
                f"exceed the +-{window:.3g} display window (drawn as open arrows at the edge)"
            )

    positions = np.arange(len(layout))
    labels = [payload if kind == "header" else LABEL[payload] for kind, payload in layout]
    axes[0].set_yticks(positions, labels)
    for tick, (kind, payload) in zip(axes[0].get_yticklabels(), layout):
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_fontsize(7.3)
        elif payload in CONTROLS:
            tick.set_color(MUTED)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)

    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"fig_icp_ablation_paired.{extension}"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: absolute recovery error by perturbation (context)
# --------------------------------------------------------------------------


def fig_absolute(
    rows: np.ndarray, output_dir: str, rng: np.random.Generator, repeats: int
) -> None:
    variants = [v for v in CORE if v in set(available_variants(rows))]
    magnitudes = perturbation_magnitudes(rows)
    x = np.arange(len(magnitudes), dtype=float)
    xlabels = [f"{r:g}° / {t:g} mm" for r, t in magnitudes]
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2.0) * 0.15

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    panel_meta = [
        ("(a)", "rot_err_deg", "coarse_rot_err_deg", "Rotation error [deg]", 0),
        ("(b)", "trans_err_mm", "coarse_trans_err_mm", "Translation error [mm]", 1),
    ]

    handles, handle_labels = [], []
    for letter, metric, coarse_metric, ylabel, perturb_column in (
        (m[0], m[1], m[2], m[3], m[4]) for m in panel_meta
    ):
        axis = axes[0] if metric == "rot_err_deg" else axes[1]

        # "No correction" reference: the error if ICP returned the identity.
        identity = [magnitude[perturb_column] for magnitude in magnitudes]
        h_id = axis.scatter(
            x, identity, marker="x", s=26, color=FAINT, linewidths=1.2, zorder=3
        )

        # Shared coarse stage (identical for every recipe inside a block).
        uniform_rows = rows[rows["variant"] == "uniform"]
        coarse_med, coarse_lo, coarse_hi = [], [], []
        for magnitude in magnitudes:
            values = np.asarray(
                select_magnitude(uniform_rows, magnitude)[coarse_metric], dtype=float
            )
            med, lo, hi = bootstrap_ci(values, rng, repeats)
            coarse_med.append(med)
            coarse_lo.append(lo)
            coarse_hi.append(hi)
        (h_coarse,) = axis.plot(
            x,
            coarse_med,
            color=INK,
            linewidth=1.1,
            linestyle=(0, (4, 2.5)),
            marker="s",
            markersize=3.4,
            markerfacecolor="white",
            zorder=3,
        )

        for variant, offset in zip(variants, offsets):
            selected = rows[rows["variant"] == variant]
            med, lo, hi = [], [], []
            for magnitude in magnitudes:
                values = np.asarray(select_magnitude(selected, magnitude)[metric], dtype=float)
                m, l, h = bootstrap_ci(values, rng, repeats)
                med.append(m)
                lo.append(m - l)
                hi.append(h - m)
            container = axis.errorbar(
                x + offset,
                med,
                yerr=np.vstack([lo, hi]),
                color=COLOR[variant],
                marker=MARKER[variant],
                markersize=3.8,
                linewidth=1.2,
                capsize=1.5,
                alpha=0.95,
                zorder=4,
            )
            if metric == "rot_err_deg":
                handles.append(container)
                handle_labels.append(LABEL[variant])

        if metric == "rot_err_deg":
            handles += [h_coarse, h_id]
            handle_labels += ["Shared coarse stage", "No correction (= perturbation)"]

        axis.set_xticks(x, xlabels)
        axis.set_xlabel("Applied perturbation")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        axis.set_title(letter, loc="left", fontweight="bold")
        axis.grid(axis="x", visible=False)

    fig.legend(
        handles,
        handle_labels,
        loc="outside upper center",
        ncol=4,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"fig_icp_ablation_absolute.{extension}"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: cue validity against independent structured-light error
# --------------------------------------------------------------------------


def auc_reliable(score: np.ndarray, error: np.ndarray, threshold_mm: float) -> float:
    finite = np.isfinite(score) & np.isfinite(error)
    score = np.asarray(score[finite], dtype=float)
    good = np.asarray(error[finite] <= threshold_mm, dtype=bool)
    positives = int(np.sum(good))
    negatives = int(np.sum(~good))
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    rank_sum = float(np.sum(ranks[good]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


CUE_SIGNALS = {
    "c_depth": ("Depth", COLOR["depth_only"]),
    "c_edge": ("Edge", COLOR["edge_only"]),
    "c_stereo": ("Stereo", COLOR["stereo_only"]),
    "w_full": ("Full product", COLOR["full"]),
}


def cue_statistics(cues: np.ndarray, auc_threshold: float) -> dict[str, dict[str, np.ndarray]]:
    """Per-pair Spearman rho and reliable-point AUC for each cue."""
    result: dict[str, dict[str, list]] = {
        signal: {"pair": [], "rho": [], "auc": []} for signal in CUE_SIGNALS
    }
    valid = cues[np.asarray(cues["inside_gt_bbox"], dtype=int) == 1]
    for pair in np.unique(valid["pair"]):
        pair_rows = valid[valid["pair"] == pair]
        error = np.asarray(pair_rows["gt_error_mm"], dtype=float)
        for signal in CUE_SIGNALS:
            score = np.asarray(pair_rows[signal], dtype=float)
            finite = np.isfinite(score) & np.isfinite(error)
            if np.sum(finite) < 3 or np.allclose(score[finite], score[finite][0]):
                rho = np.nan
            else:
                rho = float(spearmanr(score[finite], error[finite]).statistic)
            result[signal]["pair"].append(str(pair))
            result[signal]["rho"].append(rho)
            result[signal]["auc"].append(auc_reliable(score, error, auc_threshold))
    return {
        signal: {
            "pair": np.asarray(metrics["pair"]),
            "rho": np.asarray(metrics["rho"], dtype=float),
            "auc": np.asarray(metrics["auc"], dtype=float),
        }
        for signal, metrics in result.items()
    }


def fig_cues(cue_stats: dict, auc_threshold: float, output_dir: str) -> None:
    signals = list(CUE_SIGNALS)
    y = np.arange(len(signals))

    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.95), sharey=True, constrained_layout=True)
    panel_meta = [
        ("(a)", "rho", 0.0, "Spearman ρ(cue, GT error)"),
        ("(b)", "auc", 0.5, f"AUC (GT error ≤ {auc_threshold:g} mm)"),
    ]
    for axis, (letter, metric, reference, xlabel) in zip(axes, panel_meta):
        for position, signal in enumerate(signals):
            values = cue_stats[signal][metric]
            values = values[np.isfinite(values)]
            color = CUE_SIGNALS[signal][1]
            axis.scatter(
                values,
                np.full(values.shape, position),
                s=13,
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                zorder=3,
            )
            axis.plot(
                [np.median(values)] * 2,
                [position - 0.24, position + 0.24],
                color=color,
                linewidth=2.0,
                solid_capstyle="round",
                zorder=4,
            )
        axis.axvline(reference, color=INK, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
        axis.set_yticks(y, [CUE_SIGNALS[s][0] for s in signals])
        axis.set_ylim(len(signals) - 0.5, -0.5)
        axis.set_xlabel(xlabel, fontsize=7.0)
        axis.set_title(letter, loc="left", fontweight="bold")
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0)
        hint = "← desirable" if metric == "rho" else "desirable →"
        axis.text(
            0.03 if metric == "rho" else 0.97,
            0.03,
            hint,
            transform=axis.transAxes,
            ha="left" if metric == "rho" else "right",
            va="bottom",
            fontsize=6.0,
            color=MUTED,
            fontstyle="italic",
        )

    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"fig_confidence_cues.{extension}"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def delta_cell(record: dict | None, decimals: int, latex: bool) -> str:
    if record is None:
        return "--" if latex else "—"
    stars = significance_stars(record["p"])
    if latex:
        cell = tex_ci(record["med"], record["lo"], record["hi"], decimals, sign=True)
        if stars:
            cell += rf"\textsuperscript{{{stars}}}"
        return cell
    text = fmt_ci(record["med"], record["lo"], record["hi"], decimals, sign=True)
    return f"{text}{stars}"


def table_rows(stats: dict[str, dict]) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = [(None, "uniform")]
    for header, members in FOREST_GROUPS:
        present = [m for m in members if m in stats]
        if present:
            rows.append((header, None))
            rows += [(None, m) for m in present]
    return rows


def write_latex_main(stats: dict[str, dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "icp_ablation_table_main.tex")
    n = stats["uniform"]["n"]
    lines = [
        "% Auto-generated by scripts/analyze_icp_ablation.py -- do not edit.",
        "% Suggested caption: Pose-recovery error of confidence-weighted fine ICP",
        f"% on DTU (median over {n} paired blocks, 95% bootstrap CI). Deltas are",
        "% paired against the stated reference; negative is better. ESS is the",
        "% effective sample size of the normalized weights (n = 4000 source",
        "% points). Wilcoxon signed-rank: *** p<0.001, ** p<0.01, * p<0.05.",
        "% Requires \\usepackage{booktabs}. Wrap in table* for two-column layouts.",
        r"\begin{tabular}{l r c c c c}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Rotation error [deg]} & \multicolumn{2}{c}{Translation error [mm]} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Weight recipe & ESS & median [CI] & $\Delta$ vs.\ ref.\ [CI] & median [CI] & $\Delta$ vs.\ ref.\ [CI] \\",
        r"\midrule",
    ]
    for header, variant in table_rows(stats):
        if header is not None:
            lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{header.replace('  ', ' ')}}}}} \\")
            continue
        record = stats[variant]
        rot = record["rot_err_deg"]
        trans = record["trans_err_mm"]
        lines.append(
            " & ".join(
                [
                    TEX_LABEL[variant],
                    f"{record['ess']:.0f}",
                    tex_ci(rot["med"], rot["lo"], rot["hi"], 2),
                    delta_cell(record["d_rot_err_deg"], 3, latex=True),
                    tex_ci(trans["med"], trans["lo"], trans["hi"], 2),
                    delta_cell(record["d_trans_err_mm"], 3, latex=True),
                ]
            )
            + r" \\"
        )
        if variant == "uniform":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def write_latex_by_perturbation(
    rows: np.ndarray, output_dir: str, rng: np.random.Generator, repeats: int
) -> None:
    path = os.path.join(output_dir, "icp_ablation_table_by_perturbation.tex")
    magnitudes = perturbation_magnitudes(rows)
    n_blocks = None
    lines = [
        "% Auto-generated by scripts/analyze_icp_ablation.py -- do not edit.",
        "% Suggested caption: Paired change in pose-recovery error split by",
        "% perturbation magnitude (median [95% bootstrap CI]; negative is",
        "% better). Each cell aggregates the paired blocks of one magnitude.",
        "% Requires \\usepackage{booktabs}.",
        r"\begin{tabular}{l l " + " ".join(["c"] * len(magnitudes)) + "}",
        r"\toprule",
        "Weight recipe & Metric & "
        + " & ".join(rf"{r:g}$^\circ$ / {t:g}\,mm" for r, t in magnitudes)
        + r" \\",
        r"\midrule",
    ]
    for _, variant in table_rows({v: None for v in available_variants(rows)}):
        if variant in (None, "uniform") or variant in CONTROLS:
            continue
        cells = {"rot_err_deg": [], "trans_err_mm": []}
        for metric in cells:
            deltas_all = paired_deltas(rows, variant, metric)
            keys = block_keys(rows[rows["variant"] == variant])
            for magnitude in magnitudes:
                mask = np.array(
                    [np.isclose(k[2], magnitude[0]) and np.isclose(k[3], magnitude[1]) for k in keys]
                )
                n_blocks = int(np.sum(mask))
                med, lo, hi = bootstrap_ci(deltas_all[mask], rng, repeats)
                cells[metric].append(tex_ci(med, lo, hi, 3, sign=True))
        lines.append(
            f"{TEX_LABEL[variant]} & $\\Delta$rot.\\ [deg] & "
            + " & ".join(cells["rot_err_deg"])
            + r" \\"
        )
        lines.append(
            " & $\\Delta$trans.\\ [mm] & " + " & ".join(cells["trans_err_mm"]) + r" \\"
        )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", f"% {n_blocks} paired blocks per cell.", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def markdown_table(stats: dict[str, dict]) -> str:
    lines = [
        "| Weight recipe | ESS | Rot. err. [deg] | Δ rot vs. ref [deg] | Trans. err. [mm] | Δ trans vs. ref [mm] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for header, variant in table_rows(stats):
        if header is not None:
            lines.append(f"| **{header.replace('  ', ' ')}** | | | | | |")
            continue
        record = stats[variant]
        rot = record["rot_err_deg"]
        trans = record["trans_err_mm"]
        lines.append(
            "| "
            + " | ".join(
                [
                    LABEL[variant],
                    f"{record['ess']:.0f}",
                    fmt_ci(rot["med"], rot["lo"], rot["hi"], 2),
                    delta_cell(record["d_rot_err_deg"], 3, latex=False),
                    fmt_ci(trans["med"], trans["lo"], trans["hi"], 2),
                    delta_cell(record["d_trans_err_mm"], 3, latex=False),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CSV summaries
# --------------------------------------------------------------------------


def save_summary_csv(stats: dict[str, dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "icp_ablation_summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["variant", "paired_reference", "n", "weight_ess_median"]
        for metric in METRICS:
            header += [
                f"{metric}_median",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
                f"{metric}_delta_median",
                f"{metric}_delta_ci_low",
                f"{metric}_delta_ci_high",
                f"{metric}_delta_pct_worse",
                f"{metric}_delta_wilcoxon_p",
                f"{metric}_delta_n_exact_zero",
            ]
        writer.writerow(header)
        for variant, record in stats.items():
            row = [
                variant,
                record["reference"] or "none",
                record["n"],
                record["ess"],
            ]
            for metric in METRICS:
                absolute = record[metric]
                delta = record[f"d_{metric}"]
                row += [absolute["med"], absolute["lo"], absolute["hi"]]
                if delta is None:
                    row += [0.0, 0.0, 0.0, np.nan, np.nan, record["n"]]
                else:
                    row += [
                        delta["med"],
                        delta["lo"],
                        delta["hi"],
                        delta["pct_worse"],
                        delta["p"],
                        delta["n_zero"],
                    ]
            writer.writerow(row)


def save_by_perturbation_csv(
    rows: np.ndarray, output_dir: str, rng: np.random.Generator, repeats: int
) -> None:
    path = os.path.join(output_dir, "icp_ablation_by_perturbation.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "perturb_deg",
                "perturb_trans_mm",
                "n",
                "rot_err_deg_median",
                "rot_err_deg_ci_low",
                "rot_err_deg_ci_high",
                "trans_err_mm_median",
                "trans_err_mm_ci_low",
                "trans_err_mm_ci_high",
                "rot_delta_median",
                "trans_delta_median",
            ]
        )
        for variant in available_variants(rows):
            selected = rows[rows["variant"] == variant]
            keys = block_keys(selected)
            deltas = {
                metric: (
                    paired_deltas(rows, variant, metric) if variant != "uniform" else None
                )
                for metric in METRICS
            }
            for magnitude in perturbation_magnitudes(rows):
                subset = select_magnitude(selected, magnitude)
                mask = np.array(
                    [np.isclose(k[2], magnitude[0]) and np.isclose(k[3], magnitude[1]) for k in keys]
                )
                record = [variant, magnitude[0], magnitude[1], len(subset)]
                for metric in METRICS:
                    record += list(
                        bootstrap_ci(np.asarray(subset[metric], dtype=float), rng, repeats)
                    )
                for metric in METRICS:
                    record.append(
                        float(np.median(deltas[metric][mask])) if deltas[metric] is not None else 0.0
                    )
                writer.writerow(record)


def save_cue_csv(cue_stats: dict, auc_threshold: float, output_dir: str) -> None:
    path = os.path.join(output_dir, "confidence_cue_summary.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal", "pair", "spearman_rho", f"auc_{auc_threshold:g}mm"])
        for signal, metrics in cue_stats.items():
            for pair, rho, auc in zip(metrics["pair"], metrics["rho"], metrics["auc"]):
                writer.writerow([signal, pair, rho, auc])
            writer.writerow(
                [
                    signal,
                    "MEDIAN",
                    float(np.nanmedian(metrics["rho"])),
                    float(np.nanmedian(metrics["auc"])),
                ]
            )


# --------------------------------------------------------------------------
# Console report
# --------------------------------------------------------------------------


def console_report(stats: dict[str, dict], cue_stats: dict, auc_threshold: float) -> None:
    print("\nPaired effects (median [95% CI], Wilcoxon p, % blocks worse):")
    for variant, record in stats.items():
        delta = record["d_rot_err_deg"]
        if delta is None:
            continue
        tag = "  (control)" if variant in CONTROLS else ""
        print(
            f"  {LABEL[variant]:14s} vs {record['reference']:8s}"
            f"  rot {fmt_ci(delta['med'], delta['lo'], delta['hi'], 3, sign=True):>28s}"
            f"  p={delta['p']:.2g}  worse {delta['pct_worse']:3.0f}%{tag}"
        )
    print("\nCue validity (per-pair median):")
    for signal, metrics in cue_stats.items():
        print(
            f"  {CUE_SIGNALS[signal][0]:14s} rho={np.nanmedian(metrics['rho']):+.3f}"
            f"  AUC({auc_threshold:g}mm)={np.nanmedian(metrics['auc']):.3f}"
            f"  (n={int(np.sum(np.isfinite(metrics['rho'])))} pairs)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_dir", required=True)
    parser.add_argument("--out", dest="output_dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--auc-threshold", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    configure_style()
    rng = np.random.default_rng(args.seed)

    rows = load_rows(os.path.join(args.input_dir, "icp_weight_ablation.csv"))
    cues = load_rows(os.path.join(args.input_dir, "confidence_cues.csv"))

    modes = sorted(set(np.asarray(rows["icp_mode"], dtype=str)))
    if len(modes) > 1:
        raise RuntimeError(f"Mixed ICP modes in one CSV: {modes}; analyze runs separately.")

    stats = assemble_stats(rows, rng, args.bootstrap)
    cue_stats = cue_statistics(cues, args.auc_threshold)

    fig_forest(stats, output_dir, rng)
    fig_absolute(rows, output_dir, rng, args.bootstrap)
    fig_cues(cue_stats, args.auc_threshold, output_dir)

    write_latex_main(stats, output_dir)
    write_latex_by_perturbation(rows, output_dir, rng, args.bootstrap)
    save_summary_csv(stats, output_dir)
    save_by_perturbation_csv(rows, output_dir, rng, args.bootstrap)
    save_cue_csv(cue_stats, args.auc_threshold, output_dir)

    markdown_path = os.path.join(output_dir, "icp_ablation_table.md")
    with open(markdown_path, "w") as handle:
        handle.write(markdown_table(stats) + "\n")

    console_report(stats, cue_stats, args.auc_threshold)

    n_blocks = len(set(block_keys(rows)))
    print(
        f"\nAnalyzed {len(rows)} ICP runs in {n_blocks} paired blocks "
        f"(icp_mode={modes[0]}). Outputs written to {output_dir}:"
    )
    for name in sorted(os.listdir(output_dir)):
        if name.startswith(("fig_", "icp_ablation_table")):
            print(f"  {name}")


if __name__ == "__main__":
    main()
