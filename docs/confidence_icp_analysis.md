# Confidence-weighted ICP: analysis pipeline

This documents the pipeline for analysing **the effect of the per-point confidence
value on ICP** (Track C, re-scoped 2026-07-13: the deliverable is the
weighted-vs-uniform ICP ablation, not the fused surface). It replaces the vague
question "does confidence-weighted ICP help?" with a decomposition that says
*which link fails*.

## Why a decomposition, not just an ablation table

"Does confidence-weighted ICP help?" factors into three **independent** links. The
existing end-to-end benchmark (`benchmark/IcpBenchmark.cpp`) only measures the
last one, so its null/negative result is uninterpretable on its own:

1. **Informative?** — does per-point `w` actually correlate with per-point *true*
   error? If not, weighting by it cannot help, regardless of the optimizer.
2. **Leverage?** — even if informative, does the weight vary enough, and are
   high/low-weight points spatially separated enough, to move a *rigid* 6-DOF
   pose? (A constant weight is inert; co-located weights barely rotate the fit.)
3. **Benefit?** — measured against an *independent* ruler (the DTU GT scan), does
   weighted beat uniform?

The confidence weight (built in `PlyUtils::buildPointCloud`) is
`w = c_global · c_depth · c_edge · c_stereo`, where:
- `c_global` — per-cloud F-inlier ratio. **Structurally inert for a single
  registration**: a constant multiplier on every residual leaves the least-squares
  argmin unchanged. It can only matter in fusion or under a robust kernel — so it
  is out of scope for the per-pair ablation (and empirically `≈ 1.0` on DTU).
- `c_depth` — median-normalized inverse depth variance (`σ_Z ∝ Z²`), capped 100:1.
- `c_edge` — `exp(-|∇disparity|/5)`, suppresses depth discontinuities.
- `c_stereo` — left/right + photometric reliability in `[0,1]`.

## Layers

| Layer | Question | Status |
|---|---|---|
| **L0** calibration | Is `w` (and each factor) informative? | **built** (this pipeline) |
| **L1** effect size | Does `w` have variance + spatial leverage? | **built** (this pipeline) |
| L2 pose sensitivity | Does weighting change the pose (self-registration)? | planned |
| L3 real-data ablation | Weighted vs uniform vs calibration-only, many seeds, CIs | exists, needs hardening (`IcpBenchmark.cpp`) |
| **L4** controlled corruption | Oracle vs real confidence under injected corruption | **built** (`IcpConfidenceCorruption`) |

This document covers the **diagnostic core (L0 + L1)** — the cheap, ICP-free
gate — and **L4**, the mechanism test that isolates *mechanism* from *signal*.

## Running it

```bash
# 1. build (adds the ConfidenceAnalysis target)
cmake --build build --target ConfidenceAnalysis -j4

# 2. dump per-point (w, factors, camera depth, true GT error) + per-pair metadata.
#    Uses the production pair selection from config.yaml (icp_pair_mode / view range)
#    and the production stereo stack. GT is used ONLY as an evaluation ruler.
cd build && ./ConfidenceAnalysis ../config.yaml --out ../results/confidence --max-points 50000

# 3. analyse (numpy/scipy/matplotlib only — no pandas)
cd .. && python3 scripts/analyze_confidence.py --in results/confidence
```

### Composing the weight from a subset of factors

Which factors compose `w` is configurable (`config.yaml`), so the ablation can
switch factors in and out. A disabled factor contributes a neutral `1.0`:

```yaml
conf_use_global: 1   # per-cloud constant -> inert for single-pair ICP
conf_use_depth:  1
conf_use_edge:   1
conf_use_stereo: 1
```

`config.confidenceWeights()` feeds these to `buildPointCloud`, so `IcpFusion`,
`IcpBenchmark`, and `ConfidenceAnalysis` all honor them consistently.
`ConfidenceAnalysis` also takes `--weight-mode {config|full|depth_only}` to
override the YAML for a one-command A/B (e.g. `--weight-mode depth_only` sets
`w = c_depth`).

Outputs in `results/confidence/`:
- `confidence_points.csv` — one row per sub-sampled point:
  `pair_id,left,right,wx,wy,wz,cam_depth,w,c_depth,c_edge,c_stereo,valid_normal,gt_err_mm`
- `confidence_pairs.csv` — per-pair metadata (global confidence, point counts,
  10%-cull threshold, RMS radius).
- `confidence_summary.csv` — machine-readable L0/L1 metrics per factor.
- `L0_reliability_diagram.png`, `L0_depth_calibration.png`, `L0_outlier_auc.png`,
  `L1_distributions.png`, `L1_spatial_leverage.png`.

### What each metric means

- **Reliability diagram** — points binned into deciles of ascending predicted
  reliability; median (± IQR) true error per bin. A useful signal slopes **down**.
- **Spearman ρ(score, error)** — rank correlation; want strongly **negative**.
- **Outlier-detection AUC** — how well the score separates good (`err ≤ τ`) from
  bad points; `0.5` = useless, `→1` = perfect.
- **Depth calibration** — empirical error vs camera depth against the `σ_Z ∝ Z²`
  model the depth factor assumes.
- **CoV / dynamic range / top-10% mass** — is there enough spread for leverage.
- **Spatial leverage** — centroid separation of top- vs bottom-decile-weight
  points ÷ RMS radius; small ⇒ co-located ⇒ little leverage on a rigid pose.

## Initial finding (views 6–19, scan 1, 12 pairs, 600k points)

**The confidence weight `w` is not informative, for a specific and fixable reason.**

| signal | Spearman ρ(·, error) | AUC (err>2mm) | reading |
|---|---:|---:|---|
| `w` (product) | **−0.018** | 0.53 | flat — does not rank points by error |
| `c_depth` | **−0.125** | 0.59 | the only factor in the right direction (weak but real) |
| `c_edge` | +0.047 | 0.46 | slightly **inverted** |
| `c_stereo` | +0.070 | 0.45 | slightly **inverted** |

- **L1 passes**: `w` has ample variance (CoV 0.60, p99/p1 → ∞) and its high/low
  points are well separated spatially (0.94 × RMS radius). So the weight *could*
  move a rigid pose — the mechanism is not inert and variance is not the problem.
- **L0 fails**: the weight is pointed at the wrong target. `c_depth` carries the
  only genuine signal (ρ = −0.125), but multiplying it by the uninformative /
  mildly inverted `c_edge` and `c_stereo` **dilutes it to ≈ 0**. The product is
  self-defeating.
- **Hypothesis for the inversion** (to confirm, not overclaim): `c_edge` and
  `c_stereo` both reward smooth, textureless, photometrically-consistent regions —
  which are exactly where stereo matching is *ambiguous* and error is high. The
  appearance cues reward the wrong pixels.

**Consequence for Track C:** the null/negative result in `IcpBenchmark.cpp` is
explained *without running ICP* — not because weighting is inert, but because the
confidence is miscalibrated. Reporting a weighted-vs-uniform table alone would
have hidden this.

### A/B confirmation: dropping the diluting factors recovers the signal

Re-running L0 with `--weight-mode depth_only` (so the composed `w` = `c_depth`,
the weight ICP actually consumes) confirms the dilution diagnosis directly:

| composed weight `w` | ρ(w, error) | AUC(err>2 mm) | CoV | spatial leverage | verdict |
|---|---:|---:|---:|---:|---|
| full product (`c_depth·c_edge·c_stereo`) | −0.018 | 0.525 | 0.60 | 0.94 | flat / uninformative |
| **depth only** (`c_depth`) | **−0.125** | **0.591** | 0.29 | 2.11 | informative |

So the *construction* is fixable: removing the two mildly-inverted appearance
factors restores the depth signal in the weight ICP sees. **But the ceiling is
low** — even clean depth-only is only weakly predictive (ρ = −0.125, AUC ≈ 0.59).
That is the honest limit of a scalar depth heuristic on clean DTU, and it is the
reason L4 (below) uses an *oracle* confidence: to test whether the ICP *mechanism*
works when the signal is good, separately from whether this signal is good enough.

### Caveats
- Stereo clouds contain off-object/background points absent from the GT scan
  (max NN error 202 mm). All L0/L1 statistics are rank- or median-based, so these
  do not distort the result — but a GT-bounding-box crop is a worthwhile
  sensitivity check.
- One camera row of one scan. Re-run across scans / pair geometries before
  generalizing.

## L4 — controlled-corruption mechanism test

L0 is correlational ("the weight doesn't rank error"). L4 makes it **causal** and
separates the two things the null benchmark conflates: is confidence-weighting a
dead *mechanism*, or a live mechanism fed a bad *signal*?

**Design** (`benchmark/IcpConfidenceCorruption.cpp`, `scripts/analyze_corruption.py`):
self-registration — perturb a copy of one cloud by a known 6° / 0.05 rigid
transform, inject known corruption into a fraction of its points, and align it
back. Ideal answer is exactly the inverse perturbation (no surface-mismatch
confound). Four weight regimes: **uniform**, **real** (full product `w`), **depth**
(`c_depth`), **oracle** (`w=0` on exactly the corrupted points = perfect
confidence). Two corruption models — **bias** (systematic shift, the failure mode
that biases a rigid fit) and **noise** (zero-mean) — and two robustness conditions:
stack **off** (weight is the only defense — pure mechanism) and stack **on**
(production reciprocal+gate+trim+Cauchy). Point-to-point residual so displacement
is fully visible. 3 pairs × 5 seeds, 95% CI.

```bash
cmake --build build --target IcpConfidenceCorruption -j4
cd build && ./IcpConfidenceCorruption ../config.yaml --out ../results/corruption --mode bias
python3 ../scripts/analyze_corruption.py --in ../results/corruption
```

**Result — stack off, translation error (normalized), systematic bias:**

| regime | 0% | 5% | 10% | 20% | 40% |
|---|---:|---:|---:|---:|---:|
| oracle  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| uniform | 0.0000 | 0.0008 | 0.0017 | 0.0037 | **0.0089** |
| depth   | 0.0000 | 0.0008 | 0.0017 | 0.0037 | **0.0088** |
| real    | 0.0000 | 0.0008 | 0.0016 | 0.0035 | **0.0083** |

Three conclusions, in order of importance:

1. **The mechanism is sound.** Oracle confidence removes systematic corruption bias
   *completely* (0.0000 at every fraction) while uniform degrades linearly — the
   weighting machinery works perfectly when the confidence is accurate. This is the
   ceiling: the most any confidence could buy.
2. **The real signal captures ~none of it.** `real` and `depth` track `uniform`
   (0.0083 / 0.0088 vs oracle 0.0000). A confidence uncorrelated with which points
   are actually bad cannot down-weight them — the causal echo of L0's ρ ≈ −0.02 /
   −0.12. The bottleneck is definitively the signal, not the optimizer.
3. **Weighting matters for biased error, not symmetric noise.** Under zero-mean
   noise the same uniform curve reaches only 0.0006 at 40% (≈15× smaller) — the
   errors self-cancel in the rigid fit, so there is little for any weighting to
   fix. Confidence weighting earns its keep against *systematic* stereo error
   (depth bias, foreground-fattening), not random jitter.

**Caveats.** The stack-*on* condition is confounded by an ICP convergence floor
(~0.66° residual even at 0% corruption, from the tight fine gate) and high
across-seed variance; only stack-*off* gives a clean read. There is a *hint* that
depth weighting aids general convergence under the robust stack (depth < uniform at
0%), but it is not a clean corruption-rejection effect and the trim+Cauchy stack
already supplies most outlier robustness. Corruption is injected at *random* points
(uncorrelated with the real confidence by construction), so this measures general
outlier protection, which the confidence does not provide — consistent with L0.

**Net Track-C statement (causal):** confidence-weighted ICP is a sound mechanism
whose benefit is capped entirely by how well the confidence identifies bad points;
the current stereo confidence realizes essentially none of that ceiling on these
clouds. Improving Track C means improving the *signal* (matching-cost/uniqueness
confidence, per-pixel covariance → GICP), not the registration math.

## What this implies for the next layers
1. ~~**Fix the weight construction before any optimizer work.**~~ **Done** (config
   toggle `conf_use_*` / `--weight-mode depth_only`): depth-only recovers the
   signal in `w` (ρ −0.018 → −0.125), but the ceiling is low. Still open: decide
   whether `c_edge`/`c_stereo` are worth *inverting* rather than dropping, and
   replace the fixed `σ_d` with a matching-cost/uniqueness confidence.
2. **L3 (hardened ablation)** should include a **calibration-only baseline** — the
   2026-07-13 finding shows ICP can *worsen* an already-good placement, so
   "weighted vs uniform" is comparing two flavors of a possibly-unnecessary step.
3. **L4 (controlled corruption)** with an **oracle confidence** (weight = 0 on
   known-corrupted points) isolates whether the *mechanism* works when the signal
   is good — decoupling "our stereo confidence is bad" from "weighting doesn't
   help." The oracle-vs-real gap is the key number.
