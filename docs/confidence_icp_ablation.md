# Confidence-weighted ICP ablation

## Question and weight definition

The experiment measures whether the individual terms in the source-point ICP
weight improve rigid registration:

\[
w_i = c_{\mathrm{global}}\;c_{\mathrm{depth},i}\;
      c_{\mathrm{edge},i}\;c_{\mathrm{stereo},i}.
\]

- `c_global`: sparse geometric inlier ratio for the stereo pair. It is constant
  within one cloud.
- `c_depth`: relative inverse depth variance from
  \(\sigma_Z^2 = Z^4\sigma_d^2/(fB)^2\), normalized by the median valid depth
  variance and capped at 100.
- `c_edge`: `exp(-|grad disparity| / 5)`, intended to suppress depth
  discontinuities.
- `c_stereo`: soft left/right-cycle and photometric consistency after a common
  hard left/right validation step.

The ICP residual uses `sqrt(w_i)`, so Ceres minimizes
\(\sum_i w_i\lVert r_i\rVert^2\). Each ablation recipe is mean-normalized before
optimization. This removes arbitrary global scale while retaining the relative
per-point weighting.

`c_global` cannot change the minimizer of the current pairwise ICP problem: it
multiplies every source residual by the same scalar. Consequently,
`global_only == uniform` and `no_global == full` are expected sanity checks, not
empirical hypotheses.

## Experimental design

`ConfidenceIcpAblation` reconstructs overlapping DTU stereo pairs and places
them in the calibrated DTU world frame. For every registration case and random
trial it:

1. draws one target, one coarse source sample, and one fine source sample;
2. applies a known rigid perturbation to both source samples;
3. computes one shared **unweighted coarse ICP** result;
4. starts all fine-stage recipes from that exact result;
5. changes only the fine source weights; and
6. evaluates the recovered transform against the known perturbation.

This is a paired design: geometry, sampling, perturbation, initialization,
solver settings, and target are identical within a trial. The statistical unit
is a `(case, trial, perturbation)` block—not an individual point.

The fusion pipeline's separate confidence cull is intentionally not applied in
this experiment. Culling changes the point set and correspondence graph; keeping
it out isolates the effect of residual weighting itself. A culling study should
be reported as a separate ablation.

The default recipes are:

- uniform;
- global, depth, edge, and stereo individually;
- the full product; and
- the full product with each term removed in turn.

Primary endpoints are rotation and translation recovery error of
`T_fine T_coarse T_perturb`. Secondary endpoints are unweighted source-to-target
nearest-neighbor distance, distance to the independent DTU structured-light
scan, correspondence count, runtime, and effective sample size of the weights.

The benchmark also exports a point-level diagnostic. Each cue is compared with
nearest-neighbor error to the structured-light scan for points inside the scan
bounding box. The plot reports per-pair Spearman correlation and reliable-point
ROC-AUC; aggregation happens across pairs to avoid treating thousands of
spatially correlated pixels as independent trials.

## Reproduce the paper experiment

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j --target ConfidenceIcpAblation

cd build
./ConfidenceIcpAblation ../config.yaml \
  --out ../results/icp_weight_ablation \
  --cases 3 --trials 5 \
  --source-samples 4000 --coarse-samples 2000 --target-samples 12000 \
  --cue-samples 20000 \
  --perturbations 1:5,3:15,5:25 \
  --icp-mode point_to_point --seed 2026

cd ..
python3 scripts/analyze_icp_ablation.py \
  --in results/icp_weight_ablation --bootstrap 5000
```

Python dependencies are `numpy`, `scipy`, and `matplotlib`. No notebook or
external plotting code is required.

For a quick integration check:

```bash
cd build
./ConfidenceIcpAblation ../config.yaml \
  --out /tmp/icp_ablation_smoke --cases 1 --trials 1 \
  --source-samples 600 --coarse-samples 400 --target-samples 1200 \
  --cue-samples 1000 --perturbations 2:10
cd ..
python3 scripts/analyze_icp_ablation.py \
  --in /tmp/icp_ablation_smoke --bootstrap 200
```

Use `--icp-mode point_to_plane` for a residual-formulation sensitivity run.
The mode is written into every raw CSV row.

## Outputs

The C++ benchmark writes:

- `icp_weight_ablation.csv`: raw paired ICP runs;
- `confidence_cues.csv`: sampled cue values and independent GT error.

The Python analysis writes:

- `fig_icp_ablation_paired.pdf`/`.png`: **main figure** — paired per-block
  deltas as a two-panel forest plot (rotation, translation) with per-block
  dots, medians with 95% bootstrap CIs, and the negative-control rows;
- `fig_icp_ablation_absolute.pdf`/`.png`: context figure — absolute recovery
  error per perturbation with the shared coarse stage and the no-correction
  reference;
- `fig_confidence_cues.pdf`/`.png`: cue validity against the independent
  structured-light scan (per-pair values, n = 4 pairs, medians);
- `icp_ablation_table_main.tex`: booktabs main table (median [CI], paired
  deltas, Wilcoxon significance stars, effective sample size);
- `icp_ablation_table_by_perturbation.tex`: appendix table, deltas split by
  perturbation magnitude;
- `icp_ablation_table.md`: the main table as markdown;
- `icp_ablation_summary.csv`, `icp_ablation_by_perturbation.csv`,
  `confidence_cue_summary.csv`: machine-readable statistics, including
  Wilcoxon p-values, share of blocks degraded, and exact-zero counts.

Figures are sized for a two-column paper (3.34 in / 7.0 in wide) and should be
included at native size; PDFs are the vector originals, PNGs are previews.

## Interpretation rules

- A cue helps transform recovery when its paired delta is below zero and its
  bootstrap interval does not cross zero.
- The negative controls (`global_only` vs. uniform, `no_global` vs. full) are
  mathematically identical problems and empirically bound the pipeline's
  numerical noise: FLANN's randomized k-d trees give each `estimatePose` call
  slightly different approximate correspondences, and mean-normalized weights
  round to float. Do not interpret paired deltas whose magnitude is inside the
  control distribution (median |Δ| ≈ 0.002°, p90 ≈ 0.02°, isolated blocks up
  to 0.5°); rely on the across-block median, CI, and signed-rank test.
- Negative cue/error Spearman correlation and ROC-AUC above 0.5 indicate that a
  cue ranks accurate points ahead of inaccurate points.
- A lower effective sample size means the recipe concentrates the fit on fewer
  points. It is not automatically beneficial.
- Source-to-target distance is close to the ICP objective and should not replace
  transform error as the main claim. Structured-light distance is independent,
  but it also contains stereo reconstruction error and is therefore secondary.
- A flat point-to-plane ablation can mean that the fine solver returned the
  shared coarse pose, especially when noisy stereo normals make candidate steps
  fail the optimizer's unweighted best-pose check. Inspect `matches`, the coarse
  errors, and the point-to-point sensitivity run before concluding that the
  confidence signal has no effect.

## Result of the reference run

The documented command (run 2026-07-17, point-to-point, seed 2026) produced 45
paired blocks (450 fine ICP solves). Headline: **no weight recipe improves
pose recovery, and emphasizing `c_depth` significantly degrades rotation.**
All effects are rotation-specific; no recipe moves translation (all Wilcoxon
p ≥ 0.09). Median over paired blocks, 95% bootstrap CI; Wilcoxon signed-rank
stars (\*\*\* p<0.001, \*\* p<0.01, \* p<0.05); ESS = effective sample size of
the mean-normalized weights (4000 source points).

| Weight recipe | ESS | Rot. err. [deg] | Δ rot vs. ref [deg] | Trans. err. [mm] | Δ trans vs. ref [mm] |
|---|---:|---:|---:|---:|---:|
| Uniform | 4000 | 2.12 [1.14, 3.80] | — | 3.43 [1.86, 6.11] | — |
| **Single cue (vs. uniform)** | | | | | |
| Depth only | 3688 | 2.18 [1.22, 4.12] | +0.092 [+0.064, +0.163]\*\*\* | 3.38 [1.85, 6.04] | -0.002 [-0.043, +0.062] |
| Edge only | 3627 | 1.92 [1.11, 3.79] | -0.002 [-0.023, +0.010] | 3.13 [1.80, 5.87] | -0.012 [-0.025, +0.000] |
| Stereo only | 3687 | 2.15 [1.03, 3.88] | +0.037 [+0.013, +0.078]\*\*\* | 3.38 [1.91, 6.35] | +0.021 [-0.015, +0.038] |
| **All cues (vs. uniform)** | | | | | |
| Full product | 3095 | 2.35 [1.16, 4.32] | +0.046 [+0.014, +0.093]\*\* | 3.37 [1.86, 6.11] | +0.000 [-0.058, +0.043] |
| **Leave one out (vs. full)** | | | | | |
| Full − depth | 3362 | 2.14 [1.27, 3.94] | -0.037 [-0.084, +0.000]\* | 3.47 [1.75, 5.86] | +0.000 [-0.015, +0.063] |
| Full − edge | 3395 | 2.29 [1.47, 4.33] | +0.000 [-0.005, +0.007] | 3.32 [1.69, 6.05] | +0.000 [-0.031, +0.000] |
| Full − stereo | 3366 | 2.30 [1.42, 4.28] | +0.000 [-0.032, +0.010] | 3.45 [1.88, 6.52] | +0.000 [-0.010, +0.027] |
| **Negative controls** | | | | | |
| Global only | 4000 | 2.12 [1.13, 3.80] | +0.000 [+0.000, +0.000] | 3.43 [1.86, 6.08] | +0.000 [+0.000, +0.000] |
| Full − global | 3095 | 2.35 [1.51, 4.32] | +0.000 [+0.000, +0.000] | 3.37 [1.71, 6.11] | +0.000 [+0.000, +0.000] |

Reading, in order of confidence:

1. **Depth-weighting harms rotation.** +0.092° median, worse in 82% of the 45
   blocks, p = 7·10⁻⁹; mirrored by the leave-one-out (removing depth from the
   full product improves rotation, −0.037°, p = 0.036). The harm is small at
   the 1° perturbation (+0.036°) and grows at 3–5° (+0.163°/+0.092°).
   Translation is untouched (−0.002 mm, p = 0.49). This error signature fits a
   leverage explanation: `c_depth` ∝ 1/σ_Z² ∝ Z⁻⁴ concentrates weight on near
   points, shortening the lever arms that constrain rotation, while
   translation is equally constrained by any inlier subset. The ESS barely
   drops (3688/4000), so this is spatial redistribution, not sample loss.
2. **The full product inherits a diluted version of the depth harm**
   (+0.046°, p = 0.001), consistent with the L0 finding that multiplying in
   the uninformative `c_edge`/`c_stereo` dilutes `c_depth`.
3. **Appearance cues are pose-neutral.** `c_edge` — although *inverted* as a
   pointwise reliability measure (per-pair Spearman ρ = +0.13) — neither helps
   nor hurts the rigid fit (Δrot −0.002°, p = 0.26). `c_stereo` slightly hurts
   rotation (+0.037°, p = 2·10⁻⁴). Pointwise mis-calibration does not
   automatically translate into pose damage, and pointwise calibration does
   not translate into pose benefit.
4. **The negative controls behave.** Both controls have median Δ = 0 with
   16/45 and 18/45 bitwise-zero blocks; the nonzero remainder (p90 ≈ 0.02°,
   two blocks up to 0.5°) measures FLANN correspondence nondeterminism plus
   float weight rounding, and bounds what magnitude of effect is
   interpretable. The depth effect sits well above this floor; the edge and
   leave-one-out null effects are inside it and must be reported as nulls.

The independent cue diagnostic (points inside the GT bounding box, per-pair
values over 4 stereo pairs) explains the mechanism gap:

| Signal | Spearman ρ with GT error (median) | ROC-AUC, error ≤ 5 mm (median) |
|---|---:|---:|
| Depth | -0.334 | 0.664 |
| Edge | +0.131 | 0.455 |
| Stereo | +0.004 | 0.499 |
| Full product | -0.120 | 0.579 |

`c_depth` is the only cue that ranks accurate points ahead of inaccurate ones,
yet it is also the only cue that damages the pose. Together with the oracle
corruption experiment (L4: a perfect confidence removes systematic error
entirely; the real signal captures none of that ceiling), the conclusion for
the paper is: **pointwise error calibration is necessary but not sufficient
for a rigid-registration weight — the weight must also preserve the spatial
leverage of the correspondence set, and the proposed formula does the
opposite.** Two structural notes belong next to this claim: (i) the fine
optimizer accepts iterates by *unweighted* mean correspondence distance, which
damps how far any weighting can move the returned pose; (ii) the experiment
covers three consecutive-pair cases of one DTU scan — the paired design is
powerful per block, but scene diversity is limited.

### Suggested figure captions

- `fig_icp_ablation_paired`: "Paired effect of confidence weighting on fine
  ICP pose recovery (45 blocks; identical geometry, sampling, perturbation,
  and coarse initialization within each block). Dots: per-block differences;
  diamonds: median with 95% bootstrap CI; arrows: values beyond the axis
  window. Negative controls are mathematically identical to their reference
  and bound numerical noise. Depth-weighting degrades rotation; no recipe
  improves either metric."
- `fig_icp_ablation_absolute`: "Absolute pose-recovery error by perturbation
  magnitude (median, 95% bootstrap CI). The shared unweighted coarse stage and
  the no-correction reference (error if ICP returned the identity) frame the
  operating regime: translation is substantially recovered, rotation is not."
- `fig_confidence_cues`: "Do the confidence terms predict independent
  geometric accuracy? Per-stereo-pair Spearman rank correlation with
  structured-light error and reliable-point ROC-AUC (circles: pairs; bar:
  median). Only the depth term is calibrated; edge is inverted."
