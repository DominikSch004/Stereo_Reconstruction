# Review of confidence-weighted ICP and multi-view fusion

## Executive summary

The ICP optimizer is structurally sound and contains several good engineering choices: metric calibration supplies a strong initialization, correspondences are recomputed after every increment, the transform is accumulated correctly, weights enter least squares through `sqrt(weight)`, point-to-plane constraints are available, normals can reject implausible matches, and a failed refinement falls back to the calibrated pose.

The current output remains noisy mainly because the fusion stage does not fuse measurements. It concatenates every accepted cloud. Registration can align the means of noisy observations, but it cannot remove their depth noise; concatenation preserves all noisy samples and increases local thickness and point density. The final Poisson stage receives the same noisy points and locally differentiated, noisy normals.

Confidence-weighted ICP is mathematically sensible when the confidence is proportional to measurement precision (inverse residual variance). The implementation applies such a weight correctly. However, the current `finalWeight` is a heuristic product, not a calibrated inverse variance, uses only source confidence, does not robustify geometric residuals, and does not affect correspondence selection. The repository's synthetic verification confirms that true inverse-variance weighting helps. In contrast, the saved real-DTU benchmark shows that the present stereo-derived weights do not help consistently and that ICP frequently worsens an already good calibrated placement.

Recommended direction, in order:

1. Replace append-only fusion with confidence-weighted voxel/surfel fusion or a projective TSDF.
2. Improve the dense stereo validity/confidence signal before changing the optimizer.
3. Add robust residual weighting, reciprocal/unique correspondences, trimming, and adaptive gates.
4. Replace scalar confidence with anisotropic covariance and use Generalized ICP (GICP), ideally combined with robust kernels.
5. Register scans to a stable fused model and optimize all poses jointly (pose graph or multiway registration), rather than greedily aligning each scan to an ever-growing raw point set.

## 1. What the implementation does

### Point-cloud construction and confidence

For each selected stereo pair, the pipeline estimates relative pose, rectifies the images, computes dense disparity, triangulates one point per valid disparity pixel, estimates a normal from four immediate 3-D neighbors, and assigns

\[
w_i = c_{pair}\;\frac{1}{1+\sigma_{Z,i}^2}\;\exp(-\|\nabla d_i\|/5),
\qquad
\sigma_{Z,i}^2 = \frac{Z_i^4}{(fB)^2}\sigma_d^2,
\]

with fixed `sigma_d = 0.5 px`. Points below 10% of the maximum weight of their cloud are removed. Each cloud is then moved from its rectified camera frame into the calibrated DTU world frame.

The pair confidence is the cheirality inlier fraction after `recoverPose`. It is constant for all points in a cloud. The depth factor models the familiar rapid growth of stereo depth uncertainty with distance. The disparity-gradient factor suppresses depth discontinuities.

### ICP objective

At every outer iteration the current source is transformed, nearest target neighbors are found with FLANN, matches beyond a fixed normalized distance of `0.1` are removed, and—when point-to-plane mode is active—matches with normal dot product below `0.5` are pruned.

The point-to-point objective is

\[
\min_T \sum_{(i,j)} w_i\|Tp_i-q_j\|^2,
\]

and the point-to-plane objective is

\[
\min_T \sum_{(i,j)} w_i[n_j^T(Tp_i-q_j)]^2.
\]

The Ceres residual is multiplied by `sqrt(w_i)`, which is exactly correct for weighted least squares. A six-parameter angle-axis/translation increment is solved from identity and left-multiplied into the accumulated pose.

Production uses 40 unweighted coarse iterations on two 4,000-point random samples, followed by 20 confidence-weighted fine iterations of the full source against the full fused target. A refinement is kept if at least 10% of source points have an accepted final match.

### Fusion

After registration, all points, colors, weights, normals, and validity flags are appended to the accumulated cloud. There is no spatial merging, averaging, resampling, duplicate removal, free-space reasoning, or outlier rejection at this stage. Poisson reconstruction is then run on the concatenated oriented points.

## 2. What works well

### Correct optimization mechanics

- `sqrt(w)` is the correct residual scaling. The code does not make the common error of multiplying a least-squares residual by `w`, which would square the intended influence.
- Re-association after every pose increment is standard ICP behavior and is implemented correctly.
- Applying the incremental transform to already transformed points and then left-composing it into the estimate is internally consistent.
- Target normals are used for point-to-plane residuals, while invalid target normals fall back to point-to-point constraints instead of silently deleting every such sample.
- The synthetic verification covers noise-free recovery, equivalence under uniform weights, and heteroscedastic noise with `w=1/sigma^2`. The current run passed all checks. With heteroscedastic noise, weighted point-to-plane reduced RMSE from `0.00170` to `0.00054`, demonstrating that the optimizer can exploit meaningful weights.

### Strong initialization and safeguards

- Every cloud is first placed in a common metric world frame using DTU calibration. ICP therefore refines rather than discovers the full transform, which is the right use of local ICP.
- Metric baseline recovery avoids independent arbitrary scales for the stereo pairs.
- The implementation rejects near-empty reconstructions and low-overlap refinements, retaining the calibrated placement as a safe fallback.
- The same normalization is applied to every cloud, so relative geometry is preserved.
- Fixed seeds make sampling and benchmarks reproducible.

### Reasonable first confidence cues

Depth uncertainty increasing approximately as `Z^4` in variance follows from `Z=fB/d` and first-order disparity error propagation. Suppressing discontinuities also has a useful practical motivation because stereo mismatches and foreground/background interpolation are common there. These are useful ingredients; they need better calibration and separation of roles.

## 3. Main limitations and risks

### 3.1 Registration does not denoise; append-only “fusion” preserves noise

This is the dominant issue. A rigid pose has six degrees of freedom. It can correct scan-level pose error but cannot move each noisy point onto the latent surface independently. Appending registered clouds creates multiple noisy observations around each surface and may make the band thicker. It also makes later target clouds increasingly dense in frequently observed areas, so nearest-neighbor ICP implicitly weights those regions by sampling density.

This fusion is order-dependent: cloud 0 is never refined, every later cloud is aligned to a target containing all previous raw noise, and early registration errors contaminate subsequent targets. A weighted voxel centroid, surfel map, TSDF, or probabilistic surface model would use repeated observations to reduce noise rather than retain all samples.

### 3.2 The current confidence is not an inverse residual variance

Weighted least squares has a statistical interpretation when `w_i` is proportional to inverse variance in the residual direction. Here:

- `1/(1+variance)` is bounded heuristic compression, not `1/variance`. Its relative behavior depends on the units (the code uses millimetres), and adding dimensionless `1` to a value in `mm^2` has no unit-consistent interpretation.
- A fixed `sigma_d=0.5 px` ignores pixel-wise matching ambiguity, texture, occlusion, uniqueness, and left-right inconsistency—the principal causes of dense stereo outliers.
- `globalConfidence` is constant within a source cloud. Multiplying every residual in one solve by the same constant does not change the least-squares optimum; it only scales the objective. It therefore has essentially no effect on that cloud's ICP pose unless numerical stopping changes. It would matter when comparing/combining measurements from different clouds in one objective or during actual fusion.
- The edge term treats a real depth discontinuity as low confidence even when it is correctly reconstructed. Such points can be geometrically informative but need discontinuity-aware matching, not necessarily blanket downweighting.
- No lower/upper clamp or normalization controls the numerical dynamic range. A cloud with extremely small weights gives Ceres a very small objective without changing its optimum.

Conclusion: the idea makes sense; this particular confidence is not yet evidence that a point is a geometrically reliable ICP observation.

### 3.3 Only source confidence is used

A correspondence depends on both measurements. A confident source point matched to an uncertain/noisy target point still receives high weight. Conversely, the accumulated target contains points from multiple scans with different confidence, but those weights are ignored.

For scalar independent errors, a more defensible correspondence precision is approximately

\[
w_{ij}=1/(\sigma_{r,i}^2+\sigma_{r,j}^2).
\]

For point-to-plane, `sigma_r^2` should be the covariance projected onto the target normal. With anisotropic point covariances, this leads naturally to GICP/Mahalanobis residuals.

### 3.4 Confidence changes the solve, not data association or outlier robustness

Every in-gate nearest neighbor enters Ceres and `nullptr` is passed as the loss function. A wrong correspondence with moderate or high confidence can dominate the result. Sensor confidence and geometric robustness solve different problems and should be combined:

\[
\rho\!\left(w_{sensor,ij}\,r_{ij}^2\right),
\]

or via iteratively reweighted least squares with `w_total = w_sensor * w_robust`.

Useful additions are Huber/Cauchy/Geman–McClure or graduated non-convex weighting, rejection by robust residual scale, reciprocal nearest neighbors, one-to-one target assignment, trimmed ICP, and overlap-aware correspondence selection.

### 3.5 Fixed thresholds and weak acceptance criterion

- `0.1` is a fixed gate after normalization by cloud 0's global standard deviation. Its physical size changes with the scene. Coarse and fine stages use the same gate.
- A normal threshold of 60 degrees is permissive and not adapted to estimated normal uncertainty.
- The final accepted match count alone is insufficient. Ten percent of a large cloud may be a spatially concentrated, nearly planar, or repeated region that cannot constrain all six degrees of freedom.
- `lastMatchCount` is measured before the final optimization step. It is not recomputed after applying that increment.
- There is no check for cost decrease relative to calibration, robust RMSE/median, inlier spatial distribution, Hessian conditioning, transform magnitude, or consistency with expected calibration uncertainty.

A refinement should be accepted only if it improves a held-out or robust geometric score, has adequate overlap distributed across space, is well-conditioned, and remains inside plausible correction bounds.

### 3.6 Point-to-plane quality is limited by raw normals

Normals are finite differences over immediate neighboring reprojected pixels. Differentiation amplifies disparity noise; neighbors are checked for finiteness but not disparity validity, depth jumps, or foreground/background membership. This can produce unreliable normals precisely where stereo is noisy. The point-to-plane residual may then be worse than point-to-point.

Normals are oriented toward each source camera. This is usually reasonable for the visible side of an opaque surface, but cross-view consistency should be verified before using signed normal agreement or Poisson reconstruction. Multi-scale PCA/plane-fit normals on a filtered cloud, projective normals with discontinuity checks, and confidence based on local plane eigenvalues would be safer.

### 3.7 Greedy scan-to-growing-cloud registration can drift and bias

Sequential registration has no loop closure or joint correction. It also matches later scans to a raw target whose density grows with every append. Better choices include:

- scan-to-model registration against a fixed-resolution surfel/TSDF model;
- pairwise constraints followed by pose-graph optimization;
- multiway registration that jointly optimizes all scan poses;
- retaining calibrated poses as priors so weakly constrained ICP cannot move a scan excessively.

### 3.8 Benchmark evidence currently argues against enabling these weights by default

The saved `build/icp_benchmark.csv` contains three trials at each of 2, 5, and 10 degree perturbations. Mean results are:

| Perturbation | Variant | Rotation error (deg) | Translation error (mm) | GT Chamfer mean (mm) |
|---:|---|---:|---:|---:|
| 2° | no ICP | 2.000 | 2.564 | 19.365 |
| 2° | point-to-point, unweighted | 5.552 | 6.955 | 21.050 |
| 2° | point-to-point, weighted | 5.524 | 7.357 | 21.086 |
| 2° | point-to-plane, unweighted | 6.495 | 5.747 | 21.404 |
| 2° | point-to-plane, weighted | 6.921 | 6.335 | 21.400 |
| 5° | no ICP | 5.000 | 6.410 | 20.337 |
| 5° | point-to-point, unweighted | 5.995 | 6.187 | 21.381 |
| 5° | point-to-point, weighted | 5.150 | 7.957 | 21.401 |
| 5° | point-to-plane, unweighted | 7.621 | 7.228 | 21.756 |
| 5° | point-to-plane, weighted | 8.733 | 9.677 | 22.138 |
| 10° | no ICP | 10.000 | 12.820 | 21.815 |
| 10° | point-to-point, unweighted | 5.576 | 6.537 | 21.133 |
| 10° | point-to-point, weighted | 5.363 | 7.153 | 21.214 |
| 10° | point-to-plane, unweighted | 7.186 | 6.043 | 21.511 |
| 10° | point-to-plane, weighted | 8.474 | 8.348 | 21.968 |

Interpretation:

- ICP helps recover the larger 10-degree artificial pose error, but it degrades the already-close 2-degree case.
- Weighted point-to-point has slightly lower rotation error in some cases but worse translation and Chamfer than unweighted.
- Weighted point-to-plane is consistently worse than unweighted in these aggregates.
- All ICP variants make the GT Chamfer worse at 2 and 5 degrees. Even at 10 degrees, point-to-point only improves relative to the perturbed cloud; it does not establish that it improves the original calibrated placement.

These results are not a final scientific conclusion: only three trials are used, the benchmark uses pairs `(1,2)` and `(3,4)` while production uses a different sequence, it does not apply production's confidence cull, and the target/evaluation sampling deserves review. Nevertheless, it is direct evidence that the current real-data weights and point-to-plane normals should not be assumed beneficial. The benchmark should become a regression gate and report confidence intervals over many scenes and seeds.

## 4. Does confidence-weighted ICP make sense?

Yes—with an important qualification. Confidence-weighted ICP is weighted maximum likelihood if correspondences are correct, errors are approximately Gaussian, and weights are the inverse variance of the chosen residual. The implementation's `sqrt(w)` mechanics satisfy this. The synthetic test validates exactly that scenario.

Real stereo error is anisotropic, spatially correlated, non-Gaussian around occlusions, and often multimodal when correspondence is ambiguous. A single scalar derived mostly from depth is therefore incomplete. Confidence weighting should be retained as one factor, but upgraded to measurement covariance and combined with a robust correspondence model.

The closest established improvement is Generalized ICP. GICP models local covariance in both point sets and minimizes a Mahalanobis distance, effectively producing a plane-to-plane probabilistic registration. Its authors report improved robustness over point-to-point and point-to-plane ICP. This maps well to stereo, whose uncertainty is much larger along the viewing ray than laterally. See [Segal, Haehnel, and Thrun, Generalized-ICP](https://roboticsproceedings.org/rss05/p21.html).

Probabilistic point-set methods such as FilterReg replace hard nearest-neighbor assignments with soft probabilistic associations and explicitly address outliers and occlusion. They can be more robust when raw clouds have partial overlap and many outliers, at greater implementation complexity. See [Gao and Tedrake, FilterReg](https://arxiv.org/abs/1811.10136).

For the fusion problem, weighted signed-distance integration is more directly relevant than another ICP variant. Curless and Levoy's volumetric method was designed to combine aligned range images while representing directional uncertainty and using redundant observations to reduce noise. See [Curless and Levoy, A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/paper_2_levels/gamma-corrected/paper.html).

## 5. Recommended redesign

### Phase 1: high-value, moderate-effort changes

1. **Validate and filter disparity before point construction.** Add left-right consistency, uniqueness/cost-margin confidence, speckle/component filtering, valid-neighbor checks, and occlusion masking. If SGBM does not expose enough cost information, compute confidence from left-right disparity agreement and local photometric/reprojection consistency.

2. **Use robust ICP.** Attach a Ceres Huber or Cauchy loss, estimate its scale from the median absolute residual per outer iteration, use reciprocal correspondences, and trim a configurable high-residual fraction. Sensor confidence should multiply, not replace, robust weights.

3. **Use a coarse-to-fine geometry pyramid.** Voxel-downsample both scans at several physical resolutions; use a large gate initially and shrink it with voxel size/residual scale. Estimate stable normals at the corresponding scale. Random sampling does not regularize density or guarantee spatial coverage.

4. **Strengthen refinement acceptance.** Compare robust pre/post alignment metrics; require improvement, sufficient distributed overlap, bounded transform, and a well-conditioned six-DOF solve. Recompute matches after the final increment. The calibrated pose should be a prior, not merely a fallback.

5. **Implement actual fusion.** As the simplest step, maintain a voxel hash. Within each voxel or matched surfel, combine positions, colors, and normals by confidence-weighted averaging and accumulate total precision/observation count. Reject measurements too far from the current surfel along its normal. This immediately reduces duplicates and random noise.

### Phase 2: principled uncertainty and surface fusion

1. **Propagate per-pixel covariance.** Derive the Jacobian from `(u,v,d)` to `(X,Y,Z)` and propagate image/disparity covariance:

   \[
   \Sigma_X = J\Sigma_{uvd}J^T.
   \]

   Obtain `sigma_d` per pixel from matching cost curvature/margin or left-right disagreement. Transform covariance with each cloud pose.

2. **Adopt covariance-aware GICP.** Combine source and target covariances in the residual. Use local surface covariance as well as sensor covariance, plus a robust kernel. For point-to-plane, at minimum use

   \[
   w_{ij}=1/[n_j^T(R\Sigma_iR^T+\Sigma_j)n_j+\epsilon].
   \]

3. **Fuse range images projectively with TSDF.** The pipeline still has organized depth and camera poses, so it can integrate each range image along rays. Weight updates by disparity/depth uncertainty, viewing angle, confidence, and robust agreement with the existing surface. TSDF fusion uses repeated views to average noise and reasons about free space; append-and-Poisson cannot do either.

4. **Optimize all poses.** Form pairwise constraints only for overlapping views, estimate their covariance/information, retain calibration priors, and solve a pose graph. Reintegrate all depth maps after pose optimization so fusion is not order-dependent.

### Suggested end-state pipeline

```text
stereo images
  -> disparity + left/right/occlusion checks + per-pixel uncertainty
  -> organized depth + covariance
  -> calibrated initial poses
  -> multi-scale robust GICP against surfel/TSDF model
  -> pose graph with calibration priors
  -> reintegration into confidence-weighted TSDF/surfel map
  -> mesh extraction + optional edge-aware cleanup
```

## 6. Validation plan

Registration and fusion must be evaluated separately.

### Registration experiments

- Compare calibration-only, unweighted point-to-point, robust point-to-plane, current weighted ICP, and robust covariance-aware GICP.
- Use many view pairs, perturbation magnitudes, axes, overlap levels, and at least 20–50 seeds per condition.
- Report rotation/translation recovery, robust symmetric Chamfer to GT, point-to-plane distance, success rate, runtime, and confidence intervals.
- Include an unperturbed/calibration-only case. An ICP method that moves an already correct scan away from GT must fail the acceptance gate.
- Ablate each confidence factor: pair constant, depth variance, disparity edge, left-right consistency, matching ambiguity, target confidence, and robust loss.
- Calibrate confidence: bin points by predicted standard deviation and compare predicted vs empirical 3-D/normal-direction error against DTU ground truth.

### Fusion experiments

- Compare concatenation, voxel centroid fusion, surfel fusion, TSDF, and the existing Poisson output using identical poses.
- Measure accuracy and completeness against DTU ground truth, F-score at several millimetre thresholds, normal consistency, surface thickness in repeated-view regions, point/voxel count, and runtime/memory.
- Then repeat with jointly optimized poses to separate pose improvements from fusion improvements.
- Visualize per-voxel observation count, accumulated precision, rejected measurements, and residual distributions. These diagnostics reveal whether confidence actually selects accurate samples.

## 7. Concrete code-level findings

- `src/components/ICP.cpp`: residual weighting and transform accumulation are correct; add robust loss functions, target/correspondence covariance, final re-association, solve-validity checks, and richer quality metrics.
- `src/utils/PlyUtils.cpp`: replace fixed disparity uncertainty and heuristic bounded depth confidence with calibrated pixel-wise uncertainty; validate neighbors before normal construction; retain image coordinates/camera ray or covariance in `PointCloud`.
- `src/IcpFusion.cpp`: replace random sampling with voxel pyramids; avoid full raw-cloud targets; make thresholds physical/configurable; add transform acceptance tests; replace vector insertion with a fusion model; store and jointly optimize per-cloud poses.
- `benchmark/IcpVerification.cpp`: the synthetic inverse-variance test is useful, but add outliers, partial overlap, wrong confidences, anisotropic/ray-aligned noise, and degenerate geometry.
- `benchmark/IcpBenchmark.cpp`: bring it into exact parity with production preprocessing and pairs, add calibration-only/unperturbed trials, more seeds/scenes, symmetric metrics, and automated pass/fail thresholds.

## Bottom line

Keep the confidence-weighted least-squares mechanism, but do not treat the present weight formula as validated confidence and do not expect ICP itself to denoise. The fastest path to visibly cleaner results is robust correspondence filtering plus real spatial fusion. The most principled longer-term solution is robust covariance-aware GICP for registration, pose-graph optimization for all scans, and uncertainty-weighted TSDF or surfel fusion for reconstruction.
