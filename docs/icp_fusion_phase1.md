# Phase 1 implementation: robust ICP and real spatial fusion

Phase 1 from `icp_fusion_review.md` is implemented. The pipeline now removes unreliable stereo depth before triangulation, registers each cloud against a sparse fused surface model with robust multi-scale ICP, accepts a correction only when its geometry supports it, and merges repeated observations instead of appending them.

## Pipeline changes

```text
rectified stereo pair
  -> left disparity + independent right-to-left disparity
  -> cycle-consistency invalidation + photometric reliability
  -> confidence-weighted points with discontinuity-aware normals
  -> calibrated world pose
  -> robust coarse-to-fine ICP against voxel/surfel model
  -> correction quality gate
  -> confidence-weighted surfel fusion
  -> screened Poisson reconstruction
```

### Stereo validity and point confidence

`Disparity::filterAndComputeConfidence()` runs a second disparity pass in the opposite direction. A point is invalidated if its projected right-image match is out of bounds or if left/right disparity disagreement exceeds `stereo_lr_max_diff`. Valid points receive a soft confidence formed from cycle error and rectified photometric agreement.

`PlyUtils::buildPointCloud()` now combines that pixel confidence with pair confidence, edge confidence, and depth precision. The depth term is normalized inverse variance relative to the cloud median and capped at a 100:1 ratio; this avoids the dimensionally invalid former expression `1 / (1 + variance_mm²)`. Normals are not estimated when any immediate neighbor has invalid disparity or crosses a disparity jump greater than 3 px.

The confidence cull was changed from a threshold relative to the single maximum weight to a quantile cull: it removes the lowest 10% by default. This is important because calibrated inverse-variance weights naturally span a wide range and a maximum-relative cutoff could remove most valid distant samples.

### Robust, coarse-to-fine ICP

`CeresICPOptimizer` now supports:

- reciprocal source/target nearest-neighbor correspondences;
- median/MAD adaptive distance gating;
- quantile trimming of the longest correspondence residuals;
- Cauchy or Huber Ceres losses;
- final re-association after the last increment;
- reported overlap, mean/median/RMSE distance, spatial octant coverage, and an approximate 6-DOF normal-equation condition number.

`IcpFusion` registers at three voxel resolutions (`0.04`, `0.02`, and `0.01` normalized units) with shrinking gates. The coarse level is unweighted; the later levels use source confidence plus a Cauchy loss. This makes the registration less dominated by point density and less brittle to residual stereo outliers.

The correction is accepted only if the fusion-resolution evaluation has sufficient overlap and spatial coverage, improves median residual without materially worsening RMSE, remains bounded (10 degrees / 0.15 normalized units), and has a finite, acceptable conditioning estimate. Otherwise the calibrated world pose is retained.

### Confidence-weighted voxel/surfel fusion

`VoxelFusionModel` replaces raw-vector concatenation. It maintains a voxel hash whose surfels accumulate confidence-weighted position, color, and consistently oriented normal sums. New observations merge into the nearest local surfel; an observation that disagrees too far along an existing surfel normal is rejected. The Poisson stage now consumes this reduced, averaged surface representation.

This is a surfel/voxel fusion step, not yet a projective TSDF: it reduces redundant surface noise and sampling-density bias but does not model free space along camera rays.

## Configuration

The following settings are available in `config.yaml`:

| Key | Default | Meaning |
|---|---:|---|
| `stereo_confidence_filter` | `1` | Enable the explicit left/right validity pass. |
| `stereo_lr_max_diff` | `1.5` | Hard left/right disparity tolerance in pixels. |
| `stereo_photometric_scale` | `25.0` | Scale of the soft rectified photometric confidence. |
| `icp_robust` | `1` | Use Cauchy loss in the production ICP loop. |
| `icp_reciprocal` | `1` | Keep only reciprocal nearest-neighbor pairs. |
| `icp_trim_fraction` | `0.80` | Retain the closest 80% of surviving correspondence residuals. |
| `fusion_voxel_size` | `0.01` | Surfel voxel width in normalized fusion coordinates. |
| `fusion_outlier_factor` | `1.5` | Normal-direction rejection threshold in voxel widths. |

`fusion_voxel_size` is relative to the common normalization computed from cloud 0. The real metric width printed by Poisson or obtained by multiplying by the normalization scale is the value to tune for another scene.

## Verification completed

`IcpVerification` passes all previous known-transform tests. It additionally checks that two noisy observations of a plane collapse into approximately one surfel per voxel, average their opposing zero-mean depth noise to the underlying plane, and reject a nearby normal-direction spike.

A complete `IcpFusion` run on the configured DTU scan completed successfully with the new path enabled:

- explicit stereo validation retained 49–79% of the SGBM-valid pixels, depending on the view pair;
- the robust gate accepted well-supported refinements (for example, cloud 3 improved fusion-resolution median distance from `0.0179` to `0.00583`) and rejected unsafe corrections;
- 826,397 kept input points became 39,833 fused surfels before Poisson reconstruction—a 95.2% reduction through spatial averaging rather than point dropping alone;
- Poisson completed with 12,086 vertices and 23,938 triangles.

## Deliberately deferred to Phase 2

- Per-pixel covariance propagated through triangulation and covariance-aware GICP.
- Matching-cost/uniqueness confidence from an exposed SGBM cost volume; the current photometric cue is only a practical proxy.
- Projective TSDF integration with viewing-ray/free-space updates.
- Multiway pose graph optimization, calibration priors, and re-integration after global pose correction.
- More exhaustive benchmark statistics across DTU scenes, overlaps, perturbations, and random seeds.
