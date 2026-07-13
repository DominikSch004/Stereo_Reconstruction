# Relevant papers for confidence-weighted ICP and stereo point-cloud fusion

This is a project-specific reading list for the current pipeline: calibrated stereo clouds, confidence-weighted point-to-plane ICP, sequential alignment, append-only point fusion, and Poisson surface reconstruction.

## Read these first

### 1. Segal, Haehnel, and Thrun — Generalized-ICP (2009)

**Paper:** [Generalized-ICP](https://roboticsproceedings.org/rss05/p21.html)

**Relevant part:** Sections on the probabilistic formulation, covariance construction, and the plane-to-plane cost.

**Why it matters here:** This is the most direct principled replacement for the current scalar confidence-weighted point-to-plane objective. GICP represents local anisotropic uncertainty in both the source and target and minimizes a Mahalanobis distance. Stereo error is strongly anisotropic—usually much larger along the viewing ray—so a 3×3 covariance is more meaningful than one scalar source weight. It also fixes the current omission of target uncertainty.

**Use it for:** A robust covariance-aware fine ICP stage. Combine propagated stereo covariance with local surface covariance and a robust loss.

### 2. Babin, Giguère, and Pomerleau — Analysis of Robust Functions for Registration Algorithms (2019)

**Paper:** [Analysis of Robust Functions for Registration Algorithms](https://norlab-ulaval.github.io/pdf/Babin2019.pdf)

**Relevant part:** The comparison of M-estimators/outlier filters and the parameter-sensitivity experiments. Cauchy with MAD scaling, variable trimming, and L1 are particularly relevant.

**Why it matters here:** The current Ceres residual blocks use no loss function. Sensor confidence does not make a wrong nearest-neighbor correspondence safe. This paper evaluates robust weighting at large scale and shows which choices remain stable across environments.

**Use it for:** Selecting a Ceres loss and estimating its scale from the residual distribution. A practical first experiment is Cauchy or Huber with MAD-derived scale, compared with trimmed ICP.

### 3. Bouaziz, Tagliasacchi, and Pauly — Sparse Iterative Closest Point (2013)

**Paper:** [Sparse Iterative Closest Point](https://diglib.eg.org/bitstreams/f4d62230-8ca3-48ac-8fdf-5b12f9a0ee6a/download)

**Relevant part:** The sparse `L_p` correspondence-error objective, robust optimization, and incomplete/noisy geometry experiments.

**Why it matters here:** Your clouds contain stereo outliers and only partial overlap. Ordinary squared error gives large residuals excessive influence. Sparse ICP explicitly encourages many correspondence residuals to contribute little or nothing.

**Use it for:** A stronger robust alternative if standard Huber/Cauchy losses are insufficient. It is more invasive than adding a Ceres loss, so it should follow the simpler robust-kernel baseline.

### 4. Curless and Levoy — A Volumetric Method for Building Complex Models from Range Images (1996)

**Paper:** [A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/paper_2_levels/gamma-corrected/paper.html)

**Relevant part:** Sections 3–4 on weighted signed-distance integration, directional range uncertainty, redundant measurements, and free-space reasoning.

**Why it matters here:** This paper addresses the main weakness of the current implementation. Appending registered clouds does not average away noise. Weighted signed-distance fusion combines redundant observations into one surface estimate and can use measurement confidence and ray direction.

**Use it for:** Replacing vector concatenation in `IcpFusion.cpp` with projective TSDF fusion. Your pipeline already retains organized disparities, camera poses, and intrinsics, so it has the right inputs.

### 5. Newcombe et al. — KinectFusion: Real-Time Dense Surface Mapping and Tracking (2011)

**Paper:** [KinectFusion](https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_etal_ismar2011.pdf)

**Relevant part:** The depth pyramid, coarse-to-fine projective point-to-plane ICP, TSDF update, and tracking against the raycast fused model.

**Why it matters here:** It shows how registration and fusion should interact. Instead of aligning a scan to an ever-growing collection of raw points, it aligns the current depth map to a denoised model. Its multi-scale physical-resolution strategy is also better than random 4,000-point subsampling with one fixed threshold.

**Use it for:** A scan-to-model architecture: build a TSDF, raycast a model cloud and normals, then perform coarse-to-fine point-to-plane ICP against that model.

### 6. Choi, Zhou, and Koltun — Robust Reconstruction of Indoor Scenes (2015)

**Paper:** [Robust Reconstruction of Indoor Scenes](https://openaccess.thecvf.com/content_cvpr_2015/html/Choi_Robust_Reconstruction_of_2015_CVPR_paper.html)

**Relevant part:** Fragment construction, pairwise fragment registration, and robust global optimization with line processes that switch incorrect alignment edges off.

**Why it matters here:** Sequentially registering each cloud to all previous raw points is order-dependent and cannot correct early errors. This paper separates local fusion into stable fragments from global pose estimation and robustly rejects bad pairwise constraints.

**Use it for:** A pose graph over stereo-pair clouds, retaining DTU calibration as pose priors and adding ICP edges only between clouds with sufficient overlap.

### 7. Dai et al. — BundleFusion (2017)

**Paper:** [BundleFusion: Real-time Globally Consistent 3D Reconstruction](https://graphics.stanford.edu/projects/bundlefusion/)

**Relevant part:** Hierarchical global pose optimization and surface de-integration/re-integration after poses change.

**Why it matters here:** If all scan poses are jointly refined after some points have already been fused, the surface must be rebuilt or updated. BundleFusion explains why simply modifying poses without reintegrating measurements leaves the geometry inconsistent.

**Use it for:** The longer-term pipeline: jointly optimize all stereo-pair poses, clear/rebuild the TSDF, and reintegrate every organized depth map using the corrected poses.

### 8. Poggi et al. — On the Confidence of Stereo Matching in a Deep-Learning Era (2021/2022)

**Paper:** [On the Confidence of Stereo Matching in a Deep-Learning Era](https://arxiv.org/abs/2101.00431)

**Relevant part:** The taxonomy and quantitative evaluation of stereo-confidence measures, especially confidence from cost-volume/matching ambiguity, left-right consistency, and generalization between datasets.

**Why it matters here:** The current weight uses fixed `sigma_d = 0.5 px`, depth, disparity gradient, and a cloud-constant pose confidence. This paper identifies confidence measures that actually predict disparity failure. ICP weighting can only help if the input confidence correlates with geometric error.

**Use it for:** Designing and evaluating pixel-level confidence before triangulation. Start with left-right agreement, matching-cost margin/curvature, uniqueness, and occlusion cues; calibrate predicted confidence against DTU scan error.

## ICP foundations and implementation guidance

### 9. Besl and McKay — A Method for Registration of 3-D Shapes (1992)

**Paper:** [A Method for Registration of 3-D Shapes](https://graphics.stanford.edu/courses/cs348a-21-winter/Handouts/Besl92.pdf)

**Relevant part:** The alternating closest-point/pose-estimation formulation and local convergence argument.

**Why it matters here:** This is the canonical ICP foundation. It clarifies what ICP guarantees—and, crucially, what it does not guarantee under partial overlap, outliers, or poor initialization.

**Use it for:** Checking the conceptual correctness of the outer associate/solve loop and explaining why calibrated initialization is valuable.

### 10. Chen and Medioni — Object Modeling by Registration of Multiple Range Images (1991/1992)

**Paper:** [Object Modeling by Registration of Multiple Range Images](https://graphics.stanford.edu/~smr/ICP/comparison/chen-medioni-align-rob91.pdf)

**Relevant part:** The point-to-tangent-plane objective and multi-view range-image motivation.

**Why it matters here:** It is the foundation of the point-to-plane mode used by the project. It explains its fast convergence near the solution and its dependence on reliable target surface normals.

**Use it for:** Understanding why point-to-plane should be used only after stable, discontinuity-aware normal estimation and with a good initial pose.

### 11. Zhang — Iterative Point Matching for Registration of Free-Form Curves and Surfaces (1994)

**Paper:** [Iterative Point Matching for Registration of Free-Form Curves and Surfaces](https://www.microsoft.com/en-us/research/publication/iterative-point-matching-registration-free-form-curves-surfaces/)

**Relevant part:** Statistical rejection based on correspondence-distance distributions and treatment of occlusion, disappearing geometry, and subset-to-subset matching.

**Why it matters here:** These are precisely the cases created by partial views and noisy stereo boundaries. The current code only uses a fixed distance gate and a normal threshold.

**Use it for:** Adaptive correspondence rejection based on the observed residual distribution rather than a single normalized `0.1` threshold.

### 12. Rusinkiewicz and Levoy — Efficient Variants of the ICP Algorithm (2001)

**Paper:** [Efficient Variants of the ICP Algorithm](https://graphics.stanford.edu/papers/fasticp/)

**Relevant part:** The taxonomy of ICP stages—point selection, matching, weighting/rejection, error metric, and minimization—and the normal-space sampling experiment.

**Why it matters here:** It provides a clean framework for improving and ablating this implementation. Normal-space sampling is particularly useful because random sampling may overrepresent large flat regions and underconstrain rotation.

**Use it for:** Replacing random subsampling with voxel plus normal-space-stratified sampling, and structuring benchmark variants one ICP stage at a time.

### 13. Low — Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration (2004)

**Paper:** [Linear Least-Squares Optimization for Point-to-Plane ICP](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)

**Relevant part:** The small-angle linearization and derivation of the point-to-plane system.

**Why it matters here:** Your Ceres solve is valid but more general and expensive than necessary for small ICP increments. The derivation also makes degeneracy and conditioning easier to inspect through the 6×6 normal matrix.

**Use it for:** A fast solver and, more importantly, calculating Hessian eigenvalues/condition number to reject refinements that do not constrain all six pose degrees of freedom.

### 14. Pomerleau et al. — Comparing ICP Variants on Real-World Data Sets (2013)

**Paper:** [Comparing ICP Variants on Real-World Data Sets](https://doi.org/10.1007/s10514-013-9327-2)

**Relevant part:** The modular ICP chain, overlap-sensitive results, perturbation protocol, and robust evaluation metrics.

**Why it matters here:** The repository already has a useful benchmark, but only three trials per condition and results showing ICP can worsen good calibration. This paper supplies a methodology for testing sensitivity to initialization, overlap, environment, and parameter settings.

**Use it for:** Redesigning `IcpBenchmark.cpp`: more scenes/seeds, unperturbed inputs, varying overlap/noise, success rates, robust statistics, and confidence intervals.

## Surface fusion and reconstruction alternatives

### 15. Whelan et al. — ElasticFusion (2015)

**Paper:** [ElasticFusion: Dense SLAM Without a Pose Graph](https://thomaswhelan.ie/Whelan15rss.pdf)

**Relevant part:** Surfel representation, confidence-weighted surfel updates, frame-to-model tracking, and deformation of the fused surface after loop closure.

**Why it matters here:** A surfel map is a natural intermediate step between append-only points and a full TSDF. It maintains one local surface estimate—position, normal, radius, color, confidence—per region and supports thin surfaces better than some volumetric representations.

**Use it for:** Confidence-weighted voxel/surfel fusion if implementing a TSDF is too large a first step. It maps closely to the current `PointCloud` structure.

### 16. Schöps, Sattler, and Pollefeys — SurfelMeshing (2019)

**Paper:** [SurfelMeshing: Online Surfel-Based Mesh Reconstruction](https://arxiv.org/abs/1810.00729)

**Relevant part:** Fusion into a dense surfel cloud, surfel smoothing, and asynchronous triangulation.

**Why it matters here:** It demonstrates a modern non-volumetric alternative that handles varying sampling density and thin objects. That is useful if a watertight TSDF is undesirable for the DTU objects.

**Use it for:** Turning a confidence-weighted surfel map into a mesh without passing a thick, duplicated raw cloud to Poisson reconstruction.

### 17. Kazhdan and Hoppe — Screened Poisson Surface Reconstruction (2013)

**Paper:** [Screened Poisson Surface Reconstruction](https://hhoppe.com/screenedpoisson.pdf)

**Relevant part:** The screened positional constraints, the tradeoff between interpolation and smoothing, and discussion of noisy/misregistered oriented points.

**Why it matters here:** The project already has a Poisson-like reconstruction stage. This paper explains what the screening term should do and why Poisson cannot substitute for robust registration or fusion. Badly oriented normals and duplicated noisy samples still bias the indicator field.

**Use it for:** Auditing the current Poisson implementation and choosing screening/sample weights after the inputs have first been robustly fused.

## Useful when initialization or overlap becomes difficult

### 18. Gao and Tedrake — FilterReg (2019)

**Paper:** [FilterReg](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_FilterReg_Robust_and_Efficient_Probabilistic_Point-Set_Registration_Using_Gaussian_Filter_CVPR_2019_paper.html)

**Relevant part:** Soft probabilistic correspondence estimation, explicit outlier handling, and EM optimization.

**Why it matters here:** Hard nearest-neighbor association is brittle with noisy, partially overlapping stereo clouds. FilterReg is a useful comparison if robust GICP still fails because correspondences, rather than residual weighting, are the limiting factor.

**Use it for:** An experimental alternative to hard-correspondence ICP, especially for lower-overlap view pairs. It is not the first implementation priority because calibration already supplies a good pose.

### 19. Zhou, Park, and Koltun — Fast Global Registration (2016)

**Paper:** [Fast Global Registration](https://qianyi.info/docs/papers/eccv16_registration.pdf)

**Relevant part:** Joint robust optimization of feature correspondences without iterative closest-point queries and the multi-scan extension.

**Why it matters here:** It provides a coarse-registration fallback when calibration placement is unavailable or wrong. It also shows how false feature matches can be suppressed inside one robust objective.

**Use it for:** Initializing ICP from FPFH-like 3-D feature correspondences. Do not replace the calibrated initialization with it unless experiments show a need.

### 20. Yang, Shi, and Carlone — TEASER/TEASER++ (2020/2021)

**Paper:** [TEASER: Fast and Certifiable Point Cloud Registration](https://arxiv.org/abs/2001.07715)

**Relevant part:** Truncated least squares, correspondence compatibility, extreme-outlier robustness, and certifiable rotation estimation.

**Why it matters here:** This is valuable when coarse feature matching contains many false correspondences or when a cloud is far from its initialization. It solves a different problem from fine ICP.

**Use it for:** A robust global initialization/failure-recovery stage, followed by GICP. It is lower priority for the current DTU pipeline because calibrated camera poses are already available.

## Evaluation reference

### 21. Aanæs et al. — Large-Scale Data for Multiple-View Stereopsis (DTU dataset, 2016)

**Paper:** [Large-Scale Data for Multiple-View Stereopsis](https://orbit.dtu.dk/en/publications/large-scale-data-for-multiple-view-stereopsis/)

**Relevant part:** The DTU acquisition setup and the separate accuracy/completeness evaluation methodology.

**Why it matters here:** The project uses DTU data and ground-truth structured-light scans. Registration RMSE alone is not enough: a method can improve alignment for a small overlap while reducing reconstruction completeness or moving correct surfaces away from ground truth.

**Use it for:** Reporting accuracy, completeness, and their tradeoff for the final fused reconstruction, with distance thresholds expressed in millimetres.

## Suggested reading order tied to implementation

| Implementation task | Read first | Expected code change |
|---|---|---|
| Add robust residuals | Babin; Sparse ICP; Zhang | Ceres loss/IRLS, MAD scale, trimming, adaptive rejection |
| Improve sampling and coarse-to-fine ICP | Rusinkiewicz–Levoy; KinectFusion | Voxel pyramids, normal-space sampling, shrinking physical gates |
| Replace scalar confidence | GICP; Poggi et al. | Pixel confidence, 3-D covariance, source+target Mahalanobis weight |
| Actually reduce fusion noise | Curless–Levoy; KinectFusion | Weighted TSDF integration and model raycasting |
| Implement a smaller fusion step first | ElasticFusion; SurfelMeshing | Voxel-hash surfels with weighted position/normal/color updates |
| Prevent sequential drift | Choi et al.; BundleFusion | Pose graph/global optimization and surface reintegration |
| Validate the finished pipeline | Pomerleau et al.; DTU paper | Larger perturbation suite, success rates, accuracy/completeness |
| Recover from bad initialization | FGR; TEASER++; FilterReg | Feature-based robust initialization or soft association |

## Minimal shortlist

If time permits only six papers, read these in this order:

1. **Generalized-ICP** — the best match for confidence/covariance-aware registration.
2. **Analysis of Robust Functions** — the fastest practical improvement to current ICP.
3. **Curless–Levoy volumetric fusion** — the key answer to noisy append-only fusion.
4. **KinectFusion** — integration of coarse-to-fine ICP with a denoised fused model.
5. **Poggi et al. stereo confidence review** — making the point confidence meaningful.
6. **Choi et al. robust reconstruction** — replacing greedy sequential poses with robust global optimization.
