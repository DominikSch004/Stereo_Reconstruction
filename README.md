# Stereo Reconstruction

3D Scanning & Motion Capture — TUM SS26  
Group 11: Ayush Datta, Dominik Schneider, Francisco Mónica, Andrés Forero

## Project Overview
Classical stereo reconstruction pipeline with two contributions:
- **Contribution A:** Improving Fundamental Matrix estimation through robust estimator
- **Contribution B:** Multi-view fusion via confidence-guided ICP alignment

## Pipeline
1. Sparse keypoint matching (SIFT + FLANN)
2. Fundamental / essential matrix estimation (normalized 8-point + RANSAC / MAGSAC / PROSAC)
3. Stereo rectification (Loop & Zhang 1999)
4. Dense stereo matching (simplified SGM, Hirschmüller 2008)
5. Disparity → depth via triangulation (per-pixel DLT)
6. Confidence-weighted ICP multi-view fusion
7. Surface reconstruction (Poisson / SDF + Marching Cubes)

## Repository Structure
```
.
├── CMakeLists.txt              # build definition (targets, deps)
├── config.yaml                # pipeline step backends + parameters (read at runtime)
├── include/                   # public headers for every component/util
├── src/
│   ├── main.cpp               # StereoReconstruction: single stereo pair → point cloud
│   ├── IcpFusion.cpp          # IcpFusion: multi-view pipeline + confidence-weighted ICP
│   ├── Pipeline.cpp           # pipeline orchestration
│   ├── PipelineConfig.cpp     # config.yaml loading
│   ├── components/            # SparseKeyPointMatcher, FundamentalMatrix, Rectification,
│   │                          #   Disparity, Triangulation, ICP, PoissonReconstruction
│   ├── utils/                 # image/PLY/mesh/geometry/NN helpers + experiments
│   └── visualization/         # standalone per-step verification/visualization executables
├── benchmark/                 # Evaluator + ICP/ablation/cloud-eval benchmark executables
├── scripts/                   # download_dtu.sh, analyze_icp_ablation.py
├── data/dtu/                  # DTU dataset (git-ignored, populated by download script)
└── results/                   # generated point clouds / ablation output (git-ignored)
```

## Dependencies & installation
- OpenCV ≥ 4.5 (`core imgproc highgui calib3d features2d`)
- Eigen3
- Ceres Solver
- glog
- CMake ≥ 3.14, a C++17 compiler
- Python 3 with `numpy`, `scipy`, `matplotlib` (only for the ablation analysis script)

On Debian/Ubuntu the C++ dependencies install with:
```bash
sudo apt install build-essential cmake libopencv-dev libeigen3-dev libceres-dev libgoogle-glog-dev
pip install numpy scipy matplotlib   # ablation analysis only
```

## Dataset
[DTU MVS 2014](https://roboimagedata.compute.dtu.dk/?page_id=36) — place scenes under `data/dtu/`.
Run `./scripts/download_dtu.sh` and select the Sample Set (Option 1) to download automatically. The dataset, build
tree, and generated results are **not** included in this repository/archive
(they contain multi-hundred-MB files); regenerate the dataset with the download
script and the outputs by running the executables above.

## Build
The project defaults to an optimized `Release` build (`-O3`, `NDEBUG`); pass the
flag explicitly if you configure a fresh build tree and want to be sure:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run
Executables resolve `../config.yaml` and `../data/dtu/` relative to the working
directory, so run them from inside `build/`:
```bash
cd build
./StereoReconstruction            # single stereo pair → pointcloud.ply
./IcpFusion                       # multi-view fusion → pointcloud_fused.ply, mesh_fused.ply
```
Pass an alternative config path as the first argument, e.g. `./StereoReconstruction ../my_config.yaml`.
Which backend each pipeline step uses (SIFT/ORB, MAGSAC/PROSAC, custom/OpenCV
disparity, ICP mode, confidence terms, …) is selected in [`config.yaml`](config.yaml).

### Entrypoint files
Each executable is a single `main()` in the source file below. Unless noted, all
read `config.yaml` (step backends + parameters) and the DTU scene under
`data/dtu/`; a different config path can be passed as the first CLI argument.

| Executable | Source file | Input | Output |
| --- | --- | --- | --- |
| `StereoReconstruction` | `src/main.cpp` | config + one DTU stereo pair | `pointcloud.ply`; pose/reconstruction metrics to stdout |
| `IcpFusion` | `src/IcpFusion.cpp` | config + DTU view range (`icp_view_range`) | `pointcloud_fused_preicp.ply`, `pointcloud_fused.ply`, `mesh_fused.ply` |
| `IcpVerification` | `benchmark/IcpVerification.cpp` | none (synthetic clouds) | pass/fail to stdout, exit 0 = all checks pass |
| `IcpBenchmark` | `benchmark/IcpBenchmark.cpp` | config + DTU scene | benchmark CSV (unweighted vs. confidence-weighted, chamfer vs. GT) |
| `ConfidenceIcpAblation` | `benchmark/ConfidenceIcpAblation.cpp` | config + CLI flags (see below) | `icp_weight_ablation.csv`, `confidence_cues.csv` |
| `FusedCloudEval` | `benchmark/FusedCloudEval.cpp` | one or more `.ply` clouds as CLI args | chamfer distance to DTU GT scan, to stdout |
| `SparseKeyPointMatching` | `src/visualization/sparse_key_point_matching.cpp` | config + DTU pair | match visualization window |
| `EightPoint` / `EightPointRobustness` | `src/visualization/8point*.cpp` | config + DTU pair | epipolar-geometry figures / robustness CSV, saved in /experiments |
| `StereoRectification` | `src/visualization/stereo_rectification.cpp` | config + DTU pair | rectified-pair PNG |
| `StereoMatching` | `src/visualization/stereo_matching.cpp` | config + DTU pair | disparity map PNG |
| `TriangulationVerification` | `src/visualization/triangulation_verification.cpp` | config + DTU pair | reprojection-check PNG |
| `PipelineEvaluationVerbose` | `src/visualization/pipeline_eval_verbose.cpp` | config + DTU pair | per-step diagnostic PNGs |
| `MeshReconstruction` / `PoissonReconstruction` | `src/visualization/*_reconstruction.cpp` | a `.ply` point cloud | reconstructed mesh `.ply` |
| `PoseRecoveryComparison` | `src/visualization/pose_recovery_comparison.cpp` | config + DTU pair | pose-estimator comparison PNG |

### Main functions
- `Pipeline::runPipeline(imgL, imgR, K, res, config, C1, C2)` (`src/Pipeline.cpp`) —
  runs the whole 7-step pipeline and fills a `PipelineResult` with the rectified
  images, dense disparity, per-pixel confidence, back-projected 3D points, and
  the estimated relative pose.
- `PlyUtils::buildAndSavePLY(...)` (`src/utils/PlyUtils.cpp`) — back-projects the
  dense disparity into a colored, confidence-weighted point cloud and writes it
  as a `.ply`.
- `Evaluator::evaluateMetrics(params)` (`benchmark/Evaluator.cpp`) — compares the
  estimated pose and reconstruction against DTU ground truth (rotation/translation
  error, epipolar error, inlier ratio, reconstruction distance).
- `ICPOptimizer::estimatePose(source, target, initialPose)` (`src/components/ICP.cpp`) —
  the iterative confidence-weighted ICP that aligns one cloud onto another;
  used by `IcpFusion` to merge per-pair clouds into one fused model.

## Confidence-weighted ICP ablation
Reproduces the paired DTU factor ablation of `w = c_global · c_depth · c_edge · c_stereo`.
The C++ benchmark writes raw CSVs to `--out`; the Python script produces the
figures and tables (requires `numpy`, `scipy`, `matplotlib`).

```bash
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
