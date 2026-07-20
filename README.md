# Stereo Reconstruction

3D Scanning & Motion Capture — TUM SS26  
Group 11: Ayush Datta, Dominik Schneider, Francisco Mónica, Andrés Forero

## Project Overview
Classical stereo reconstruction pipeline with two contributions:
- **Contribution A:** Improving sparse matching and essential matrix estimation via confidence-weighted correspondences
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
├── docs/                      # pipeline.md, papers/
├── scripts/                   # download_dtu.sh, analyze_icp_ablation.py
├── data/dtu/                  # DTU dataset (git-ignored, populated by download script)
└── results/                   # generated point clouds / ablation output (git-ignored)
```

## Dependencies
- OpenCV ≥ 4.5 (`core imgproc highgui calib3d features2d`)
- Eigen3
- Ceres Solver
- glog
- CMake ≥ 3.14, a C++17 compiler

## Dataset
[DTU MVS 2014](https://roboimagedata.compute.dtu.dk/?page_id=36) — place scenes under `data/dtu/`.
Run `./scripts/download_dtu.sh` to download automatically.

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

### Executables
| Target | Purpose |
| --- | --- |
| `StereoReconstruction` | Full pipeline on one stereo pair, writes a point cloud |
| `IcpFusion` | Runs the pipeline per view pair, then fuses via confidence-weighted ICP |
| `IcpVerification` | Synthetic known-transform recovery (exit 0 = all checks pass) |
| `IcpBenchmark` | DTU benchmark: unweighted vs. confidence-weighted, chamfer vs. GT scan |
| `ConfidenceIcpAblation` | Paired factor ablation of `w = c_global·c_depth·c_edge·c_stereo` |
| `FusedCloudEval` | Chamfer accuracy of saved clouds vs. the DTU GT scan |
| `SparseKeyPointMatching`, `EightPoint`, `EightPointRobustness`, `StereoRectification`, `StereoMatching`, `TriangulationVerification`, `PipelineEvaluationVerbose`, `MeshReconstruction`, `PoissonReconstruction`, `PoseRecoveryComparison` | Per-step visualization / verification |

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
