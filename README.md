# Stereo Reconstruction

A modular C++ pipeline for 3D stereo reconstruction and multi-view ICP fusion. This repository provides a complete feature-to-mesh pipeline, transitioning from sparse keypoint matching and fundamental matrix estimation to dense disparity maps and confidence-weighted surface reconstruction.

## Pipeline Architecture
1. **Sparse Keypoint Matching:** SIFT + FLANN
2. **Pose Estimation:** Normalized 8-point algorithm with robust estimators (RANSAC / MAGSAC / PROSAC)
3. **Stereo Rectification:** Loop & Zhang (1999)
4. **Dense Stereo Matching:** Simplified Semi-Global Matching (SGM, Hirschmüller 2008)
5. **Depth Triangulation:** Per-pixel DLT (Disparity → Depth)
6. **Multi-View Fusion:** Confidence-weighted Iterative Closest Point (ICP)
7. **Surface Reconstruction:** Poisson / SDF + Marching Cubes

## Dependencies
- OpenCV ≥ 4.5 (`core imgproc highgui calib3d features2d`)
- Eigen3
- Ceres Solver
- glog
- CMake ≥ 3.14, a C++17 compiler
- Python 3 with `numpy`, `scipy`, `matplotlib` (for benchmarking scripts only)

On Debian/Ubuntu, the C++ dependencies can be installed via:
```bash
sudo apt install build-essential cmake libopencv-dev libeigen3-dev libceres-dev libgoogle-glog-dev
pip install numpy scipy matplotlib

```

## Quickstart: DTU MVS 2014 Dataset

The pipeline is pre-configured to run out-of-the-box on the [DTU MVS 2014 Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36).

To automatically fetch the Sample Set:

```bash
./scripts/download_dtu.sh

```

*(Select Option 1. The data will be extracted to `data/dtu/`)*.

## Build

The project defaults to an optimized `Release` build (`-O3`, `NDEBUG`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

```

## Usage

Executables resolve relative to the working directory. Ensure you run them from inside `build/`:

```bash
cd build
./StereoReconstruction            # single stereo pair → pointcloud.ply
./IcpFusion                       # multi-view fusion → pointcloud_fused.ply, mesh_fused.ply

```

Configuration is handled via YAML. You can swap backends for almost every pipeline step (e.g., SIFT vs. ORB, MAGSAC vs. PROSAC, ICP modes) by editing `config.yaml`. To pass an alternative configuration path:

```bash
./StereoReconstruction ../my_config.yaml

```

## Benchmarking & Evaluation

The repository includes an ablation testing suite to evaluate the confidence-weighted ICP pipeline (`w = c_global · c_depth · c_edge · c_stereo`).

To run the C++ benchmark and generate the Python evaluation figures:

```bash
# 1. Build the benchmark target
cmake --build build -j --target ConfidenceIcpAblation

# 2. Run the C++ benchmark
cd build
./ConfidenceIcpAblation ../config.yaml \
  --out ../results/icp_weight_ablation \
  --cases 3 --trials 5 \
  --source-samples 4000 --coarse-samples 2000 --target-samples 12000 \
  --cue-samples 20000 \
  --perturbations 1:5,3:15,5:25 \
  --icp-mode point_to_point --seed 2026
cd ..

# 3. Generate analysis figures and tables
python3 scripts/analyze_icp_ablation.py \
  --in results/icp_weight_ablation --bootstrap 5000

```
