# Stereo Reconstruction

3D Scanning & Motion Capture — TUM SS26  
Group 11: Ayush Datta, Dominik Schneider, Francisco Mónica, Andrés Forero

## Project Overview
Classical stereo reconstruction pipeline with two contributions:
- **Contribution A:** Improving sparse matching and essential matrix estimation via confidence-weighted correspondences
- **Contribution B:** Multi-view fusion via confidence-guided ICP alignment

## Pipeline
1. Sparse keypoint matching (SIFT + FLANN)
2. Essential matrix estimation (normalized 8-point algorithm + RANSAC)
3. Stereo rectification (Loop & Zhang 1999)
4. Dense stereo matching (SSD / SAD / NCC)
5. Disparity → depth via triangulation
6. ICP multi-view fusion
7. Mesh reconstruction (SDF + Marching Cubes)

## Dataset
[DTU MVS 2014](https://roboimagedata.compute.dtu.dk/?page_id=36) — place scenes under `data/dtu/`
Run `./scripts/download_dtu.sh` to download automatically.

## Dependencies
- OpenCV
- Eigen3
- Ceres Solver
- CMake
- FreeImage

## Build
```bash
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

## Repository Structure
- `src/` — implementation
- `include/` — headers
- `data/` — datasets
- `docs/` — pipeline design notes and paper summaries
- `scripts/` — Utility shell scripts
- `benchmark/` — evaluation metrics