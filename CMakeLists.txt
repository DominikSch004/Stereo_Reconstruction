cmake_minimum_required(VERSION 3.14)
project(StereoReconstructionICP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui calib3d features2d)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)

find_path(FREEIMAGE_INCLUDE_DIR
    NAMES FreeImage.h
    NO_CACHE
    REQUIRED
)

find_library(FREEIMAGE_LIBRARY
    NAMES freeimage FreeImage
    NO_CACHE
    REQUIRED
)

add_library(FreeImage::FreeImage UNKNOWN IMPORTED)
set_target_properties(FreeImage::FreeImage PROPERTIES
    IMPORTED_LOCATION "${FREEIMAGE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FREEIMAGE_INCLUDE_DIR}"
)

add_library(stereo_core STATIC
    src/Pipeline.cpp
    src/PipelineConfig.cpp
    src/components/SparseKeyPointMatcher.cpp
    src/components/FundamentalMatrix.cpp
    src/components/Rectification.cpp
    src/components/Disparity.cpp
    src/components/Triangulation.cpp
    src/components/ICP.cpp
    src/components/PoissonReconstruction.cpp
    src/utils/ImgUtils.cpp
    src/utils/FreeImageHelper.cpp
    src/utils/PlyUtils.cpp
    src/utils/NearestNeighborSearch.cpp
    src/utils/IcpUtils.cpp
    src/utils/MeshUtils.cpp
    src/utils/GeometryUtils.cpp
    src/utils/VisualizationUtils.cpp
    benchmark/Evaluator.cpp
)

target_include_directories(stereo_core PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(stereo_core PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    FreeImage::FreeImage
    glog::glog
    Ceres::ceres
)

function(register_visualisation_target target_name source_path)
    add_executable(${target_name} ${source_path})
    target_include_directories(${target_name} PRIVATE include)
    target_link_libraries(${target_name} PRIVATE stereo_core)
endfunction()

# Main stereo reconstruction pipeline (step backends selected via config.yaml)
add_executable(StereoReconstruction src/main.cpp)
target_include_directories(StereoReconstruction PRIVATE include)
target_link_libraries(StereoReconstruction PRIVATE stereo_core)

# ICP fusion pipeline (runs the stereo pipeline per view pair, then fuses via ICP)
add_executable(IcpFusion src/IcpFusion.cpp)
target_include_directories(IcpFusion PRIVATE include ${CERES_INCLUDE_DIRS})
target_link_libraries(IcpFusion PRIVATE stereo_core glog::glog Ceres::ceres)

# Synthetic ICP verification (known-transform recovery, exit code 0 = all checks pass)
add_executable(IcpVerification benchmark/IcpVerification.cpp)
target_include_directories(IcpVerification PRIVATE include)
target_link_libraries(IcpVerification PRIVATE stereo_core)

# DTU ICP benchmark (unweighted vs confidence-weighted, chamfer vs GT scan)
add_executable(IcpBenchmark benchmark/IcpBenchmark.cpp)
target_include_directories(IcpBenchmark PRIVATE include)
target_link_libraries(IcpBenchmark PRIVATE stereo_core)

# Chamfer accuracy of saved fused/per-pair clouds against the DTU GT scan
add_executable(FusedCloudEval benchmark/FusedCloudEval.cpp)
target_include_directories(FusedCloudEval PRIVATE include)
target_link_libraries(FusedCloudEval PRIVATE stereo_core)

# Visualization / verification executables
register_visualisation_target(SparseKeyPointMatching src/visualization/sparse_key_point_matching.cpp)
register_visualisation_target(EightPointRansac       src/visualization/8point_ransac.cpp)
register_visualisation_target(StereoRectification    src/visualization/stereo_rectification.cpp)
register_visualisation_target(StereoMatching         src/visualization/stereo_matching.cpp)
register_visualisation_target(TriangulationVerification     src/visualization/triangulation_verification.cpp)
register_visualisation_target(MeshReconstruction     src/visualization/mesh_reconstruction.cpp)
register_visualisation_target(PoissonReconstruction  src/visualization/poisson_reconstruction.cpp)
register_visualisation_target(PoseRecoveryComparison src/visualization/pose_recovery_comparison.cpp)