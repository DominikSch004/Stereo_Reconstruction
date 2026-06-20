cmake_minimum_required(VERSION 3.14)
project(StereoReconstructionICP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

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

add_library(stereo_core
    src/components/fundamental_matrix.cpp
    src/components/8point_algorithm.cpp
    src/components/Rectification_compute.cpp
    src/components/SparseKeyPointMatcher.cpp
    src/Utils/ImgUtils.cpp
    src/Utils/FreeImageHelper.cpp
    src/Utils/PlyUtils.cpp
    src/components/Disparity.cpp
    src/components/Triangulation.cpp
    src/Pipeline.cpp
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
)

add_executable(${PROJECT_NAME}
    src/main.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE include ${CERES_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE stereo_core Ceres::ceres)

add_executable(SparseKeyPointMatching src/visualization/sparse_key_point_matching.cpp)
target_include_directories(SparseKeyPointMatching PRIVATE include)
target_link_libraries(SparseKeyPointMatching PRIVATE stereo_core)

add_executable(EightPointRansac src/visualization/8point_ransac.cpp)
target_include_directories(EightPointRansac PRIVATE include)
target_link_libraries(EightPointRansac PRIVATE stereo_core)

add_executable(Rectification src/visualization/rectification.cpp)
target_include_directories(Rectification PRIVATE include)
target_link_libraries(Rectification PRIVATE stereo_core)

add_executable(IcpFusion
    src/icp_fusion.cpp
    src/ICP.cpp
    src/Cloud.cpp
)
target_include_directories(IcpFusion PRIVATE include ${CERES_INCLUDE_DIRS})
target_link_libraries(IcpFusion PRIVATE stereo_core Ceres::ceres)