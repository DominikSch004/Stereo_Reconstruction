cmake_minimum_required(VERSION 3.14)
project(StereoReconstructionICP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui calib3d features2d)
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

add_library(stereo_core STATIC
    src/Pipeline.cpp
    src/components/SparseKeyPointMatcher.cpp
    src/components/FundamentalMatrix.cpp
    src/components/Rectification.cpp
    src/components/Disparity.cpp
    src/components/Triangulation.cpp
    src/components/ICP.cpp
    src/utils/ImgUtils.cpp
    src/utils/FreeImageHelper.cpp
    src/utils/PlyUtils.cpp
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

function(register_visualisation_target target_name source_path)
    add_executable(${target_name} ${source_path})
    target_include_directories(${target_name} PRIVATE include)
    target_link_libraries(${target_name} PRIVATE stereo_core)
endfunction()

register_visualisation_target(SparseKeyPointMatching src/visualization/sparse_key_point_matching.cpp)
register_visualisation_target(EightPointRansac       src/visualization/8point_ransac.cpp)
register_visualisation_target(StereoRectification    src/visualization/stereo_rectification.cpp)
register_visualisation_target(StereoMatching         src/visualization/stereo_matching.cpp)
register_visualisation_target(PipelineOpenCVEvaluation     src/visualization/pipeline_opencv_eval.cpp)
register_visualisation_target(PipelineCustomEvaluation     src/visualization/pipeline_custom_eval.cpp)

add_executable(IcpFusion
    src/IcpFusion.cpp
    src/components/ICP.cpp
)
target_include_directories(IcpFusion PRIVATE include ${CERES_INCLUDE_DIRS})
target_link_libraries(IcpFusion PRIVATE stereo_core Ceres::ceres)