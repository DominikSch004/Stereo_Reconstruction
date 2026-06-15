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

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/FreeImageHelper.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)



add_executable(SiftFlann
    src/sift_flann.cpp
    src/sift_flann_matcher.cpp
    src/match_serializer.cpp
    src/FreeImageHelper.cpp
    src/ImgUtils.cpp
)

target_include_directories(SiftFlann PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(SiftFlann PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)

add_executable(EightPointRansac
    src/8point_ransac.cpp
    src/8point_algorithm.cpp
    src/fundamental_matrix.cpp
    src/FreeImageHelper.cpp
    src/match_serializer.cpp
    src/ImgUtils.cpp
)

target_include_directories(EightPointRansac PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(EightPointRansac PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)

add_executable(Rectification
    src/rectification.cpp
    src/match_serializer.cpp
    src/FreeImageHelper.cpp
    src/ImgUtils.cpp
    src/fundamental_matrix.cpp
    src/8point_algorithm.cpp
)

target_include_directories(Rectification PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(Rectification PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)

add_executable(DisparityOpenCV
    src/disparity_opencv.cpp
    src/FreeImageHelper.cpp
)

target_include_directories(DisparityOpenCV PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(DisparityOpenCV PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)

foreach(METHOD ssd sad ncc sgbm)
    add_executable(Disparity_${METHOD}
        src/disparity_${METHOD}.cpp
        src/FreeImageHelper.cpp
    )
    target_include_directories(Disparity_${METHOD} PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
    )
    target_link_libraries(Disparity_${METHOD} PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Ceres::ceres
        FreeImage::FreeImage
    )
endforeach()

add_executable(IcpFusion
    src/icp_fusion.cpp
    src/FreeImageHelper.cpp
    src/fundamental_matrix.cpp
    src/8point_algorithm.cpp
)

target_include_directories(IcpFusion PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(IcpFusion PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Ceres::ceres
    FreeImage::FreeImage
)
