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