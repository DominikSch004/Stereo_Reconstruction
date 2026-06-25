#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include "FreeImageHelper.h"

// Convert Eigen 3x3 to cv::Mat (CV_64F)
cv::Mat toCvMat(const Eigen::Matrix3d &M);

// Convert FreeImageB (RGBA) to OpenCV grayscale cv::Mat
cv::Mat toGray(const FreeImageB &fi);

// Convert cv::Mat img to gray
cv::Mat toGray(const cv::Mat &img);

// Convert FreeImage buffers to OpenCV mats
cv::Mat toBGR(const FreeImageB &fi);