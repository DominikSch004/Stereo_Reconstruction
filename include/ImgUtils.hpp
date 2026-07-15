#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>

// Convert Eigen 3x3 to cv::Mat (CV_64F)
cv::Mat toCvMat(const Eigen::Matrix3d &M);

// Convert cv::Mat img to gray
cv::Mat toGray(const cv::Mat &img);

// Convert 3x3 OpenCV matrix (CV_64F) to Eigen Matrix3d
Eigen::Matrix3d toEigenMat(const cv::Mat &cvMat);

// Convert 3x1 OpenCV matrix (CV_64F) to Eigen Vector3d
Eigen::Vector3d toEigenVec(const cv::Mat &cvMat);