#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "FreeImageHelper.h"

// Convert FreeImageB (RGBA) to OpenCV grayscale cv::Mat
cv::Mat toGray(const FreeImageB& fi);

// Convert FreeImage buffers to OpenCV mats
cv::Mat toBGR(const FreeImageB& fi);