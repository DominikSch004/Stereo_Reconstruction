#include "ImgUtils.hpp"

// Convert FreeImageB (RGBA) to OpenCV grayscale cv::Mat
cv::Mat toGray(const FreeImageB& fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}