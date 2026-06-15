#include "ImgUtils.hpp"

cv::Mat toGray(const FreeImageB& fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}

cv::Mat toBGR(const FreeImageB& fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}