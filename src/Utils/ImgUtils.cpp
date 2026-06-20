#include "ImgUtils.hpp"


cv::Mat toCvMat(const Eigen::Matrix3d& M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M(i, j);
    return out;
}

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