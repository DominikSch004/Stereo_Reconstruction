#include "ImgUtils.hpp"

cv::Mat toCvMat(const Eigen::Matrix3d &M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M(i, j);
    return out;
}

cv::Mat toGray(const FreeImageB &fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray;
}

cv::Mat toGray(const cv::Mat &img)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat toBGR(const FreeImageB &fi)
{
    cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

Eigen::Matrix3d toEigenMat(const cv::Mat &cvMat)
{
    Eigen::Matrix3d out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out(i, j) = cvMat.at<double>(i, j);
    return out;
}

Eigen::Vector3d toEigenVec(const cv::Mat &cvMat)
{
    Eigen::Vector3d out;
    for (int i = 0; i < 3; ++i)
        out(i) = cvMat.at<double>(i, 0);
    return out;
}