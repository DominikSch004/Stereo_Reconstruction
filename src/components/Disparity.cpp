#include "Disparity.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

cv::Mat Disparity::computeDisparity(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, DisparityMethod method)
{
    switch (method)
    {
    case DisparityMethod::OpenCVSGBM:
        return computeSGBMOpenCV(left, right, minDisp, numDisp, blockSize);
    case DisparityMethod::Custom:
        return computeCustom(left, right, minDisp, numDisp, blockSize);
    default:
        std::cout << "Failed! Select a valid disparity method";
        return cv::Mat();
    }
}

cv::Mat Disparity::computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    int numChannels = left.channels(); // images are grayscale (1) & (3) BGR
    auto sgbm = cv::StereoSGBM::create(
        minDisp,
        numDisp,
        blockSize,
        8 * numChannels * blockSize * blockSize,  // P1 smoothness penalty
        32 * numChannels * blockSize * blockSize, // P2 smoothness penalty
        1,                                        // disp12MaxDiff
        0,                                        // preFilterCap
        10,                                       // uniquenessRatio
        100,                                      // speckleWindowSize
        32,                                       // speckleRange
        cv::StereoSGBM::MODE_SGBM);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    // Convert fixed-point 16-bit integer output representations back to real floating point coordinates
    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);
    return dispFloat;
}

cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    // TODO: Disparity custom calculation
    std::cout << "Disparity not yet implemented \n";
    return cv::Mat();
}
