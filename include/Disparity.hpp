#pragma once

#include <opencv2/core.hpp>

class Disparity
{
public:
    static cv::Mat computeSSD(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);

    static cv::Mat computeSAD(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);

    static cv::Mat computeNCC(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);

    static cv::Mat computeSGBM(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);

    static cv::Mat computeOpenCVSGBM(
        const cv::Mat& left,
        const cv::Mat& right,
        int minDisp,
        int numDisp,
        int blockSize);
};