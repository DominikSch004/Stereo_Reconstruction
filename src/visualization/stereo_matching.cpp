#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "DTULoader.hpp"
#include "Rectification.hpp"
#include "Disparity.hpp"
#include "ImgUtils.hpp"

int main()
{
    const std::string leftPath  = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";
    const std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);
    if (!pair.imageLeft.data || !pair.imageRight.data) return -1;

    cv::Mat grayL = toGray(pair.imageLeft);
    cv::Mat grayR = toGray(pair.imageRight);

    // Hardcode dataset boundaries (DTU setup parameters)
    int minDisp = 40;
    int numDisp = 64; 
    int blockSize = 7;

    // Generate Custom Match Cost Map (try diff metrics by changing the last argument)
    std::cout << "Computing Custom SAD Disparity Map...\n";
    cv::Mat customDisp = Disparity::computeCustom(grayL, grayR, minDisp, numDisp, blockSize, DisparityMethod::SAD);

    // enerate OpenCV Semi-Global Baseline
    std::cout << "Computing OpenCV SGBM Baseline...\n";
    cv::Mat sgbmDisp = Disparity::computeSGBM(grayL, grayR, minDisp, numDisp, blockSize);

    // Normalize maps to [0, 255] bounds for rendering contrast clarity
    cv::Mat customViz, sgbmViz;
    cv::normalize(customDisp, customViz, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(sgbmDisp,   sgbmViz,   0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat customColor, sgbmColor;
    cv::applyColorMap(customViz, customColor, cv::COLORMAP_JET);
    cv::applyColorMap(sgbmViz,   sggbmColor,   cv::COLORMAP_JET);

    cv::Mat combined;
    cv::hconcat(customColor, sgbmColor, combined);
    cv::imshow("Dense Disparity: Custom SAD (Left) vs OpenCV SGBM (Right)", combined);
    cv::waitKey(0);

    return 0;
}