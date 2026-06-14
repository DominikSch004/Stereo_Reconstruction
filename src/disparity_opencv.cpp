#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include "StereoPipeline.hpp"

int main() {
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res)) return -1;

    // Compute disparity map
    auto stereo = cv::StereoSGBM::create(
        0,    // minDisparity
        64,   // numDisparities (must be divisible by 16)
        11     // blockSize
    );

    cv::Mat disparity16;
    stereo->compute(res.rectLeft, res.rectRight, disparity16);

    cv::Mat disparity;
    disparity16.convertTo(disparity, CV_32F, 1.0 / 16.0);

    // Normalize for visualization
    cv::Mat disp8;
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity", disp8);
    cv::waitKey(0);

    const int maxDisp = 64;
    buildAndSavePLY(disparity, res, maxDisp, "pointcloud_opencv.ply");

    return 0;   
}