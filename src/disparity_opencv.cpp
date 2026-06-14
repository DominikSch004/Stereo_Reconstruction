#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include "StereoPipeline.hpp"

int main() {
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    // Calibrated rectification using the DTU ground-truth poses.
    if (!runPipeline(leftPath, rightPath, res)) return -1;

    // Dense disparity over the geometry-derived search range.
    const int block = 5;
    auto stereo = cv::StereoSGBM::create(
        res.minDisp, res.numDisp, block,
        8  * block * block,      // P1
        32 * block * block,      // P2
        1,                       // disp12MaxDiff
        0,                       // preFilterCap
        10,                      // uniquenessRatio
        100,                     // speckleWindowSize
        2);                      // speckleRange

    cv::Mat disparity16;
    stereo->compute(res.rectLeft, res.rectRight, disparity16);

    cv::Mat disparity;
    disparity16.convertTo(disparity, CV_32F, 1.0 / 16.0);

    buildAndSavePLY(disparity, res, res.numDisp, "pointcloud_opencv.ply");

    std::cout << "Open pointcloud_opencv.ply in MeshLab/CloudCompare to view it.\n";
    return 0;
}