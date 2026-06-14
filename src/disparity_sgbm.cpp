#include <opencv2/calib3d.hpp>
#include "StereoPipeline.hpp"

int main(int argc, char** argv)
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res)) return -1;

    const int numDisp = 128, blockSz = 9;
    std::cout << "SGBM disparity (numDisp=" << numDisp << ", block=" << blockSz << ")...\n";

    auto sgbm = cv::StereoSGBM::create(
        0, numDisp, blockSz,
        8  * 3 * blockSz * blockSz,
        32 * 3 * blockSz * blockSz,
        1, 0, 10, 100, 32,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp16;
    sgbm->compute(res.rectLeft, res.rectRight, disp16);

    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);

    buildAndSavePLY(dispFloat, res, numDisp, "pointcloud_sgbm.ply");
    return 0;
}
