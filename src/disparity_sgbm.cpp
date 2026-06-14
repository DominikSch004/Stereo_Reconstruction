#include <opencv2/calib3d.hpp>
#include "StereoPipeline.hpp"

int main(int argc, char** argv)
{
    if (argc != 3) { std::cerr << "Usage: " << argv[0] << " <left> <right>\n"; return -1; }

    PipelineResult res;
    if (!runPipeline(argv[1], argv[2], res)) return -1;

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
