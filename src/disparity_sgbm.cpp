#include "StereoPipeline.hpp"
#include "Disparity.hpp"

int main()
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res))
        return -1;

    const int numDisp = 128;
    const int blockSz = 9;

    cv::Mat disp =
        Disparity::computeSGBM(
            res.rectLeft,
            res.rectRight,
            numDisp,
            blockSz);

    buildAndSavePLY(
        disp,
        res,
        numDisp,
        "pointcloud_sgbm.ply");

    return 0;
}
