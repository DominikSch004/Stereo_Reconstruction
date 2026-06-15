#include "StereoPipeline.hpp"
#include "Disparity.hpp"

int main()
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res))
        return -1;

    const int maxDisp = 128;
    const int blockSz = 11;

    cv::Mat disp =
        Disparity::computeSAD(
            res.rectLeft,
            res.rectRight,
            maxDisp,
            blockSz);

    buildAndSavePLY(
        disp,
        res,
        maxDisp,
        "pointcloud_sad.ply");

    return 0;
}