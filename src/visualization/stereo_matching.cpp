#include <iostream>
#include <cstdlib>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "Evaluator.hpp"
#include "VisualizationUtils.hpp"

int main(int argc, char **argv)
{
    // 0. Load pipeline configuration (per-step backend selection)
    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig cfg;
    try
    {
        cfg = PipelineConfig::load(configPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    cfg.print();

    // 1. Load data and run the ACTUAL pipeline so we verify the disparity that the
    //    point cloud is built from (not a divergent re-implementation).
    DTULoader loader("../data/dtu/");
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);

    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, cfg))
    {
        std::cerr << "ERROR: pipeline execution failed.\n";
        return -1;
    }

    DisparityRes dispRes = Evaluator::evaluateDisparity(res.denseDisparity, res.rectLeft, res.rectRight,
                                                         res.inPtsL, res.inPtsR, res.K,
                                                         res.R1, res.P1r, res.R2, res.P2r,
                                                         res.minDisp, res.numDisp);
    Evaluator::printDisparity(dispRes, cfg.processingScale);

    // 2. Visualization: rectified left | colorized disparity | photometric error | textureless
    //    overlay, combined into one mosaic plus each panel saved individually
    RectifyResult rect;
    rect.R1 = res.R1;
    rect.R2 = res.R2;
    rect.P1 = res.P1r;
    rect.P2 = res.P2r;
    rect.Q = res.Q;
    rect.rectLeft = res.rectLeft;
    rect.rectRight = res.rectRight;
    rect.rectColor = res.rectColor;

    const std::string outPath = std::string("disparity_verification_") +
                                (cfg.disparity == DisparityMethod::OpenCVSGBM ? "opencv" : "custom") + ".png";
    VisualizationUtils::visualizeDisparity(res.denseDisparity, rect, res.inPtsL, res.inPtsR, res.K,
                                           res.minDisp, res.numDisp, dispRes,
                                           "Disparity Verification", outPath);

    return 0;
}
