#include <limits>
#include <opencv2/imgproc.hpp>
#include "StereoPipeline.hpp"

static cv::Mat computeSSD(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSSD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat diff = leftF - rightShifted;
        cv::Mat ssd;
        cv::filter2D(diff.mul(diff), ssd, CV_32F, kernel,
                     cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = ssd < bestSSD;
        ssd.copyTo(bestSSD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols,  h))           = 0;
    disp(cv::Rect(0, left.rows - h, left.cols,  h))           = 0;
    disp(cv::Rect(0,              0,          h, left.rows))   = 0;
    disp(cv::Rect(left.cols - h,  0,          h, left.rows))   = 0;
    return disp;
}

int main(int argc, char** argv)
{
    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res)) return -1;

    const int maxDisp = 128, blockSz = 11;
    std::cout << "SSD disparity (maxDisp=" << maxDisp << ", block=" << blockSz << ")...\n";
    cv::Mat disp = computeSSD(res.rectLeft, res.rectRight, maxDisp, blockSz);

    buildAndSavePLY(disp, res, maxDisp, "pointcloud_ssd.ply");
    return 0;
}
