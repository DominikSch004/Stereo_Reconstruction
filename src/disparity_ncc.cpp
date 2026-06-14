#include <limits>
#include <opencv2/imgproc.hpp>
#include "StereoPipeline.hpp"

// NCC = (n*sumLR - sumL*sumR) / sqrt((n*sumLL - sumL^2) * (n*sumRR - sumR^2))
// Computed efficiently with box filters. We MAXIMISE NCC (best match → 1).
static cv::Mat computeNCC(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    const float n = float(blockSize * blockSize);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    // Precompute left statistics (don't change with d)
    cv::Mat sumL, sumLL;
    cv::filter2D(leftF,            sumL,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(leftF.mul(leftF), sumLL, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat bestNCC(left.rows, left.cols, CV_32F, -std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat sumR, sumRR, sumLR;
        cv::filter2D(rightShifted,                  sumR,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(rightShifted.mul(rightShifted), sumRR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(leftF.mul(rightShifted),        sumLR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat numerator   = n * sumLR - sumL.mul(sumR);
        cv::Mat denominator;
        cv::sqrt((n * sumLL - sumL.mul(sumL)).mul(n * sumRR - sumR.mul(sumR)), denominator);

        cv::Mat ncc = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        cv::divide(numerator, denominator + 1e-6f, ncc);

        cv::Mat better = ncc > bestNCC;
        ncc.copyTo(bestNCC, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols,  h))          = 0;
    disp(cv::Rect(0, left.rows - h, left.cols,  h))          = 0;
    disp(cv::Rect(0,              0,          h, left.rows))  = 0;
    disp(cv::Rect(left.cols - h,  0,          h, left.rows))  = 0;
    return disp;
}

int main(int argc, char** argv)
{
    //if (argc != 3) { std::cerr << "Usage: " << argv[0] << " <left> <right>\n"; return -1; }

    std::string leftPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png");
    std::string rightPath("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png");

    PipelineResult res;
    if (!runPipeline(leftPath, rightPath, res)) return -1;

    const int maxDisp = 128, blockSz = 11;
    std::cout << "NCC disparity (maxDisp=" << maxDisp << ", block=" << blockSz << ")...\n";
    cv::Mat disp = computeNCC(res.rectLeft, res.rectRight, maxDisp, blockSz);

    buildAndSavePLY(disp, res, maxDisp, "pointcloud_ncc.ply");
    return 0;
}
