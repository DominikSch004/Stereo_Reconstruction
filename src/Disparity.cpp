#include "Disparity.hpp"

#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat Disparity::computeSSD(const cv::Mat& left, const cv::Mat& right,
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

cv::Mat Disparity::computeSAD(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSAD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat absDiff;
        cv::absdiff(leftF, rightShifted, absDiff);

        cv::Mat sad;
        cv::filter2D(absDiff, sad, CV_32F, kernel,
                     cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = sad < bestSAD;
        sad.copyTo(bestSAD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols,  h))          = 0;
    disp(cv::Rect(0, left.rows - h, left.cols,  h))          = 0;
    disp(cv::Rect(0,              0,          h, left.rows))  = 0;
    disp(cv::Rect(left.cols - h,  0,          h, left.rows))  = 0;
    return disp;
}

// NCC = (n*sumLR - sumL*sumR) / sqrt((n*sumLL - sumL^2) * (n*sumRR - sumR^2))
// Computed efficiently with box filters. We MAXIMISE NCC (best match → 1).
cv::Mat Disparity::computeNCC(const cv::Mat& left, const cv::Mat& right,
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

cv::Mat Disparity::computeSGBM(
    const cv::Mat& left,
    const cv::Mat& right,
    int numDisp,
    int blockSize)
{
    auto sgbm = cv::StereoSGBM::create(
        0,
        numDisp,
        blockSize,
        8 * 3 * blockSize * blockSize,
        32 * 3 * blockSize * blockSize,
        1,
        0,
        10,
        100,
        32,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);

    return dispFloat;
}

cv::Mat Disparity::computeOpenCVSGBM(
    const cv::Mat& left,
    const cv::Mat& right,
    int minDisp,
    int numDisp,
    int blockSize)
{
    auto stereo = cv::StereoSGBM::create(
        minDisp,
        numDisp,
        blockSize,
        8 * blockSize * blockSize,
        32 * blockSize * blockSize,
        1,
        0,
        10,
        100,
        2);

    cv::Mat disparity16;
    stereo->compute(left, right, disparity16);

    cv::Mat disparity;
    disparity16.convertTo(disparity, CV_32F, 1.0 / 16.0);

    return disparity;
}