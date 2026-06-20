#include "Disparity.hpp"
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Helper to handle signed horizontal image shifting safely
static cv::Mat shiftImage(const cv::Mat& src, int d)
{
    cv::Mat shifted = cv::Mat::zeros(src.rows, src.cols, src.type());
    if (d >= 0) {
        if (d < src.cols) {
            src(cv::Rect(0, 0, src.cols - d, src.rows))
               .copyTo(shifted(cv::Rect(d, 0, src.cols - d, src.rows)));
        }
    } else {
        int absD = -d;
        if (absD < src.cols) {
            src(cv::Rect(absD, 0, src.cols - absD, src.rows))
               .copyTo(shifted(cv::Rect(0, 0, src.cols - absD, src.rows)));
        }
    }
    return shifted;
}

cv::Mat Disparity::computeSSD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSSD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, float(minDisp));
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int i = 0; i < numDisp; ++i)
    {
        int d = minDisp + i;
        cv::Mat rightShifted = shiftImage(rightF, d);
        cv::Mat diff = leftF - rightShifted;
        
        cv::Mat ssd;
        cv::filter2D(diff.mul(diff), ssd, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = ssd < bestSSD;
        ssd.copyTo(bestSSD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0, 0, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, left.rows - h, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, 0, h, left.rows)) = float(minDisp);
    disp(cv::Rect(left.cols - h, 0, h, left.rows)) = float(minDisp);
    return disp;
}

cv::Mat Disparity::computeSAD(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSAD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, float(minDisp));
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int i = 0; i < numDisp; ++i)
    {
        int d = minDisp + i;
        cv::Mat rightShifted = shiftImage(rightF, d);

        cv::Mat absDiff;
        cv::absdiff(leftF, rightShifted, absDiff);

        cv::Mat sad;
        cv::filter2D(absDiff, sad, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = sad < bestSAD;
        sad.copyTo(bestSAD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0, 0, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, left.rows - h, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, 0, h, left.rows)) = float(minDisp);
    disp(cv::Rect(left.cols - h, 0, h, left.rows)) = float(minDisp);
    return disp;
}

cv::Mat Disparity::computeNCC(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    const float n = float(blockSize * blockSize);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    cv::Mat sumL, sumLL;
    cv::filter2D(leftF,            sumL,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(leftF.mul(leftF), sumLL, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat bestNCC(left.rows, left.cols, CV_32F, -std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, float(minDisp));

    for (int i = 0; i < numDisp; ++i)
    {
        int d = minDisp + i;
        cv::Mat rightShifted = shiftImage(rightF, d);

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
    disp(cv::Rect(0, 0, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, left.rows - h, left.cols, h)) = float(minDisp);
    disp(cv::Rect(0, 0, h, left.rows)) = float(minDisp);
    disp(cv::Rect(left.cols - h, 0, h, left.rows)) = float(minDisp);
    return disp;
}

cv::Mat Disparity::computeSGBM(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    auto sgbm = cv::StereoSGBM::create(
        minDisp, numDisp, blockSize,
        8 * 3 * blockSize * blockSize,
        32 * 3 * blockSize * blockSize,
        1, 0, 10, 100, 32,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);
    return dispFloat;
}

cv::Mat Disparity::computeOpenCVSGBM(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    auto stereo = cv::StereoSGBM::create(
        minDisp, numDisp, blockSize,
        8 * blockSize * blockSize,
        32 * blockSize * blockSize,
        1, 0, 10, 100, 2);

    cv::Mat disparity16;
    stereo->compute(left, right, disparity16);

    cv::Mat disparity;
    disparity16.convertTo(disparity, CV_32F, 1.0 / 16.0);
    return disparity;
}