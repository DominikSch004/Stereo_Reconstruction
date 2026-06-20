#include "Disparity.hpp"
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

/**
 * @brief Helper to handle signed horizontal image shifting safely.
 * @details Shifts the right image along the epipolar line by disparity 'd' 
 * to allow vectorized matrix-wide differences.
 */
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

/**
 * @brief Helper to set boundary half-windows to a default value to avoid uninitialized noise.
 */
static void fillBorders(cv::Mat& disp, int blockSize, float value)
{
    int h = blockSize / 2;
    disp(cv::Rect(0, 0, disp.cols, h)) = value; // Top border
    disp(cv::Rect(0, disp.rows - h, disp.cols, h)) = value; // Bottom border
    disp(cv::Rect(0, 0, h, disp.rows)) = value; // Left border
    disp(cv::Rect(disp.cols - h, 0, h, disp.rows)) = value; // Right border
}

cv::Mat Disparity::computeCustom(
    const cv::Mat& left,
    const cv::Mat& right,
    int minDisp,
    int numDisp,
    int blockSize,
    DisparityMethod method)
{
    switch (method) {
        case DisparityMethod::SSD:
            return computeSSD(left, right, minDisp, numDisp, blockSize);
        case DisparityMethod::NCC:
            return computeNCC(left, right, minDisp, numDisp, blockSize);
        case DisparityMethod::SAD:
        default:
            return computeSAD(left, right, minDisp, numDisp, blockSize);
    }
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
        
        // Sum of Squared Differences via box-filtering
        cv::Mat ssd;
        cv::filter2D(diff.mul(diff), ssd, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        // Update step: minimize tracking error
        cv::Mat better = ssd < bestSSD;
        ssd.copyTo(bestSSD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    fillBorders(disp, blockSize, float(minDisp));
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

        // Sum of Absolute Differences via box-filtering
        cv::Mat sad;
        cv::filter2D(absDiff, sad, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        // Update step: minimize tracking error
        cv::Mat better = sad < bestSAD;
        sad.copyTo(bestSAD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    fillBorders(disp, blockSize, float(minDisp));
    return disp;
}

cv::Mat Disparity::computeNCC(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    const float n = float(blockSize * blockSize);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    // Precompute sums for the left image to optimize NCC calculations
    cv::Mat sumL, sumLL;
    cv::filter2D(leftF,            sumL,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(leftF.mul(leftF), sumLL, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat bestNCC(left.rows, left.cols, CV_32F, -std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, float(minDisp));

    for (int i = 0; i < numDisp; ++i)
    {
        int d = minDisp + i;
        cv::Mat rightShifted = shiftImage(rightF, d);

        // Compute the sums of the right image, squared right image, and cross-term for NCC
        cv::Mat sumR, sumRR, sumLR;
        cv::filter2D(rightShifted,                  sumR,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(rightShifted.mul(rightShifted), sumRR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(leftF.mul(rightShifted),        sumLR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        // NCC Mathematical Formulations: (n*sum(xy) - sum(x)*sum(y)) / sqrt([n*sum(x^2)-sum(x)^2] * [n*sum(y^2)-sum(y)^2])
        cv::Mat numerator   = n * sumLR - sumL.mul(sumR);
        cv::Mat denominator;
        cv::sqrt((n * sumLL - sumL.mul(sumL)).mul(n * sumRR - sumR.mul(sumR)), denominator);

        cv::Mat ncc = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        cv::divide(numerator, denominator + 1e-6f, ncc); // Add epsilon to prevent 0 division

        // Update step: maximize cross-correlation alignment
        cv::Mat better = ncc > bestNCC;
        ncc.copyTo(bestNCC, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    fillBorders(disp, blockSize, float(minDisp));
    return disp;
}

cv::Mat Disparity::computeSGBMOpenCV(const cv::Mat& left, const cv::Mat& right, int minDisp, int numDisp, int blockSize)
{
    auto sgbm = cv::StereoSGBM::create(
        minDisp, 
        numDisp, 
        blockSize,
        8 * 3 * blockSize * blockSize,   // P1 smoothness penalty
        32 * 3 * blockSize * blockSize,  // P2 smoothness penalty
        1,   // disp12MaxDiff
        0,   // preFilterCap
        10,  // uniquenessRatio
        100, // speckleWindowSize
        32,  // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    // Convert fixed-point 16-bit integer output representations back to real floating point coordinates
    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);
    return dispFloat;
}
