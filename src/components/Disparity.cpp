#include "Disparity.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cstdint>

cv::Mat Disparity::computeDisparity(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, DisparityMethod method)
{
    switch (method)
    {
    case DisparityMethod::OpenCVSGBM:
        return computeSGBMOpenCV(left, right, minDisp, numDisp, blockSize);
    case DisparityMethod::Custom:
        return computeCustom(left, right, minDisp, numDisp, blockSize);
    default:
        std::cout << "Failed! Select a valid disparity method";
        return cv::Mat();
    }
}

cv::Mat Disparity::computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    int numChannels = left.channels(); // images are grayscale (1) & (3) BGR
    auto sgbm = cv::StereoSGBM::create(
        minDisp,
        numDisp,
        blockSize,
        8 * numChannels * blockSize * blockSize,  // P1 smoothness penalty
        32 * numChannels * blockSize * blockSize, // P2 smoothness penalty
        1,                                        // disp12MaxDiff
        0,                                        // preFilterCap
        10,                                       // uniquenessRatio
        100,                                      // speckleWindowSize
        32,                                       // speckleRange
        cv::StereoSGBM::MODE_SGBM);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    // Convert fixed-point 16-bit integer output representations back to real floating point coordinates
    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);
    return dispFloat;
}

void Disparity::computeBTIntervals(const cv::Mat &src, cv::Mat &Imin, cv::Mat &Imax)
{
    Imin.create(src.size(), CV_32F);
    Imax.create(src.size(), CV_32F);

    const int rows = src.rows;
    const int cols = src.cols;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float x = src.at<float>(r, c);

            float x_minus = (c > 0) ? src.at<float>(r, c - 1) : x; // clamping to edge
            float x_plus = (c < cols - 1) ? src.at<float>(r, c + 1) : x;

            float x_m_interp = (x + x_minus) * 0.5f;
            float x_p_interp = (x + x_plus) * 0.5f;

            Imin.at<float>(r, c) = std::min({x, x_m_interp, x_p_interp});
            Imax.at<float>(r, c) = std::max({x, x_p_interp, x_m_interp});
        }
    }
}

std::vector<uint16_t> Disparity::computeCostVolume(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp)
{
    // Birchfield & Tomasi (BT) 98 - Pixel Dissimilarity d(xi, yi) Section 2.1.2
    cv::Mat Imin_left, Imax_left, Imin_right, Imax_right;
    computeBTIntervals(left, Imin_left, Imax_left);
    computeBTIntervals(right, Imin_right, Imax_right);

    // The SGM paper: scaling costs to roughly 11 bits keeps the 16-directions
    // sum within the 16-bit range used the aggregated costs.
    std::vector<uint16_t> costVolume(static_cast<size_t>(numDisp) * left.rows * left.cols, 2047); // max 11-bit cost for out-of-bounds

    for (int r = 0; r < left.rows; ++r)
    {
        for (int c = 0; c < left.cols; ++c)
        {
            for (int d = 0; d < numDisp; ++d)
            {
                int actual_disp = d + minDisp;
                int c_right = c - actual_disp; // d = left - right

                if (c_right >= 0 && c_right < right.cols)
                {
                    float left_val = left.at<float>(r, c);
                    float Imin_left_val = Imin_left.at<float>(r, c);
                    float Imax_left_val = Imax_left.at<float>(r, c);

                    float right_val = right.at<float>(r, c_right);
                    float Imin_right_val = Imin_right.at<float>(r, c_right);
                    float Imax_right_val = Imax_right.at<float>(r, c_right);

                    float cost_LR = std::max({0.0f, left_val - Imax_right_val, Imin_right_val - left_val});
                    float cost_RL = std::max({0.0f, right_val - Imax_left_val, Imin_left_val - right_val});

                    float cost_value = std::min(cost_LR, cost_RL);

                    int idx = (r * left.cols + c) * numDisp + d;
                    costVolume[idx] = static_cast<uint16_t>(std::min(cost_value, 2047.0f));
                }
            }
        }
    }
    return costVolume;
}

std::vector<uint16_t> Disparity::aggregateCost(const std::vector<uint16_t> &C, int rows, int cols, int numDisp, int P1, int P2)
{
    std::vector<uint16_t> S(rows * cols * numDisp, 0); // aggregated cost volume
    std::vector<int> prev(numDisp, 0); // previous pixel's Lr values along the path
    std::vector<int> curr(numDisp, 0); // this pixel's Lr values, computed from prev

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int minPrev = *std::min_element(prev.begin(), prev.end()); // minimum cost from the previous pixel in the path

            for (int d = 0; d < numDisp; ++d)
            {
                int idx = (r * cols + c) * numDisp + d;

                if (c == 0) // first pixel in the row, no predecessor: Lr(p, d) = C(p, d)
                {
                    curr[d] = static_cast<int>(C[idx]);
                    S[idx] = static_cast<uint16_t>(curr[d]);
                }
                else
                {
                    // Lr(p, d) = C(p, d)
                    //            + min(
                    //                Lr(p-r, d),           // same disparity - no penalty
                    //                Lr(p-r, d-1) + P1,     // disparity change of 1
                    //                Lr(p-r, d+1) + P1,     // disparity change of 1
                    //                min_k Lr(p-r, k) + P2  // any larger change
                    //              )
                    //            - min_k Lr(p-r, k)         // subtract to keep values bounded (16-bit safe)
                    int best = prev[d];                             // Lr(p-r, d) - same disparity, no penalty
                    if (d > 0)
                        best = std::min(best, prev[d - 1] + P1);    // Lr(p-r, d-1) + P1 - disparity change of 1
                    if (d < numDisp - 1)
                        best = std::min(best, prev[d + 1] + P1);    // Lr(p-r, d+1) + P1 - disparity change of 1
                    best = std::min(best, minPrev + P2);            // min_k Lr(p-r, k) + P2 - any larger change

                    int cost = static_cast<int>(C[idx]);            // C(p, d)
                    curr[d] = cost + best - minPrev;                // ... - min_k Lr(p-r, k)
                    S[idx] = static_cast<uint16_t>(curr[d]);
                }
            }
            prev = curr;
        }
    }

    return S;
}


cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    const int P1 = 8;  // smoothness penalty for disparity change of 1
    const int P2 = 32; // smoothness penalty for disparity change greater than

    cv::Mat leftF, rightF;
    left.convertTo(leftF, CV_32F);
    right.convertTo(rightF, CV_32F);

    std::vector<uint16_t> costVolume = computeCostVolume(leftF, rightF, minDisp, numDisp);
    std::vector<uint16_t> aggregatedCost = aggregateCost(costVolume, left.rows, left.cols, numDisp, P1, P2);

    // Cost volume
    cv::Mat disparity = cv::Mat::zeros(left.rows, left.cols, CV_32F);

    return disparity;
}
