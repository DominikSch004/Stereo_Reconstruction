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

std::vector<uint16_t> Disparity::computeCostVolume(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase)
{
    // Birchfield & Tomasi (BT) 98 - Pixel Dissimilarity d(xi, yi) Section 2.1.2
    
    // L-R consistency check
    // The returned cost volume is always indexed over "base"'s pixel positions; "match" is
    // the other image being searched. d = xL - xR always, so the search direction flips
    // between the two passes

    // rightBase param selects which image the returned cost volume is indexed over
    // so a left-base pixel xL looks up the right image at xL - d (rightBase=false),
    // while a right-base pixel xR looks up the left image at xR + d (rightBase=true)

    const cv::Mat &base = rightBase ? right : left;
    const cv::Mat &match = rightBase ? left : right;
    const int sign = rightBase ? 1 : -1;

    cv::Mat Imin_base, Imax_base, Imin_match, Imax_match;
    computeBTIntervals(base, Imin_base, Imax_base);
    computeBTIntervals(match, Imin_match, Imax_match);

    // The SGM paper: scaling costs to roughly 11 bits keeps the 16-directions
    // sum within the 16-bit range used the aggregated costs.
    std::vector<uint16_t> costVolume(static_cast<size_t>(numDisp) * base.rows * base.cols, 2047); // max 11-bit cost for out-of-bounds

    for (int r = 0; r < base.rows; ++r)
    {
        for (int c = 0; c < base.cols; ++c)
        {
            for (int d = 0; d < numDisp; ++d)
            {
                int actual_disp = d + minDisp;
                int matchCol = c + sign * actual_disp; // xR = xL - d, or xL = xR + d

                if (matchCol >= 0 && matchCol < match.cols)
                {
                    float base_val = base.at<float>(r, c);
                    float Imin_base_val = Imin_base.at<float>(r, c);
                    float Imax_base_val = Imax_base.at<float>(r, c);

                    float match_val = match.at<float>(r, matchCol);
                    float Imin_match_val = Imin_match.at<float>(r, matchCol);
                    float Imax_match_val = Imax_match.at<float>(r, matchCol);

                    float cost_base_match = std::max({0.0f, base_val - Imax_match_val, Imin_match_val - base_val});
                    float cost_match_base = std::max({0.0f, match_val - Imax_base_val, Imin_base_val - match_val});

                    float cost_value = std::min(cost_base_match, cost_match_base);

                    int idx = (r * base.cols + c) * numDisp + d;
                    costVolume[idx] = static_cast<uint16_t>(std::min(cost_value, 2047.0f));
                }
            }
        }
    }
    return costVolume;
}

void Disparity::aggregateDirection(const std::vector<uint16_t> &C, std::vector<uint16_t> &S,
                                   int rows, int cols, int numDisp, int dx, int dy, int P1, int P2)
{
    std::vector<int> Lr(static_cast<size_t>(rows) * cols * numDisp);

    int yStart = (dy >= 0) ? 0 : rows - 1;
    int yEnd   = (dy >= 0) ? rows : -1;
    int yStep  = (dy >= 0) ? 1 : -1;

    int xStart = (dx >= 0) ? 0 : cols - 1;
    int xEnd   = (dx >= 0) ? cols : -1;
    int xStep  = (dx >= 0) ? 1 : -1;

    for (int y = yStart; y != yEnd; y += yStep)
    {
        for (int x = xStart; x != xEnd; x += xStep)
        {
            int idx = (y * cols + x) * numDisp;
            int px = x - dx, py = y - dy; // predecessor pixel coordinates
            bool hasPred = (px >= 0 && px < cols && py >= 0 && py < rows);

            if (!hasPred)
            {
                // boundary: Lr(p,d) = C(p,d)
                for (int d = 0; d < numDisp; ++d)
                    Lr[idx + d] = C[idx + d];

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
                int predIdx = (py * cols + px) * numDisp; // predecessor pixel index
                int minPrev = *std::min_element(&Lr[predIdx], &Lr[predIdx + numDisp]); // min_k Lr(p-r, k)

                for (int d = 0; d < numDisp; ++d)
                {
                    int best = Lr[predIdx + d]; // same disparity
                    if (d > 0)
                        best = std::min(best, Lr[predIdx + d - 1] + P1); // Lr(p-r, d-1) + P1 - disparity change of 1
                    if (d < numDisp - 1)
                        best = std::min(best, Lr[predIdx + d + 1] + P1); // Lr(p-r, d+1) + P1 - disparity change of 1
                    best = std::min(best, minPrev + P2); // min_k Lr(p-r, k) + P2 - any larger change

                    Lr[idx + d] = C[idx + d] + best - minPrev; // ... - min_k Lr(p-r, k)
                }
            }

            for (int d = 0; d < numDisp; ++d)
                S[idx + d] += Lr[idx + d];
        }
    }
}

// Full cost volume -> 16-direction aggregation -> WTA pipeline for one base image.
cv::Mat Disparity::computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2, bool rightBase)
{
    static const int dirs[16][2] = { // All 16 directions for aggregation
        {1,0},{-1,0},{0,1},{0,-1},
        {1,1},{-1,-1},{1,-1},{-1,1},
        {2,1},{-2,-1},{2,-1},{-2,1},
        {1,2},{-1,-2},{1,-2},{-1,2}
    };

    std::vector<uint16_t> costVolume = computeCostVolume(left, right, minDisp, numDisp, rightBase);

    std::vector<uint16_t> S(static_cast<size_t>(rows) * cols * numDisp, 0);
    for (auto &d : dirs)
        aggregateDirection(costVolume, S, rows, cols, numDisp, d[0], d[1], P1, P2);

    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int idx = (r * cols + c) * numDisp;
            auto minCost = std::min_element(&S[idx], &S[idx + numDisp]); // minimum aggregated cost
            int bestDispIdx = static_cast<int>(std::distance(&S[idx], minCost)); // index of best disparity
            disparity.at<float>(r, c) = static_cast<float>(minDisp + bestDispIdx);
        }
    }

    return disparity;
}

cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    const int P1 = 8;  // smoothness penalty for disparity change of 1
    const int P2 = 32; // smoothness penalty for disparity change greater than
    const int rows = left.rows;
    const int cols = left.cols;

    cv::Mat leftF, rightF;
    left.convertTo(leftF, CV_32F);
    right.convertTo(rightF, CV_32F);

    // Run the full winner takes all (WTA) disparity calculaation twice:
    // once treating the left image as the base (D_left), once the right image (D_right)
    cv::Mat dispLeft = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, false);
    cv::Mat dispRight = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, true);

    // L-R consistency check: a left pixel's disparity is only trusted if walking to
    // its claimed match in the right image and reading D_right there gives
    // (about - 1px tolerance) the same disparity back.
    // Disagreement -> occlusion or a bad match -> invalidate
    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float d = dispLeft.at<float>(r, c);
            int qx = cvRound(c - d); // corresponding x in the right image

            bool consistent = (qx >= 0 && qx < cols) && (std::abs(d - dispRight.at<float>(r, qx)) <= 1.0f);
            disparity.at<float>(r, c) = consistent ? d : static_cast<float>(minDisp - 1);
        }
    }

    return disparity;
}
