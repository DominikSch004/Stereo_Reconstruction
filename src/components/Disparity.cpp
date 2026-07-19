#include "Disparity.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cmath>

// Scales a pixel-unit threshold linearly with image scale
int scaleLinear(double base, double scale, int minVal)
{
    return std::max(minVal, static_cast<int>(std::round(base * scale)));
}

// Scales a pixel-area threshold quadratically with image scale
int scaleArea(double base, double scale, int minVal)
{
    return std::max(minVal, static_cast<int>(std::round(base * scale * scale)));
}

// For each pixel, the nearest valid (>= minDisp) disparity found by walking strictly in
// direction (dx,dy) from that pixel (NaN if the walk reaches the image border without
// finding one). Implemented as a single pass per direction: pixels are visited
// in the order opposite to (dx,dy) so that, by the time a pixel is reached, the answer for
// its (dx,dy) neighbor is already known and can just be carried forward.
cv::Mat Disparity::nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy)
{
    const float NA = std::numeric_limits<float>::quiet_NaN();
    cv::Mat result(rows, cols, CV_32F, cv::Scalar(NA));
    auto isValid = [minDisp](float v)
    { return v >= static_cast<float>(minDisp); };

    if (dy == 0)
    {
        // horizontal: state resets every row
        for (int y = 0; y < rows; ++y)
        {
            float nextValid = NA;
            if (dx > 0)
            {
                for (int x = cols - 1; x >= 0; --x)
                {
                    result.at<float>(y, x) = nextValid;
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid = v;
                }
            }
            else
            {
                for (int x = 0; x < cols; ++x)
                {
                    result.at<float>(y, x) = nextValid;
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid = v;
                }
            }
        }
    }
    else if (dx == 0)
    {
        // vertical: state persists per-column across rows
        std::vector<float> nextValid(cols, NA);
        if (dy > 0)
        {
            for (int y = rows - 1; y >= 0; --y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    result.at<float>(y, x) = nextValid[x];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[x] = v;
                }
            }
        }
        else
        {
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    result.at<float>(y, x) = nextValid[x];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[x] = v;
                }
            }
        }
    }
    else if (dx == dy)
    {
        // main diagonal (1,1) or (-1,-1): state persists per diagonal k = x - y
        std::vector<float> nextValid(rows + cols - 1, NA);
        auto k = [rows](int x, int y)
        { return x - y + (rows - 1); };
        if (dx > 0)
        {
            for (int y = rows - 1; y >= 0; --y)
            {
                for (int x = cols - 1; x >= 0; --x)
                {
                    int i = k(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[i] = v;
                }
            }
        }
        else
        {
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    int i = k(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[i] = v;
                }
            }
        }
    }
    else
    {
        // anti-diagonal (1,-1) or (-1,1): state persists per anti-diagonal s = x + y
        std::vector<float> nextValid(rows + cols - 1, NA);
        auto s = [](int x, int y)
        { return x + y; };
        if (dx > 0) // (1,-1): predecessor (x+1,y-1) is on an earlier row -> scan y ascending
        {
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    int i = s(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[i] = v;
                }
            }
        }
        else // (-1,1): predecessor (x-1,y+1) is on a later row -> scan y descending
        {
            for (int y = rows - 1; y >= 0; --y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    int i = s(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v))
                        nextValid[i] = v;
                }
            }
        }
    }

    return result;
}

cv::Mat Disparity::computeDisparity(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, DisparityMethod method, double scale, const DisparityRefinementConfig &refinement)
{
    switch (method)
    {
    case DisparityMethod::OpenCVSGBM:
        return computeSGBMOpenCV(left, right, minDisp, numDisp, blockSize, scale);
    case DisparityMethod::Custom:
        return computeCustom(left, right, minDisp, numDisp, blockSize, scale, refinement);
    default:
        std::cout << "Failed! Select a valid disparity method";
        return cv::Mat();
    }
}

// Not itself from Hirschmuller 2008: a project-specific soft confidence for
// confidence-weighted ICP. The hard cutoff it starts from is still the paper's
// L/R consistency check (Sec 2.3, Eq. 15, "the disparity is set to invalid...
// if [it and its right-image counterpart] differ" by more than lrMaxDiff), but
// here the pass/fail is turned into a continuous [0,1] weight: the product of
// two exponential falloffs, one over the L/R disagreement and one over the
// left/right photometric error.
cv::Mat Disparity::filterAndComputeConfidence(
    const cv::Mat &left, const cv::Mat &right, cv::Mat &disparityLeft,
    int minDisp, int numDisp, int blockSize, DisparityMethod method,
    float lrMaxDiff, float photometricScale)
{
    CV_Assert(disparityLeft.type() == CV_32F && disparityLeft.size() == left.size());
    CV_Assert(left.type() == CV_8U && right.type() == CV_8U && left.size() == right.size());

    lrMaxDiff = std::max(lrMaxDiff, 1e-3f);
    photometricScale = std::max(photometricScale, 1e-3f);

    cv::Mat disparityRight;
    bool rightUsesPositiveConvention = false;
    if (method == DisparityMethod::OpenCVSGBM)
    {
        // Swapping the images reverses OpenCV's disparity sign.  Search the
        // mirrored interval so d_left + d_right is the cycle-consistency error.
        const int minRight = -(minDisp + numDisp);
        disparityRight = computeSGBMOpenCV(right, left, minRight, numDisp, blockSize, 1.0);
    }
    else
    {
        cv::Mat leftFloat, rightFloat;
        left.convertTo(leftFloat, CV_32F);
        right.convertTo(rightFloat, CV_32F);
        disparityRight = computeWTADisparity(
            leftFloat, rightFloat, left.rows, left.cols,
            minDisp, numDisp, 8, 32, true);
        rightUsesPositiveConvention = true;
    }

    cv::Mat confidence(left.size(), CV_32F, cv::Scalar(0.0f));
    const float invalidDisparity = static_cast<float>(minDisp - 1);
    size_t validBefore = 0;
    size_t validAfter = 0;

    for (int y = 0; y < left.rows; ++y)
    {
        for (int x = 0; x < left.cols; ++x)
        {
            float &d = disparityLeft.at<float>(y, x);
            if (!std::isfinite(d) || d <= static_cast<float>(minDisp))
                continue;
            ++validBefore;

            const int rightX = cvRound(static_cast<float>(x) - d);
            if (rightX < 0 || rightX >= right.cols)
            {
                d = invalidDisparity;
                continue;
            }

            const float dRight = disparityRight.at<float>(y, rightX);
            const float lrError = rightUsesPositiveConvention
                ? std::abs(d - dRight)
                : std::abs(d + dRight);
            if (!std::isfinite(dRight) || lrError > lrMaxDiff)
            {
                d = invalidDisparity;
                continue;
            }

            const float photoError = std::abs(
                static_cast<float>(left.at<uchar>(y, x)) -
                static_cast<float>(right.at<uchar>(y, rightX)));
            const float lrConfidence = std::exp(
                -0.5f * lrError * lrError / (lrMaxDiff * lrMaxDiff));
            const float photoConfidence = std::exp(-photoError / photometricScale);
            confidence.at<float>(y, x) = lrConfidence * photoConfidence;
            ++validAfter;
        }
    }

    std::cout << "[Disparity] left/right validation kept " << validAfter << "/"
              << validBefore << " valid disparities ("
              << (validBefore ? 100.0 * static_cast<double>(validAfter) / validBefore : 0.0)
              << "%).\n";
    return confidence;
}

void Disparity::computeDynamicSearchRangeCalibrated(
    const std::vector<cv::Point2f> &inPtsL, const std::vector<cv::Point2f> &inPtsR,
    const cv::Mat &K, const cv::Mat &R1, const cv::Mat &P1,
    const cv::Mat &R2, const cv::Mat &P2, const cv::Size &imgSize,
    int &minDisp, int &numDisp)
{
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<float> disps;
    std::vector<cv::Point2f> rL, rR;

    // Warp points to rectified space using the raw matrices
    cv::undistortPoints(inPtsL, rL, K, dist, R1, P1);
    cv::undistortPoints(inPtsR, rR, K, dist, R2, P2);

    for (size_t i = 0; i < rL.size(); ++i)
    {
        disps.push_back(rL[i].x - rR[i].x);
    }

    if (disps.empty())
    {
        minDisp = 0;
        numDisp = 16;
        return;
    }

    std::sort(disps.begin(), disps.end());
    float dLo = disps[(size_t)(0.02 * disps.size())];
    float dHi = disps[(size_t)(0.98 * (disps.size() - 1))];

    const int margin = 32;
    int dMin = (int)std::floor((dLo - margin) / 16.0) * 16;
    int dMax = (int)std::ceil((dHi + margin) / 16.0) * 16;

    minDisp = std::max(dMin, 0);
    numDisp = std::max(16, ((dMax - minDisp + 15) / 16) * 16);
    numDisp = std::min(numDisp, imgSize.width - minDisp - 1);
    numDisp = (numDisp / 16) * 16;
}

cv::Mat Disparity::computeSGBMOpenCV(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale)
{
    int numChannels = left.channels(); // images are grayscale (1) & (3) BGR
    
    // scale wtih given scale to make everything comparable as much as possible
    // between scales/resolutions.
    const int disp12MaxDiff = scaleLinear(1, scale, 1);     // simple unit tolerence -> scale linearly
    const int speckleWindowSize = scaleArea(100, scale, 1); // window of base area   -> scale quadritically
    const int speckleRange = scaleLinear(32, scale, 1);     // simple unit tolerence -> scale linearly

    auto sgbm = cv::StereoSGBM::create(
        minDisp,
        numDisp,
        blockSize,
        8 * numChannels * blockSize * blockSize,  // P1 smoothness penalty
        32 * numChannels * blockSize * blockSize, // P2 smoothness penalty
        disp12MaxDiff,
        0,                                        // preFilterCap    (intensity  - scale invariant)
        10,                                       // uniquenessRatio (percentage - scale invariant)
        speckleWindowSize,
        speckleRange,
        cv::StereoSGBM::MODE_SGBM);

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    // Convert fixed-point 16-bit integer output representations back to real floating point coordinates
    cv::Mat dispFloat;
    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);
    return dispFloat;
}

// Birchfield & Tomasi 1998, cited in Hirschmuller 2008 Sec 2.1.2: the per-pixel
// intensity interval [Imin,Imax]
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

std::vector<uint16_t> Disparity::computePixelwiseCost(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, bool rightBase)
{
    // Birchfield & Tomasi (BT) 98 - Pixel Dissimilarity d(xi, yi) Section 2.1.2
    // Hirschmuller 2008 Sec 2.1: pixelwise matching cost C(p,d).

    // L-R consistency check
    // The returned cost C(p,d) is always indexed over "base"'s pixel positions; "match" is
    // the other image being searched. d = xL - xR always, so the search direction flips
    // between the two passes

    // rightBase param selects which image the returned cost is indexed over
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
    std::vector<uint16_t> pixelwiseCost(static_cast<size_t>(numDisp) * base.rows * base.cols, 2047); // max 11-bit cost for out-of-bounds

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
                    pixelwiseCost[idx] = static_cast<uint16_t>(std::min(cost_value, 2047.0f));
                }
            }
        }
    }
    return pixelwiseCost;
}

// Hirschmuller 2008 Sec 2.2, "Cost Aggregation," Eq. 13: accumulates one 1D
// path's cost L_r(p,d) in direction r=(dx,dy) into the aggregated cost
// S(p,d) = sum_r L_r(p,d), implementing "the new idea of aggregating matching
// costs in 1D from all directions equally" that approximates minimizing the
// 2D global energy of Eq. 11 (NP-complete for many discontinuity-preserving
// energies) without solving it directly.
void Disparity::aggregatePathCost(const std::vector<uint16_t> &C, std::vector<uint16_t> &S,
                                   int rows, int cols, int numDisp, int dx, int dy, int P1, int P2)
{
    std::vector<int> Lr(static_cast<size_t>(rows) * cols * numDisp);

    int yStart = (dy >= 0) ? 0 : rows - 1;
    int yEnd = (dy >= 0) ? rows : -1;
    int yStep = (dy >= 0) ? 1 : -1;

    int xStart = (dx >= 0) ? 0 : cols - 1;
    int xEnd = (dx >= 0) ? cols : -1;
    int xStep = (dx >= 0) ? 1 : -1;

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
                int predIdx = (py * cols + px) * numDisp;                              // predecessor pixel index
                int minPrev = *std::min_element(&Lr[predIdx], &Lr[predIdx + numDisp]); // min_k Lr(p-r, k)

                for (int d = 0; d < numDisp; ++d)
                {
                    int best = Lr[predIdx + d]; // same disparity
                    if (d > 0)
                        best = std::min(best, Lr[predIdx + d - 1] + P1); // Lr(p-r, d-1) + P1 - disparity change of 1
                    if (d < numDisp - 1)
                        best = std::min(best, Lr[predIdx + d + 1] + P1); // Lr(p-r, d+1) + P1 - disparity change of 1
                    best = std::min(best, minPrev + P2);                 // min_k Lr(p-r, k) + P2 - any larger change

                    Lr[idx + d] = C[idx + d] + best - minPrev; // ... - min_k Lr(p-r, k)
                }
            }

            for (int d = 0; d < numDisp; ++d)
                S[idx + d] += Lr[idx + d];
        }
    }
}

// Subpixel estimation (Hirschmuller 2008 Sec 2.3: fit a parabola through the
// aggregated cost at the winning disparity index and its two neighbors
// (bestDispIdx-1, bestDispIdx+1), and return the position of its minimum as a
// sub-integer correction. Returns 0 at the search range's edges, where one of
// the neighbors doesn't exist.
float Disparity::estimateSubpixel(const uint16_t *costsAtPixel, int numDisp, int bestDispIdx)
{
    if (bestDispIdx <= 0 || bestDispIdx >= numDisp - 1)
        return 0.0f;

    const float cMinus = static_cast<float>(costsAtPixel[bestDispIdx - 1]);
    const float cZero = static_cast<float>(costsAtPixel[bestDispIdx]);
    const float cPlus = static_cast<float>(costsAtPixel[bestDispIdx + 1]);
    
    // y(x) = ax^2 + bx + c = 0
    // cMinus = y(-1) = a - b + c; cZero = y(0) = c; cPlus = y(1) = a + b + c
    // so c = cZero; b = (cPlus - cMinus)/2; a = (cMinus + cPlus - 2*cZero)/2
    // x = -b/(2a) = (cMinus - cPlus) / (2 * (cMinus + cPlus - 2*cZero))
    const float denom = cMinus + cPlus - 2.0f * cZero; // 2a; >= 0 since cZero is the min of the three costs
    if (denom <= 0.0f)
        return 0.0f;
    return 0.5f * (cMinus - cPlus) / denom; // vertex of the parabola
}

// Hirschmuller 2008 Sec 2.3, "Disparity Computation": pixelwise cost -> 16-path
// aggregation -> "selecting for each pixel p the disparity d that corresponds
// to the minimum cost, that is, min_d S[p,d]". What the paper's
// introduction calls, for local methods generally, "winner takes all."
cv::Mat Disparity::computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2, bool rightBase)
{
    // All 16 directions for aggregation
    static const int dirs[16][2] = {
                                    {1, 0},
                                    {-1, 0},
                                    {0, 1},
                                    {0, -1},
                                    {1, 1},
                                    {-1, -1},
                                    {1, -1},
                                    {-1, 1},
                                    {2, 1},
                                    {-2, -1},
                                    {2, -1},
                                    {-2, 1},
                                    {1, 2},
                                    {-1, -2},
                                    {1, -2},
                                    {-1, 2}};

    std::vector<uint16_t> pixelwiseCost = computePixelwiseCost(left, right, minDisp, numDisp, rightBase);

    std::vector<uint16_t> S(static_cast<size_t>(rows) * cols * numDisp, 0);
    for (auto &d : dirs)
        aggregatePathCost(pixelwiseCost, S, rows, cols, numDisp, d[0], d[1], P1, P2);

    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int idx = (r * cols + c) * numDisp;
            auto minCost = std::min_element(&S[idx], &S[idx + numDisp]);         // minimum aggregated cost
            int bestDispIdx = static_cast<int>(std::distance(&S[idx], minCost)); // index of best disparity

            const float offset = estimateSubpixel(&S[idx], numDisp, bestDispIdx);
            disparity.at<float>(r, c) = static_cast<float>(minDisp + bestDispIdx) + offset;
        }
    }

    return disparity;
}

// Gap interpolation (Hirschmuller 2008 Sec 2.5.3, "Discontinuity Preserving Interpolation")
// fills whatever pixels the earlier stages left invalid. Not every gap is the same kind of hole,
// so they aren't filled the same way:
//   - Occlusion: no match exists anywhere, because the pixel is visible from
//     this camera but hidden from the other one. Since disparity falls with
//     depth, the occluder is the largest disparity neighbor, filling from it
//     would smear that object's edge outward, so these take the second-lowest
//     disparity among the 8 direction-neighbors instead: the background surface
//     behind the occluder.
//   - Mismatch: a real match does exist, it just wasn't found (or got rejected
//     by the L/R check). With no foreground/background bias to correct for,
//     these take the median of all surrounding valid disparities instead.
// Which case applies is decided per pixel below, by checking whether the
// right-image disparity map has any candidate match along the epipolar line.
cv::Mat Disparity::interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, const cv::Mat &baseF, int minDisp, int numDisp)
{
    static const int dirs[8][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

    const int rows = disparity.rows;
    const int cols = disparity.cols;

    // 1: nearest valid disparity along each of the 8 directions, one full-image pass each.
    cv::Mat neighbors[8];
    for (int i = 0; i < 8; ++i)
        neighbors[i] = nearestValidInDirection(disparity, minDisp, rows, cols, dirs[i][0], dirs[i][1]);

    cv::Mat filled = disparity.clone();
    cv::Mat wasInvalid = cv::Mat::zeros(rows, cols, CV_8U);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float d = disparity.at<float>(y, x);
            if (d >= static_cast<float>(minDisp))
                continue; // already valid, nothing to fill

            // Black rectification border
            if (baseF.at<float>(y, x) <= 0.0f)
                continue;

            wasInvalid.at<uchar>(y, x) = 255;

            // 2: classify as occlusion or mismatch by checking whether p's row in the
            // right-base disparity map contains any pixel consistent with a hypothesized d.
            // If D_right(x - d, y) == d for some d in range, a match exists elsewhere along
            // the epipolar line -> mismatch. Otherwise no candidate match exists -> occlusion.
            bool isMismatch = false;
            for (int dd = 0; dd < numDisp; ++dd)
            {
                int actualDisp = dd + minDisp;
                int qx = x - actualDisp;
                if (qx < 0 || qx >= cols)
                    continue;
                if (std::abs(dispRight.at<float>(y, qx) - static_cast<float>(actualDisp)) <= 1.0f)
                {
                    isMismatch = true;
                    break;
                }
            }

            // Step 2 (cont.): collect the values found in each of the 8 directions.
            std::vector<float> v;
            v.reserve(8);
            for (int i = 0; i < 8; ++i)
            {
                float nv = neighbors[i].at<float>(y, x);
                if (std::isfinite(nv))
                    v.push_back(nv);
            }
            if (v.empty())
                continue; // no valid disparity in any direction - leave invalid

            std::sort(v.begin(), v.end());

            // Step 3: occlusions extrapolate from the background (second-lowest disparity,
            // i.e. the more distant of the two nearest surfaces) to avoid pulling in the
            // occluder; mismatches take the median of all directions (unbiased fill).
            float fillValue;
            if (isMismatch)
            {
                size_t n = v.size();
                fillValue = (n % 2 == 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
            }
            else
            {
                fillValue = (v.size() >= 2) ? v[1] : v[0];
            }

            filled.at<float>(y, x) = fillValue;
        }
    }

    // Step 4: optional 3x3 median filter, applied only to the pixels that were just filled
    // in, to smooth out remaining outliers without disturbing already-consistent matches.
    cv::Mat result = filled.clone();
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (!wasInvalid.at<uchar>(y, x))
                continue;

            std::vector<float> window;
            window.reserve(9);
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox)
                {
                    int ny = y + oy, nx = x + ox;
                    if (ny < 0 || ny >= rows || nx < 0 || nx >= cols)
                        continue;
                    float v = filled.at<float>(ny, nx);
                    if (v >= static_cast<float>(minDisp))
                        window.push_back(v);
                }
            if (window.empty())
                continue;

            std::sort(window.begin(), window.end());
            result.at<float>(y, x) = window[window.size() / 2];
        }
    }

    return result;
}

// Peak filtering (Hirschmuller 2008, Sec 2.5.1): removes small isolated segment of
// wrong disparity ("peaks") that slipped past the L-R check: typically noise or
// a bad match in a low-texture region
//
// The idea: a real surface (a wall, an object) is a large patch of pixels whose
// disparity varies smoothly, with no sharp jumps. A peak is the opposite: one
// or a few pixels whose disparity disagrees sharply with everything around them.
// So: starting from each valid pixel, grow outward to its 4 (up/down/left/right)
// neighbors, but only step into a neighbor if its disparity is within
// maxSegmentDispDiff of the current pixel's. Each resulting connected segment is
// either a big chunk of real structure, or a tiny isolated peak. Segments
// smaller than minSegmentSize pixels are assumed to be peaks and invalidated
// (set to minDisp - 1).
cv::Mat Disparity::removePeaks(const cv::Mat &disparity, int minDisp, int minSegmentSize, float maxSegmentDispDiff)
{
    const int rows = disparity.rows;
    const int cols = disparity.cols;
    const float minDispF = static_cast<float>(minDisp);

    cv::Mat result = disparity.clone();
    cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8U);

    // stack: pixels that still need to be explored (DFS)
    // segment: all pixels belonging to the current connected component
    std::vector<cv::Point> stack, segment;

    // right, left, down, up
    static const int dxs[4] = {1, -1, 0, 0};
    static const int dys[4] = {0, 0, 1, -1};

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (visited.at<uchar>(y, x))
                continue;

            float d = disparity.at<float>(y, x);
            if (!(d > minDispF))
            {
                // Invalid pixels do not belong to any segment.
                visited.at<uchar>(y, x) = 255;
                continue;
            }

            // Start a new connected segment using 4-connected DFS
            // with chained 1 px disparity agreement.
            segment.clear();
            stack.clear();
            stack.push_back({x, y});
            // Mark as visited before pushing to avoid duplicates
            visited.at<uchar>(y, x) = 255;

            // DFS
            while (!stack.empty())
            {
                cv::Point p = stack.back();
                stack.pop_back();
                segment.push_back(p);
                float dp = disparity.at<float>(p.y, p.x);

                // Check the 4 adjacent neighbours
                for (int k = 0; k < 4; ++k)
                {
                    int nx = p.x + dxs[k], ny = p.y + dys[k];

                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows)
                        continue;
                    if (visited.at<uchar>(ny, nx))
                        continue;

                    // neighbour disparity
                    float dn = disparity.at<float>(ny, nx);
                    
                    // Grow into this neighbor only if it's valid and smoothly
                    // continues the current disparity (differs by at most
                    // maxSegmentDispDiff px from current pixel).
                    if (!(dn > minDispF) || std::abs(dn - dp) > maxSegmentDispDiff)
                        continue;

                    visited.at<uchar>(ny, nx) = 255;
                    stack.push_back({nx, ny});
                }
            }

            // remove tiny regions which are too smal to
            // be real structure -> peak
            if (static_cast<int>(segment.size()) < minSegmentSize)
                for (const auto &p : segment)
                    result.at<float>(p.y, p.x) = minDispF - 1.0f;
        }
    }

    return result;
}

// Hirschmuller 2008 end-to-end: computePixelwiseCost (Sec 2.1) -> aggregatePathCost
// (Sec 2.2) -> computeWTADisparity with estimateSubpixel (Sec 2.3) -> the L/R
// consistency check. Sec 2.5, "Disparity Refinement": removePeaks (Sec 2.5.1) and
// interpolateGaps (Sec 2.5.3), are postprocessing techniques, each individually
// optional via DisparityRefinementConfig. 
cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale, const DisparityRefinementConfig &refinement)
{
    // blockSize is not used as BT cost volume is computer per-pixel and not within a window
    const int P1 = 8;  // smoothness penalty for disparity change of 1
    const int P2 = 32; // smoothness penalty for disparity change greater than
    const int rows = left.rows;
    const int cols = left.cols;

    // Mirrors computeSGBMOpenCV's disp12MaxDiff/speckleWindowSize scaling. The "1"
    // base values are fixed by the paper's own definitions (Eq. 15's tolerance, and
    // Sec 2.5.1's "vary by one pixel" peak-segment criterion).
    const float LRConsistencyTol = static_cast<float>(scaleLinear(1, scale, 1));   // cv::StereoSGBM's disp12MaxDiff
    const int minPeakSegment = scaleArea(refinement.minPeakSegmentPx, scale, 1);   // cv::StereoSGBM's speckleWindowSize
    const float maxSegmentDispDiff = static_cast<float>(scaleLinear(1, scale, 1)); // cv::StereoSGBM's speckleRange

    cv::Mat leftF, rightF;
    left.convertTo(leftF, CV_32F);
    right.convertTo(rightF, CV_32F);

    // Run the full winner takes all (WTA) disparity calculation twice: once treating
    // the left image as the base (D_left), once the right image (D_right)
    cv::Mat dispLeft = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, false);
    cv::Mat dispRight = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, true);

    // L-R consistency check (Hirschmuller 2008, Sec 2.3, Eq. 15):
    // a left pixel's disparity is only trusted if walking to its claimed
    // match in the right image and reading D_right there gives the same disparity back,
    // within LRConsistencyTol (1px).
    // Disagreement -> occlusion or a bad match -> invalidate
    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float d = dispLeft.at<float>(r, c);
            int qx = cvRound(c - d); // corresponding x in the right image

            bool consistent = (qx >= 0 && qx < cols) && (std::abs(d - dispRight.at<float>(r, qx)) <= LRConsistencyTol);
            disparity.at<float>(r, c) = consistent ? d : static_cast<float>(minDisp - 1);
        }
    }

    // Peak filtering (Hirschmuller 2008, Sec 2.5.1): removes small isolated
    // patches of disparity that survived the L-R check but disagree with their
    // surroundings. The paper doesn't give a universal size threshold to use.
    // Using 100px here as it matches cv::StereoSGBM's speckleWindowSize value
    // we are currently using.

    // NOTE: coverage vs. dense-vs-sparse accuracy is a continuous tradeoff here
    // As we increase minPeakSegment, it decreases coverage and increases
    // dense-vs-sparse accuracy.
    if (refinement.peakFiltering)
        disparity = removePeaks(disparity, minDisp, minPeakSegment, maxSegmentDispDiff);

    // Gap interpolation (Hirschmuller 2008, Sec 2.5.3): pushes coverage to 100% by
    // interpolating remaining gaps. Trades some dense-vs-sparse accuracy for coverage;
    // helpful on scenes like scan6, see config.yaml.
    return refinement.gapFill ? interpolateGaps(disparity, dispRight, leftF, minDisp, numDisp) : disparity;
}
