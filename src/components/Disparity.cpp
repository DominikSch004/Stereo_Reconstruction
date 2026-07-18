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

cv::Mat Disparity::computeDisparity(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, DisparityMethod method, double scale,
                                    bool useIntensityConsistentSelection, bool useGapFill)
{
    switch (method)
    {
    case DisparityMethod::OpenCVSGBM:
        return computeSGBMOpenCV(left, right, minDisp, numDisp, blockSize, scale);
    case DisparityMethod::Custom:
        return computeCustom(left, right, minDisp, numDisp, blockSize, scale, useIntensityConsistentSelection, useGapFill);
    default:
        std::cout << "Failed! Select a valid disparity method";
        return cv::Mat();
    }
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
        10,                                       // uniquenessRatio (percentage - invariant)
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

// Birchfield & Tomasi (BT) 98
float Disparity::btCost(float baseVal, float minBase, float maxBase, float matchVal, float minMatch, float maxMatch)
{
    float cost_base_match = std::max({0.0f, baseVal - maxMatch, minMatch - baseVal});
    float cost_match_base = std::max({0.0f, matchVal - maxBase, minBase - matchVal});
    return std::min(cost_base_match, cost_match_base);
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

                    float cost_value = btCost(base_val, Imin_base_val, Imax_base_val, match_val, Imin_match_val, Imax_match_val);

                    int idx = (r * base.cols + c) * numDisp + d;
                    costVolume[idx] = static_cast<uint16_t>(std::min(cost_value, 2047.0f));
                }
            }
        }
    }
    return costVolume;
}

void Disparity::aggregateDirection(const std::vector<uint16_t> &C, std::vector<uint16_t> &S, const cv::Mat &baseF,
                                   int rows, int cols, int numDisp, int dx, int dy, int P1, int P2Base)
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
                //
                // P2 adapts to the base-image intensity gradient along this path step
                // (Hirschmuller 2008, Eq. after (13)): P2 = P2' / |I_p - I_{p-r}|, clamped
                // to >= P1. Real depth discontinuities usually coincide with intensity
                // edges, so this lets the disparity jump freely there while keeping the
                // full penalty in flat/textured regions.
                float intensityDiff = std::abs(baseF.at<float>(y, x) - baseF.at<float>(py, px));
                int P2 = std::max(P1, static_cast<int>(std::round(P2Base / std::max(1.0f, intensityDiff))));

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

// Full cost volume -> 16-direction aggregation -> WTA pipeline for one base image.
cv::Mat Disparity::computeWTADisparity(const cv::Mat &left, const cv::Mat &right, int rows, int cols, int minDisp, int numDisp, int P1, int P2Base, bool rightBase)
{
    static const int dirs[16][2] = {// All 16 directions for aggregation
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

    std::vector<uint16_t> costVolume = computeCostVolume(left, right, minDisp, numDisp, rightBase);

    // P2 adapts to gradients in whichever image is the base for this pass (Hirschmuller
    // 2008 defines it over I_b specifically, not a fixed image).
    const cv::Mat &baseF = rightBase ? right : left;

    std::vector<uint16_t> S(static_cast<size_t>(rows) * cols * numDisp, 0);
    for (auto &d : dirs)
        aggregateDirection(costVolume, S, baseF, rows, cols, numDisp, d[0], d[1], P1, P2Base);

    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            int idx = (r * cols + c) * numDisp;
            auto minCost = std::min_element(&S[idx], &S[idx + numDisp]);         // minimum aggregated cost
            int bestDispIdx = static_cast<int>(std::distance(&S[idx], minCost)); // index of best disparity

            // Subpixel refinement:
            // fit a parabola through  the neighboring costs, that is, at the next
            // higher and lower disparity (bestDispIdx-1, bestDispIdx+1),
            // and the position of the minimum is calculated (take its vertex as a sub-integer correction)
            // Skipped at the search range's edges
            float subpixelOffset = 0.0f;
            if (bestDispIdx > 0 && bestDispIdx < numDisp - 1)
            {
                float cMinus = static_cast<float>(S[idx + bestDispIdx - 1]);
                float cZero = static_cast<float>(S[idx + bestDispIdx]);
                float cPlus = static_cast<float>(S[idx + bestDispIdx + 1]);
                // y(x) = ax^2 + bx + c = 0
                // cMinus = y(-1) = a - b + c; cZero = y(0) = c cPlus = y(1) = a + b + c
                // so c = cZero, b = (cPlus - cMinus)/2, a = (cMinus + cPlus - 2*cZero)/2
                // x = -b/(2a) = (cMinus - cPlus) / (2 * (cMinus + cPlus - 2*cZero))
                float denom = cMinus + cPlus - 2.0f * cZero; // 2a; >= 0 since cZero is the min of the three costs
                if (denom > 0.0f)
                    subpixelOffset = 0.5f * (cMinus - cPlus) / denom; // vertex of the parabola
            }

            disparity.at<float>(r, c) = static_cast<float>(minDisp + bestDispIdx) + subpixelOffset;
        }
    }

    return disparity;
}

cv::Mat Disparity::interpolateGaps(const cv::Mat &disparity, const cv::Mat &dispRight, int minDisp, int numDisp)
{
    // Hirschmuller 2008, Sec 2.5.3
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

// Peak filtering (Hirschmuller 2008, Sec 2.5.1): segments the valid disparity
// map into 4-connected regions where adjacent pixels' disparities agree within
// 1px, then invalidates (minDisp - 1) every segment smaller than
// minSegmentSize. This removes small isolated patches of incorrect disparity
// ("peaks", e.g. from noise/low texture) while preserving real scene
// strucutre, which forms much larger connected segments.
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

                    // accept the neighbour only if it has a valid disparity
                    // and differs by at most maxSegmentDispDiff px from the current pixel.
                    if (!(dn > minDispF) || std::abs(dn - dp) > maxSegmentDispDiff)
                        continue;

                    visited.at<uchar>(ny, nx) = 255;
                    stack.push_back({nx, ny});
                }
            }

            // remove tiny regions which are unlikely to
            // represent valid scene structure.
            if (static_cast<int>(segment.size()) < minSegmentSize)
                for (const auto &p : segment)
                    result.at<float>(p.y, p.x) = minDispF - 1.0f;
        }
    }

    return result;
}

// --- Hirschmuller 2008 Sec 2.5.2: Intensity Consistent Disparity Selection ---
//
// Why: adaptive P2 (on aggregateDirection, above) already places disparity discontinuities
// correctly at intensity edges, but SGM only aggregates along 1D paths, so large
// untextured interiors (walls, floors, tables) can still come out fuzzy/noisy. This
// section recovers a clean disparity there by assuming each such interior is a single
// *plane*, then picking whichever candidate plane the actual stereo cost supports best.
//
// Three steps, one function each, called in this order from selectIntensityConsistentDisparity:
//   1. meanShiftModes                     -- groups the base image into same-intensity regions.
//   2. segmentByIntensity                 -- turns those regions into labeled connected segments.
//   3. selectIntensityConsistentDisparity -- per segment, fits candidate planes from the
//      segment's own (possibly noisy) disparity, scores each against the real stereo
//      cost, and overwrites the whole segment with the winner.
//
// Step 1: per-pixel mode-seeking (fixed-bandwidth Mean Shift) over the base image's
// (x, y, intensity) domain. Starting at each pixel, repeatedly average the (x, y, I) of
// every pixel within a sigmaS x sigmaS window whose intensity is within sigmaR of the
// current estimate, then move to that average. This "climbs" toward the nearest mode
// (intensity plateau). sigmaS=5 (paper: "a rather low value for fast processing", an
// 11x11 window), sigmaR=P1 (paper's exact value). Returns each pixel's converged mode
// intensity; pixels on the same plateau converge to nearly the same value.
cv::Mat Disparity::meanShiftModes(const cv::Mat &baseF, int sigmaS, float sigmaR, int maxIters, float convergeEps)
{
    const int rows = baseF.rows;
    const int cols = baseF.cols;
    cv::Mat modes(rows, cols, CV_32F);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float zx = static_cast<float>(x);
            float zy = static_cast<float>(y);
            float zI = baseF.at<float>(y, x);

            for (int iter = 0; iter < maxIters; ++iter)
            {
                int cx = cvRound(zx), cy = cvRound(zy);
                int xlo = std::max(0, cx - sigmaS), xhi = std::min(cols - 1, cx + sigmaS);
                int ylo = std::max(0, cy - sigmaS), yhi = std::min(rows - 1, cy + sigmaS);

                double sumX = 0.0, sumY = 0.0, sumI = 0.0;
                int count = 0;
                for (int yy = ylo; yy <= yhi; ++yy)
                {
                    for (int xx = xlo; xx <= xhi; ++xx)
                    {
                        float v = baseF.at<float>(yy, xx);
                        if (std::abs(v - zI) > sigmaR)
                            continue;
                        sumX += xx;
                        sumY += yy;
                        sumI += v;
                        ++count;
                    }
                }
                if (count == 0)
                    break; // (x,y) itself always satisfies the range test, so this can't happen

                float nzx = static_cast<float>(sumX / count);
                float nzy = static_cast<float>(sumY / count);
                float nzI = static_cast<float>(sumI / count);

                float shift = std::abs(nzx - zx) + std::abs(nzy - zy) + std::abs(nzI - zI);
                zx = nzx;
                zy = nzy;
                zI = nzI;
                if (shift < convergeEps)
                    break;
            }

            modes.at<float>(y, x) = zI;
        }
    }

    return modes;
}

// Step 2: turns the per-pixel modes from meanShiftModes into labeled segments, using the
// same 4-connected flood-fill DFS as removePeaks. Here the merge test is "modes within
// mergeTolerance. mergeTolerance is deliberately tight: pixels on the same mode plateau
// already converged to nearly identical values in step 1, so this only needs to absorb
// float/iteration-cap noise, not do the actual grouping. Segments smaller than
// minSegmentSize are left unlabeled (-1): the paper only wants this to fix large uniform
// interiors, not small textured detail which SGM already handles fine.
//
// The rectified image's black border (from rectification's rotation/skew) is not real
// content and must never join a segment: if it did, the whole segment (border included)
// would later inherit that segment's winning plane hypothesis, producing "valid" disparity
// outside the actual photo leading to a bug: coverage() > 100%.
cv::Mat Disparity::segmentByIntensity(const cv::Mat &modes, float mergeTolerance, int minSegmentSize)
{
    const int rows = modes.rows;
    const int cols = modes.cols;

    cv::Mat labels(rows, cols, CV_32S, cv::Scalar(-1));
    cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8U);

    std::vector<cv::Point> stack, segment;
    static const int dxs[4] = {1, -1, 0, 0};
    static const int dys[4] = {0, 0, 1, -1};

    int nextLabel = 0;
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (visited.at<uchar>(y, x))
                continue;

            // Black rectification border (see comment above).
            if (modes.at<float>(y, x) <= 0.0f)
            {
                visited.at<uchar>(y, x) = 255;
                continue;
            }

            segment.clear();
            stack.clear();
            stack.push_back({x, y});
            visited.at<uchar>(y, x) = 255;

            while (!stack.empty())
            {
                cv::Point p = stack.back();
                stack.pop_back();
                segment.push_back(p);
                float ip = modes.at<float>(p.y, p.x);

                for (int k = 0; k < 4; ++k)
                {
                    int nx = p.x + dxs[k], ny = p.y + dys[k];
                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows)
                        continue;
                    if (visited.at<uchar>(ny, nx))
                        continue;

                    float in = modes.at<float>(ny, nx);
                    if (in <= 0.0f)
                        continue; // border pixel, never join a real segment
                    if (std::abs(in - ip) > mergeTolerance)
                        continue;

                    visited.at<uchar>(ny, nx) = 255;
                    stack.push_back({nx, ny});
                }
            }

            if (static_cast<int>(segment.size()) >= minSegmentSize)
            {
                for (const auto &p : segment)
                    labels.at<int32_t>(p.y, p.x) = nextLabel;
                ++nextLabel;
            }
        }
    }

    return labels;
}

bool Disparity::fitPlane(const std::vector<cv::Point> &pixels, const cv::Mat &disparity, PlaneHypothesis &out)
{
    // Normal equations for [a b c] minimizing sum (a*x + b*y + c - d)^2.
    double Sxx = 0, Sxy = 0, Sx = 0, Syy = 0, Sy = 0, S1 = 0, Sxd = 0, Syd = 0, Sd = 0;
    for (const auto &p : pixels)
    {
        double x = p.x, y = p.y, d = disparity.at<float>(p.y, p.x);
        Sxx += x * x; Sxy += x * y; Sx += x;
        Syy += y * y; Sy += y; S1 += 1.0;
        Sxd += x * d; Syd += y * d; Sd += d;
    }
    cv::Matx33d A(Sxx, Sxy, Sx,
                  Sxy, Syy, Sy,
                  Sx, Sy, S1);
    cv::Vec3d B(Sxd, Syd, Sd);
    cv::Vec3d sol;
    if (!cv::solve(A, B, sol, cv::DECOMP_SVD))
        return false;
    out.a = sol[0]; out.b = sol[1]; out.c = sol[2];
    return true;
}

// Step 3 of Sec 2.5.2 (the segment-level decision, see the pipeline overview above
// meanShiftModes): for each intensity segment, decide what disparity it should have.
// Three parts, one function each:
//   a. findPlaneHypotheses (here) -- sub-segment the segment by disparity continuity
//      (same 4-connected, |neighbour disparity - own disparity| <= 1px rule as
//      removePeaks) and fit a plane D = a*x + b*y + c through each piece bigger than
//      12px (Hirschmuller's threshold for trusting a plane fit). These are the
//      candidate hypotheses for the whole segment. If no piece is big enough to fit
//      a plane, the segment is left untouched later.
//   b. scoreHypothesis (below) -- score every hypothesis against the WHOLE segment
//      using the SGM energy (Eq. 11), skipping occluded pixels.
//   c. selectIntensityConsistentDisparity (below) -- the lowest-cost hypothesis wins
//      and overwrites every pixel of the segment (Eq. 17b).
std::vector<Disparity::PlaneHypothesis> Disparity::findPlaneHypotheses(const std::vector<cv::Point> &Si, int segLabel, const SegmentEvalContext &ctx)
{
    std::vector<PlaneHypothesis> hypotheses;
    cv::Mat subVisited(ctx.rows, ctx.cols, CV_8U, cv::Scalar(0));
    std::vector<cv::Point> stack, subSegment;
    static const int dxs[4] = {1, -1, 0, 0};
    static const int dys[4] = {0, 0, 1, -1};

    for (const auto &seed : Si)
    {
        if (subVisited.at<uchar>(seed.y, seed.x))
            continue;

        float d0 = ctx.disparity.at<float>(seed.y, seed.x);
        if (!(d0 > ctx.minDispF))
        {
            subVisited.at<uchar>(seed.y, seed.x) = 255;
            continue;
        }

        subSegment.clear();
        stack.clear();
        stack.push_back(seed);
        subVisited.at<uchar>(seed.y, seed.x) = 255;

        while (!stack.empty())
        {
            cv::Point p = stack.back();
            stack.pop_back();
            subSegment.push_back(p);
            float dp = ctx.disparity.at<float>(p.y, p.x);

            for (int k = 0; k < 4; ++k)
            {
                int nx = p.x + dxs[k], ny = p.y + dys[k];
                if (nx < 0 || nx >= ctx.cols || ny < 0 || ny >= ctx.rows)
                    continue;
                if (subVisited.at<uchar>(ny, nx))
                    continue;
                if (ctx.labels.at<int32_t>(ny, nx) != segLabel)
                    continue; // stay within this intensity segment

                float dn = ctx.disparity.at<float>(ny, nx);
                if (!(dn > ctx.minDispF) || std::abs(dn - dp) > 1.0f)
                    continue;

                subVisited.at<uchar>(ny, nx) = 255;
                stack.push_back({nx, ny});
            }
        }

        // Ignore sub-segments <= 12px (Hirschmuller 2008 Sec 2.5.2, Assumption 3
        // discussion): too small to trust a plane fit from.
        if (subSegment.size() <= 12)
            continue;

        PlaneHypothesis hyp;
        if (fitPlane(subSegment, ctx.disparity, hyp))
            hypotheses.push_back(hyp);
    }

    return hypotheses;
}

// Part b (see findPlaneHypotheses above): called once per candidate hypothesis. For
// every pixel of the WHOLE segment (not just the sub-segment that produced this
// hypothesis), sums two things: (1) the data term: btCost checks whether that
// pixel's actual color really matches the pixel it would land on in the other image
// under this hypothesis's predicted disparity, grounding the plane in what the two
// cameras actually photographed, not just in its own math; (2) the usual P1/P2
// smoothness cost against its neighbours. Occluded pixels are skipped entirely (the
// paper's occlusion test: if some other, closer pixel already claims the same
// match-image column, this pixel is occluded): neither the data nor the smoothness
// term gets added for them. selectIntensityConsistentDisparity picks whichever
// hypothesis's total across the whole segment is lowest.
double Disparity::scoreHypothesis(const PlaneHypothesis &hyp, const std::vector<cv::Point> &Si, int segLabel, const SegmentEvalContext &ctx)
{
    // Effective disparity at (x,y) if this segment were replaced by hyp: the
    // hypothesis value inside the segment, the pixel's own (unaffected) disparity
    // outside.
    auto effectiveDisp = [&](int x, int y) -> float
    {
        if (ctx.labels.at<int32_t>(y, x) == segLabel)
            return hyp.at(x, y);
        return ctx.disparity.at<float>(y, x);
    };

    static const int dxs[4] = {1, -1, 0, 0};
    static const int dys[4] = {0, 0, 1, -1};

    double cost = 0.0;
    for (const auto &p : Si) // for each pixel in the segment
    {
        float dHyp = hyp.at(p.x, p.y); // continuous disparity the plane predicts at p
        int dRound = cvRound(dHyp);    // rounded to the nearest integer disparity level, for cost lookups below
        if (dRound < ctx.minDisp || dRound > ctx.maxDisp)
            continue; // hypothesis extrapolates outside the search range here 

        // Occlusion test: qx is the match-image column p would land on under this
        // hypothesis. Walk every disparity dPrime CLOSER than p's own (dPrime > dRound,
        // i.e. nearer to the camera) and find which base-image pixel bx would ALSO
        // land on qx if it had that disparity. If bx's own disparity
        // is at least dPrime, it really is that close: so bx sits in front of qx and
        // visually blocks it, meaning p can't actually be seen matching there.
        int qx = p.x - dRound;
        bool occluded = false;
        for (int dPrime = dRound + 1; dPrime <= ctx.maxDisp; ++dPrime)
        {
            int bx = qx + dPrime; // base-image pixel that would also project onto qx at disparity dPrime
            if (bx < 0 || bx >= ctx.cols)
                continue;
            float dThere = effectiveDisp(bx, p.y); // bx's actual disparity (this hypothesis inside Si, unchanged outside)
            if (dThere >= static_cast<float>(dPrime))
            {
                occluded = true; // bx really is that close: it hides p from view at qx
                break;
            }
        }
        if (occluded)
            continue; // can't verify a hidden match: this pixel contributes no cost either way

        // p's actual match-image column (same value as qx above); pixels whose
        // hypothesis disparity would look outside the image can't be scored at all.
        int matchCol = p.x - dRound;
        if (matchCol < 0 || matchCol >= ctx.cols)
            continue;

        // Data term: how well p's actual pixel value matches the pixel at matchCol
        // under this hypothesis's disparity. Low cost = the hypothesis's disparity is
        // photometrically plausible here; high cost = the colors don't really agree.
        cost += btCost(ctx.baseF.at<float>(p.y, p.x), ctx.Imin_base.at<float>(p.y, p.x), ctx.Imax_base.at<float>(p.y, p.x),
                       ctx.matchF.at<float>(p.y, matchCol), ctx.Imin_match.at<float>(p.y, matchCol), ctx.Imax_match.at<float>(p.y, matchCol));

        // Smoothness term: compare p's rounded hypothesis disparity against each of
        // its 4 neighbours' disparity (their own unaffected value if they're outside
        // Si, or this same hypothesis if they're inside it: effectiveDisp handles
        // both). No penalty if they already agree; P1 for a 1px difference; the
        // larger P2Base ceiling for anything bigger. Neighbours with no valid
        // disparity of their own don't contribute (nothing to compare against).
        for (int k = 0; k < 4; ++k)
        {
            int nx = p.x + dxs[k], ny = p.y + dys[k];
            if (nx < 0 || nx >= ctx.cols || ny < 0 || ny >= ctx.rows)
                continue;
            float dq = effectiveDisp(nx, ny);
            if (!(dq > ctx.minDispF && dq <= static_cast<float>(ctx.maxDisp)))
                continue; // invalid sentinel, or (inside Si) a hypothesis value extrapolated: out of range
            int diff = std::abs(dRound - cvRound(dq));
            if (diff == 1)
                cost += ctx.P1;
            else if (diff > 1)
                cost += ctx.P2Base; // flat P2' here -> the local-gradient adaptive P2 is a per-path aggregation concept
        }
    }

    return cost;
}

// Part c (see findPlaneHypotheses above): orchestrates the segment-level decision --
// builds the intensity segments, then per segment calls findPlaneHypotheses (a) and
// scoreHypothesis (b) and applies the winning (lowest-cost) hypothesis to every pixel.
cv::Mat Disparity::selectIntensityConsistentDisparity(const cv::Mat &disparity,
                                                       const cv::Mat &baseF, const cv::Mat &matchF,
                                                       const cv::Mat &Imin_base, const cv::Mat &Imax_base,
                                                       const cv::Mat &Imin_match, const cv::Mat &Imax_match,
                                                       int minDisp, int numDisp, int P1, int P2Base)
{
    const int rows = disparity.rows;
    const int cols = disparity.cols;
    const float minDispF = static_cast<float>(minDisp);
    const int maxDisp = minDisp + numDisp - 1;

    // sigmaS=5, sigmaR=P1 (Hirschmuller 2008 Sec 2.5.2 exact values -- see meanShiftModes).
    // mergeTolerance=2.0 is deliberately tight, not sigmaR again (see segmentByIntensity).
    // minSegmentSize=100, same threshold the paper uses and removePeaks/speckleWindowSize use here.
    cv::Mat modes = meanShiftModes(baseF, 5, static_cast<float>(P1)); // step 1
    cv::Mat labels = segmentByIntensity(modes, 2.0f, 100); // step 2

    // step 3
    int segCount = 0;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            segCount = std::max(segCount, labels.at<int32_t>(y, x) + 1);
    if (segCount == 0)
        return disparity.clone();

    std::vector<std::vector<cv::Point>> segments(segCount);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
        {
            int l = labels.at<int32_t>(y, x);
            if (l >= 0)
                segments[l].push_back({x, y});
        }

    const SegmentEvalContext ctx{labels, disparity, baseF, matchF,
                                  Imin_base, Imax_base, Imin_match, Imax_match,
                                  minDisp, maxDisp, minDispF, P1, P2Base, rows, cols};

    cv::Mat result = disparity.clone();

    for (int segLabel = 0; segLabel < segCount; ++segLabel)
    {
        const std::vector<cv::Point> &Si = segments[segLabel];

        std::vector<PlaneHypothesis> hypotheses = findPlaneHypotheses(Si, segLabel, ctx);
        if (hypotheses.empty())
            continue; // no evidence for this segment; leave its disparities untouched

        // Evaluate each hypothesis over the WHOLE segment (Eq. 11's data + smoothness
        // terms, restricted to unoccluded pixels), pick the minimum-cost one.
        double bestCost = std::numeric_limits<double>::infinity();
        int bestIdx = -1;
        for (size_t h = 0; h < hypotheses.size(); ++h)
        {
            double cost = scoreHypothesis(hypotheses[h], Si, segLabel, ctx);
            if (cost < bestCost)
            {
                bestCost = cost;
                bestIdx = static_cast<int>(h);
            }
        }
        if (bestIdx < 0)
            continue;

        // D'_p = Fi(p) for every pixel of Si (Eq. 17b): replaces incorrect disparities
        // *and* fills previously-invalid ones within the segment.
        const PlaneHypothesis &winner = hypotheses[bestIdx];
        for (const auto &p : Si)
            result.at<float>(p.y, p.x) = winner.at(p.x, p.y);
    }

    return result;
}

cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize, double scale,
                                 bool useIntensityConsistentSelection, bool useGapFill)
{
    // blockSize is not used as BT cost volume is computer per-pixel and not within a window
    const int P1 = 8;   // fixed smoothness penalty for a disparity change of 1
    const int P2 = 32;  // P2' ceiling for larger changes; aggregateDirection divides this
                        // by the local base-image intensity gradient (Hirschmuller 2008,
                        // Eq. after (13)). Matches cv::StereoSGBM's fixed P1:P2 ratio
                        // of 1:4 (8*bs^2 : 32*bs^2).

    const int rows = left.rows;
    const int cols = left.cols;

    // Mirrors computeSGBMOpenCV's disp12MaxDiff/speckleWindowSize scaling
    const float LRConsistencyTol = static_cast<float>(scaleLinear(1, scale, 1));   // cv::StereoSGBM's disp12MaxDiff
    const int minPeakSegment = scaleArea(100, scale, 1);                           // cv::StereoSGBM's speckleWindowSize
    const float maxSegmentDispDiff = static_cast<float>(scaleLinear(1, scale, 1)); // cv::StereoSGBM's speckleRange

    cv::Mat leftF, rightF;
    left.convertTo(leftF, CV_32F);
    right.convertTo(rightF, CV_32F);

    // Run the full winner takes all (WTA) disparity calculaation twice:
    // once treating the left image as the base (D_left), once the right image (D_right)
    cv::Mat dispLeft = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, false);
    cv::Mat dispRight = computeWTADisparity(leftF, rightF, rows, cols, minDisp, numDisp, P1, P2, true);

    // L-R consistency check (Hirschmuller 2008, Eq. 15): a left pixel's
    // disparity is only trusted if walking to its claimed match in the right
    // image and reading D_right there gives the same disparity back,
    // within 1px.
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
    disparity = removePeaks(disparity, minDisp, minPeakSegment, maxSegmentDispDiff);

    // Intensity consistent disparity selection (Hirschmuller 2008, Sec 2.5.2)
    // the adaptive P2 in aggregateDirection places discontinuities correctly
    // at intensity edges, but along untextured interiors it can still leave
    // fuzzy/noisy disparity, since SGM only sees 1D paths and not the 2D
    // segment as a whole. This recovers coverage/accuracy there by fitting
    // competing plane hypotheses per intensity segment and picking whichever
    // the (unoccluded) pixel evidence best supports.
    if (useIntensityConsistentSelection)
    {
        cv::Mat Imin_left, Imax_left, Imin_right, Imax_right;
        computeBTIntervals(leftF, Imin_left, Imax_left);
        computeBTIntervals(rightF, Imin_right, Imax_right);
        disparity = selectIntensityConsistentDisparity(disparity, leftF, rightF,
                                                        Imin_left, Imax_left, Imin_right, Imax_right,
                                                        minDisp, numDisp, P1, P2);
    }

    // Gap interpolation (Hirschmuller 2008, Sec 2.5.3): pushes coverage to 100% but currently
    // worsens accuracy (mean error 31->34px, photometric MAE 3.9->34.2) as >80% of
    // pixels start invalid. This should be used to fill small gaps not going to help with such
    // low coverage. Kept available, off by default (see PipelineConfig).
    return useGapFill ? interpolateGaps(disparity, dispRight, minDisp, numDisp) : disparity;
}
