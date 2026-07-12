#include "Disparity.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>

// For each pixel, the nearest valid (> minDisp) disparity found by walking strictly in
// direction (dx,dy) from that pixel (NaN if the walk reaches the image border without
// finding one). Implemented as a single pass per direction: pixels are visited
// in the order opposite to (dx,dy) so that, by the time a pixel is reached, the answer for
// its (dx,dy) neighbor is already known and can just be carried forward.
cv::Mat Disparity::nearestValidInDirection(const cv::Mat &disp, int minDisp, int rows, int cols, int dx, int dy)
{
    const float NA = std::numeric_limits<float>::quiet_NaN();
    cv::Mat result(rows, cols, CV_32F, cv::Scalar(NA));
    auto isValid = [minDisp](float v) { return v > static_cast<float>(minDisp); };

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
                    if (isValid(v)) nextValid[x] = v;
                }
            }
        } else
        {
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    result.at<float>(y, x) = nextValid[x];
                    float v = disp.at<float>(y, x);
                    if (isValid(v)) nextValid[x] = v;
                }
            }
        }
    }
    else if (dx == dy)
    {
        // main diagonal (1,1) or (-1,-1): state persists per diagonal k = x - y
        std::vector<float> nextValid(rows + cols - 1, NA);
        auto k = [rows](int x, int y) { return x - y + (rows - 1); };
        if (dx > 0)
        {
            for (int y = rows - 1; y >= 0; --y)
            {
                for (int x = cols - 1; x >= 0; --x)
                {
                    int i = k(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v)) nextValid[i] = v;
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
                    if (isValid(v)) nextValid[i] = v;
                }
            }
        }
    }
    else
    {
        // anti-diagonal (1,-1) or (-1,1): state persists per anti-diagonal s = x + y
        std::vector<float> nextValid(rows + cols - 1, NA);
        auto s = [](int x, int y) { return x + y; };
        if (dx > 0) // (1,-1): predecessor (x+1,y-1) is on an earlier row -> scan y ascending
        {
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    int i = s(x, y);
                    result.at<float>(y, x) = nextValid[i];
                    float v = disp.at<float>(y, x);
                    if (isValid(v)) nextValid[i] = v;
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
                    if (isValid(v)) nextValid[i] = v;
                }
            }
        }
    }

    return result;
}

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

cv::Mat Disparity::filterAndComputeConfidence(
    const cv::Mat &left, const cv::Mat &right, cv::Mat &disparityLeft,
    int minDisp, int numDisp, int blockSize, DisparityMethod method,
    float lrMaxDiff, float photometricScale)
{
    CV_Assert(disparityLeft.type() == CV_32F && disparityLeft.size() == left.size());
    lrMaxDiff = std::max(lrMaxDiff, 1e-3f);
    photometricScale = std::max(photometricScale, 1e-3f);

    cv::Mat disparityRight;
    bool rightUsesPositiveConvention = false;
    if (method == DisparityMethod::OpenCVSGBM)
    {
        // OpenCV always defines disparity as x_base-x_match. Swapping the images
        // therefore produces the negative of the left disparity. Cover the exact
        // mirrored search interval (one spare integer at the lower edge is harmless).
        const int minRight = -(minDisp + numDisp);
        disparityRight = computeSGBMOpenCV(right, left, minRight, numDisp, blockSize);
    }
    else
    {
        cv::Mat leftF, rightF;
        left.convertTo(leftF, CV_32F);
        right.convertTo(rightF, CV_32F);
        disparityRight = computeWTADisparity(leftF, rightF, left.rows, left.cols,
                                             minDisp, numDisp, 8, 32, true);
        rightUsesPositiveConvention = true;
    }

    cv::Mat confidence(left.size(), CV_32F, cv::Scalar(0));
    size_t validBefore = 0, validAfter = 0;
    const float invalid = static_cast<float>(minDisp - 1);
    for (int y = 0; y < left.rows; ++y)
    {
        for (int x = 0; x < left.cols; ++x)
        {
            float &d = disparityLeft.at<float>(y, x);
            if (!std::isfinite(d) || d <= static_cast<float>(minDisp))
                continue;
            ++validBefore;

            const float xr = static_cast<float>(x) - d;
            const int qx = cvRound(xr);
            if (qx < 0 || qx >= right.cols)
            {
                d = invalid;
                continue;
            }

            const float dr = disparityRight.at<float>(y, qx);
            const float lrError = rightUsesPositiveConvention ? std::abs(d - dr)
                                                               : std::abs(d + dr);
            if (!std::isfinite(dr) || lrError > lrMaxDiff)
            {
                d = invalid;
                continue;
            }

            // Bilinear sampling is unnecessary at half-resolution here; nearest-pixel
            // photometric agreement is used only as a soft cue, never as a hard reject.
            const float photoError = std::abs(static_cast<float>(left.at<uchar>(y, x)) -
                                              static_cast<float>(right.at<uchar>(y, qx)));
            const float lrConfidence = std::exp(-0.5f * (lrError * lrError) /
                                                (lrMaxDiff * lrMaxDiff));
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
                //so c = cZero, b = (cPlus - cMinus)/2, a = (cMinus + cPlus - 2*cZero)/2
                // x = -b/(2a) = (cMinus - cPlus) / (2 * (cMinus + cPlus - 2*cZero))
                float denom = cMinus + cPlus - 2.0f * cZero ; // 2a; >= 0 since cZero is the min of the three costs
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
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
    };

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
            if (d > static_cast<float>(minDisp))
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
                if (qx < 0 || qx >= cols) continue;
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
                    if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) continue;
                    float v = filled.at<float>(ny, nx);
                    if (v > static_cast<float>(minDisp))
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

cv::Mat Disparity::computeCustom(const cv::Mat &left, const cv::Mat &right, int minDisp, int numDisp, int blockSize)
{
    // blockSize is not used as BT cost volume is computer per-pixel and not within a window
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
    // (about - 1.5px tolerance) the same disparity back.
    // Disagreement -> occlusion or a bad match -> invalidate
    cv::Mat disparity(rows, cols, CV_32F);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float d = dispLeft.at<float>(r, c);
            int qx = cvRound(c - d); // corresponding x in the right image

            bool consistent = (qx >= 0 && qx < cols) && (std::abs(d - dispRight.at<float>(r, qx)) <= 1.5f);
            disparity.at<float>(r, c) = consistent ? d : static_cast<float>(minDisp - 1);
        }
    }

    // Gap interpolation (Hirschmuller 2008, Sec 2.5.3): pushes coverage to 100% but currently
    // worsens accuracy (mean error 31->34px, photometric MAE 3.9->34.2) as >80% of
    // pixels start invalid. This should be used to fill small gaps not going to help with such
    // low coverage. Kept available but disabled by default.
    const bool useGapFill = false;
    return useGapFill ? interpolateGaps(disparity, dispRight, minDisp, numDisp) : disparity;
}
