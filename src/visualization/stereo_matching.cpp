#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "DTULoader.hpp"
#include "Pipeline.hpp"

// Colorize a disparity map over the *known* search range [minDisp, minDisp+numDisp).
// Invalid pixels (outside the range, e.g. SGBM's no-match marker) are painted black so
// they cannot bias the eye or the colour normalization.
static cv::Mat colorizeDisparity(const cv::Mat &disp, int minDisp, int numDisp)
{
    cv::Mat norm(disp.size(), CV_8U, cv::Scalar(0));
    const float lo = float(minDisp);
    const float hi = float(minDisp + numDisp);
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (d >= lo && d < hi)
                norm.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.0f * (d - lo) / (hi - lo));
        }
    cv::Mat color;
    cv::applyColorMap(norm, color, cv::COLORMAP_JET);
    // Force invalid pixels to black after colormap (JET maps 0 -> dark blue, not black).
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d >= lo && d < hi))
                color.at<cv::Vec3b>(y, x) = {0, 0, 0};
        }
    return color;
}

int main(int argc, char **argv)
{
    // 0. Load pipeline configuration (per-step backend selection)
    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig cfg;
    try
    {
        cfg = PipelineConfig::load(configPath);
        // ensure custom method is being used regardless of config
        cfg.disparity = DisparityMethod::Custom;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
    cfg.print();

    // 1. Load data and run the ACTUAL pipeline so we verify the disparity that the
    //    point cloud is built from (not a divergent re-implementation).
    DTULoader loader("../data/dtu/");
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);

    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, cfg))
    {
        std::cerr << "ERROR: pipeline execution failed.\n";
        return -1;
    }

    const cv::Mat &disp = res.denseDisparity;
    const cv::Mat &rectL = res.rectLeft;
    const cv::Mat &rectR = res.rectRight;
    const float lo = float(res.minDisp);
    const float hi = float(res.minDisp + res.numDisp);

    std::cout << "\n--- Disparity map summary ---\n";
    std::cout << "  resolution      : " << disp.cols << " x " << disp.rows << "\n";
    std::cout << "  search range    : [" << res.minDisp << ", " << res.minDisp + res.numDisp << ")\n";

    // 2. Coverage + range stats over valid pixels.
    long valid = 0;
    double dSum = 0.0, dMin = 1e9, dMax = -1e9;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (d >= lo && d < hi)
            {
                ++valid;
                dSum += d;
                dMin = std::min(dMin, static_cast<double>(d));
                dMax = std::max(dMax, static_cast<double>(d));
            }
        }

    // Measure coverage only over the valid rectified image region. Rectification
    // fills pixels outside the source image with zero, where no disparity can exist
    cv::Mat nonBlackMask = rectL > 0;
    long nonBlackPixels = cv::countNonZero(nonBlackMask);
    double coverage = nonBlackPixels > 0 ? 100.0 * valid / nonBlackPixels : 0.0;
    std::cout << "  coverage  : " << std::fixed << std::setprecision(1) << coverage
               << "% (of " << nonBlackPixels << " non black rectified pixels, "
               << std::setprecision(1) << (100.0 * nonBlackPixels / (disp.rows * disp.cols))
               << "% of frame)\n";
    if (valid > 0)
        std::cout << "  disparity range : [" << dMin << ", " << dMax << "], mean "
                  << dSum / valid << "\n";

    // 3. GROUND-TRUTH CHECK: at each sparse inlier correspondence we know the true
    //    rectified disparity (xL_rect - xR_rect). The dense disparity sampled at that
    //    pixel should match it. This is the real verification of stereo matching.
    cv::Mat distC = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(res.inPtsL, rL, res.K, distC, res.R1, res.P1r);
    cv::undistortPoints(res.inPtsR, rR, res.K, distC, res.R2, res.P2r);

    // Inspect sparse disparity distribution and compare with the selected search range.
    std::vector<double> rawDisp;
    rawDisp.reserve(rL.size());
    for (size_t i = 0; i < rL.size(); ++i)
        rawDisp.push_back(rL[i].x - rR[i].x);

    std::vector<double> sorted = rawDisp;
    std::sort(sorted.begin(), sorted.end());
    int negCount = 0;
    double sum = 0.0;
    for (double d : rawDisp)
    {
        if (d < 0.0)
            ++negCount;
        sum += d;
    }

    std::cout << "\n--- Sparse inlier disparity distribution (xL_rect - xR_rect, " << sorted.size() << " pts) ---\n";
    if (!sorted.empty())
    {
        double p2 = sorted[(size_t)(0.02 * sorted.size())];
        double p98 = sorted[(size_t)(0.98 * (sorted.size() - 1))];
        std::cout << "  min=" << sorted.front() << "  p2=" << p2 << "  mean=" << (sum / sorted.size())
                    << "  median=" << sorted[sorted.size() / 2] << "  p98=" << p98
                    << "  max=" << sorted.back() << "\n";
        std::cout << "  negative disparities : " << negCount << " / " << sorted.size()
                    << " (sign-convention check; should be 0 for a standard left-right pair)\n";
        std::cout << "  pipeline chose search range [" << res.minDisp << ", " << (res.minDisp + res.numDisp)
                    << ")  vs  sparse [p2, p98] = [" << p2 << ", " << p98 << "]\n";
    }

    std::vector<double> sparseErr;
    for (size_t i = 0; i < rL.size(); ++i)
    {
        int x = cvRound(rL[i].x), y = cvRound(rL[i].y);
        if (x < 0 || y < 0 || x >= disp.cols || y >= disp.rows)
            continue;
        float dDense = disp.at<float>(y, x);
        if (!(std::isfinite(dDense) && dDense >= lo && dDense < hi))
            continue; // dense matcher produced no valid value here
        double dTrue = rL[i].x - rR[i].x;
        sparseErr.push_back(std::abs(dDense - dTrue));
    }

    double sMean = 0, sMax = 0;
    for (double e : sparseErr)
    {
        sMean += e;
        sMax = std::max(sMax, e);
    }
    sMean = sparseErr.empty() ? 0 : sMean / sparseErr.size();
    std::vector<double> ss = sparseErr;
    std::sort(ss.begin(), ss.end());
    double sMed = ss.empty() ? 0 : ss[ss.size() / 2];
    int within2 = 0;
    for (double e : sparseErr)
    {
        if (e <= 2.0)
            ++within2;
    }

    std::cout << "\n--- Dense-vs-sparse disparity agreement (pixels) ---\n";
    std::cout << "  checked points  : " << sparseErr.size() << " / " << rL.size()
              << " (rest invalid/out-of-bounds)\n";
    std::cout << "  mean |d_dense - d_true| : " << sMean << "\n";
    std::cout << "  median                  : " << sMed << "\n";
    std::cout << "  max                     : " << sMax << "\n";
    std::cout << "  within 2px              : " << within2 << " / " << sparseErr.size()
              << " (" << (sparseErr.empty() ? 0.0 : 100.0 * within2 / sparseErr.size()) << "%)\n";

    // 4. PHOTOMETRIC CONSISTENCY CHECK: warp the right image into the left frame using
    //    the dense disparity (a left pixel (x,y) matches right pixel (x-d, y)).
    //    Correct disparities should reconstruct the left image with low intensity error.
    cv::Mat warpedR(rectL.size(), rectL.type(), cv::Scalar(0));
    cv::Mat photoMask(rectL.size(), CV_8U, cv::Scalar(0));
    double photoSum = 0; long photoN = 0;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d >= lo && d < hi))
                continue;
            int xr = cvRound(x - d);
            if (xr < 0 || xr >= rectR.cols)
                continue;
            uchar vr = rectR.at<uchar>(y, xr);
            warpedR.at<uchar>(y, x) = vr;
            photoMask.at<uchar>(y, x) = 255;
            photoSum += std::abs((int)rectL.at<uchar>(y, x) - (int)vr);
            ++photoN;
        }
    double photoMAE = photoN ? photoSum / photoN : 0;
    std::cout << "\n--- Photometric reconstruction error  (right -> left warp ; intensity 0-255) ---\n";
    std::cout << "  mean abs error  : " << photoMAE << " over " << photoN << " px\n";

    // 5. TEXTURELESS REGION ANALYSIS: quantify how much of the non-black rectified image
    //    has near-zero gradient, and whether invalid disparity pixels correlate with it.
    cv::Mat gradX, gradY, gradMag;
    cv::Sobel(rectL, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(rectL, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);
    const float textureThresh = 5.0f; // gradient magnitude < 5 -> textureless
    cv::Mat texturelessMask = (gradMag < textureThresh) & nonBlackMask;

    long texturelessCount = cv::countNonZero(texturelessMask);
    long invalidNonBlack = 0, invalidAndTextureless = 0;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            if (!nonBlackMask.at<uchar>(y, x))
                continue;
            float d = disp.at<float>(y, x);
            bool isInvalid = !(d >= lo && d < hi);
            bool isTextureless = texturelessMask.at<uchar>(y, x) != 0;
            if (isInvalid)
                ++invalidNonBlack;
            if (isInvalid && isTextureless)
                ++invalidAndTextureless;
        }
    std::cout << "\n--- Textureless region analysis (grad magnitude < " << textureThresh << ") ---\n";
    std::cout << "  textureless    : " << texturelessCount << " / " << nonBlackPixels << " non-black px ("
               << (nonBlackPixels ? 100.0 * texturelessCount / nonBlackPixels : 0.0) << "%)\n";
    std::cout << "  invalid disparity pixels that are textureless : "
               << (invalidNonBlack ? 100.0 * invalidAndTextureless / invalidNonBlack : 0.0) << "% of "
               << invalidNonBlack << " invalid non-black px\n";
    std::cout << "  textureless pixels that are invalid           : "
               << (texturelessCount ? 100.0 * invalidAndTextureless / texturelessCount : 0.0) << "%\n";

    bool pass = (!sparseErr.empty() && sMean <= 2.5 && coverage > 40.0 && (100.0 * within2 / sparseErr.size()) >= 80.0);
    std::cout << "\n  verdict         : "
              << (pass ? "PASS (dense disparity agrees with geometry)"
                       : "CHECK (disparity disagrees with sparse matches / low coverage)")
              << "\n";

    // 6. Visualization: rectified left | colorized disparity | photometric error | textureless overlay.
    cv::Mat vizL;
    cv::cvtColor(rectL, vizL, cv::COLOR_GRAY2BGR);

    cv::Mat vizDisp = colorizeDisparity(disp, res.minDisp, res.numDisp);

    // Textureless overlay: cyan = textureless but still matched, red = textureless AND invalid.
    cv::Mat vizTex;
    cv::cvtColor(rectL, vizTex, cv::COLOR_GRAY2BGR);
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            if (!texturelessMask.at<uchar>(y, x))
                continue;
            float d = disp.at<float>(y, x);
            bool isInvalid = !(d >= lo && d < hi);
            vizTex.at<cv::Vec3b>(y, x) = isInvalid ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 255, 0);
        }

    // Absolute photometric error heatmap (warpedR vs left), masked to valid pixels.
    cv::Mat absErr(rectL.size(), CV_8U, cv::Scalar(0));
    cv::absdiff(rectL, warpedR, absErr);
    absErr.setTo(0, photoMask == 0);
    cv::Mat vizErr;
    cv::applyColorMap(absErr, vizErr, cv::COLORMAP_HOT);
    vizErr.setTo(cv::Scalar(0, 0, 0), photoMask == 0);

    // Overlay sparse ground-truth points on the disparity panel, colored by agreement.
    for (size_t i = 0; i < rL.size(); ++i)
    {
        int x = cvRound(rL[i].x), y = cvRound(rL[i].y);
        if (x < 0 || y < 0 || x >= disp.cols || y >= disp.rows) continue;
        float dDense = disp.at<float>(y, x);
        cv::Scalar c;
        if (!(dDense >= lo && dDense < hi)) c = {255, 0, 255};            // magenta: no dense value
        else if (std::abs(dDense - (rL[i].x - rR[i].x)) <= 2.0) c = {0, 255, 0}; // green: agree
        else c = {0, 0, 255};                                            // red: disagree
        cv::circle(vizDisp, {x, y}, 3, c, 1, cv::LINE_AA);
    }

    auto label = [](cv::Mat &img, const std::string &s) {
        cv::putText(img, s, {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
        cv::putText(img, s, {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 1, cv::LINE_AA);
    };
    label(vizL, "Rectified Left");
    {
        std::ostringstream s;
        s << "Disparity  cover=" << std::fixed << std::setprecision(0) << coverage
          << "%  dErr=" << std::setprecision(2) << sMean << "px";
        label(vizDisp, s.str());
    }
    {
        std::ostringstream s;
        s << "Photo error  MAE=" << std::fixed << std::setprecision(1) << photoMAE;
        label(vizErr, s.str());
    }
    {
        std::ostringstream s;
        s << "Textureless  " << std::fixed << std::setprecision(0)
          << (nonBlackPixels ? 100.0 * texturelessCount / nonBlackPixels : 0.0)
          << "%  (cyan=matched, red=invalid)";
        label(vizTex, s.str());
    }

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{vizL, vizDisp, vizErr, vizTex}, combined);

    const std::string outPath = "disparity_verification_custom.png";
    if (cv::imwrite(outPath, combined))
        std::cout << "\nSaved verification image to: " << outPath << "\n";
    else
        std::cerr << "\nWARNING: failed to write " << outPath << "\n";

    if (std::getenv("DISPLAY") != nullptr)
    {
        cv::namedWindow("Disparity Verification", cv::WINDOW_NORMAL);
        cv::imshow("Disparity Verification", combined);
        std::cout << "Press any key in the window to exit.\n";
        cv::waitKey(0);
    }
    return 0;
}
