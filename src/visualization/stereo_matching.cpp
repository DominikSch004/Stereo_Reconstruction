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
            if (d > lo && d < hi)
                norm.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.0f * (d - lo) / (hi - lo));
        }
    cv::Mat color;
    cv::applyColorMap(norm, color, cv::COLORMAP_JET);
    // Force invalid pixels to black after colormap (JET maps 0 -> dark blue, not black).
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d > lo && d < hi))
                color.at<cv::Vec3b>(y, x) = {0, 0, 0};
        }
    return color;
}

int main()
{
    // 1. Load data and run the ACTUAL pipeline so we verify the disparity that the
    //    point cloud is built from (not a divergent re-implementation).
    DTULoader loader("../data/dtu/");
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat K = loader.loadIntrinsicCV(1);

    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res))
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
            if (d > lo && d < hi)
            {
                ++valid;
                dSum += d;
                dMin = std::min(dMin, (double)d);
                dMax = std::max(dMax, (double)d);
            }
        }
    double coverage = 100.0 * valid / (disp.rows * disp.cols);
    std::cout << "  valid coverage  : " << std::fixed << std::setprecision(1) << coverage << "%\n";
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

    std::vector<double> sparseErr;
    for (size_t i = 0; i < rL.size(); ++i)
    {
        int x = cvRound(rL[i].x), y = cvRound(rL[i].y);
        if (x < 0 || y < 0 || x >= disp.cols || y >= disp.rows)
            continue;
        float dDense = disp.at<float>(y, x);
        if (!(dDense > lo && dDense < hi))
            continue; // dense matcher produced no valid value here
        double dTrue = rL[i].x - rR[i].x;
        sparseErr.push_back(std::abs(dDense - dTrue));
    }

    double sMean = 0, sMax = 0;
    for (double e : sparseErr) { sMean += e; sMax = std::max(sMax, e); }
    sMean = sparseErr.empty() ? 0 : sMean / sparseErr.size();
    std::vector<double> ss = sparseErr; std::sort(ss.begin(), ss.end());
    double sMed = ss.empty() ? 0 : ss[ss.size() / 2];
    int within2 = 0; for (double e : sparseErr) if (e <= 2.0) ++within2;

    std::cout << "\n--- Dense-vs-sparse disparity agreement (pixels) ---\n";
    std::cout << "  checked points  : " << sparseErr.size() << " / " << rL.size()
              << " (rest invalid/out-of-bounds)\n";
    std::cout << "  mean |d_dense - d_true| : " << sMean << "\n";
    std::cout << "  median                  : " << sMed << "\n";
    std::cout << "  max                     : " << sMax << "\n";
    std::cout << "  within 2px              : " << within2 << " / " << sparseErr.size()
              << " (" << (sparseErr.empty() ? 0.0 : 100.0 * within2 / sparseErr.size()) << "%)\n";

    // 4. PHOTOMETRIC LEFT-RIGHT CHECK: warp the right image into the left frame using
    //    the dense disparity (a left pixel (x,y) matches right pixel (x-d, y)). Where
    //    disparity is correct the warped right should look like the left.
    cv::Mat warpedR(rectL.size(), rectL.type(), cv::Scalar(0));
    cv::Mat photoMask(rectL.size(), CV_8U, cv::Scalar(0));
    double photoSum = 0; long photoN = 0;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d > lo && d < hi)) continue;
            int xr = cvRound(x - d);
            if (xr < 0 || xr >= rectR.cols) continue;
            uchar vr = rectR.at<uchar>(y, xr);
            warpedR.at<uchar>(y, x) = vr;
            photoMask.at<uchar>(y, x) = 255;
            photoSum += std::abs((int)rectL.at<uchar>(y, x) - (int)vr);
            ++photoN;
        }
    double photoMAE = photoN ? photoSum / photoN : 0;
    std::cout << "\n--- Photometric L-R consistency (intensity 0-255) ---\n";
    std::cout << "  mean abs error  : " << photoMAE << " over " << photoN << " px\n";

    bool pass = (!sparseErr.empty() && sMean < 2.0 && coverage > 30.0);
    std::cout << "\n  verdict         : "
              << (pass ? "PASS (dense disparity agrees with geometry)"
                       : "CHECK (disparity disagrees with sparse matches / low coverage)")
              << "\n";

    // 5. Visualization: rectified left | colorized disparity | photometric error.
    cv::Mat vizL;
    cv::cvtColor(rectL, vizL, cv::COLOR_GRAY2BGR);

    cv::Mat vizDisp = colorizeDisparity(disp, res.minDisp, res.numDisp);

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
        if (!(dDense > lo && dDense < hi)) c = {255, 0, 255};            // magenta: no dense value
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

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{vizL, vizDisp, vizErr}, combined);

    const std::string outPath = "disparity_verification.png";
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
