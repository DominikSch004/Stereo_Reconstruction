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
#include "Evaluator.hpp"

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

    DisparityRes dispRes = Evaluator::evaluateDisparity(disp, rectL, rectR, res.inPtsL, res.inPtsR, res.K,
                                                         res.R1, res.P1r, res.R2, res.P2r,
                                                         res.minDisp, res.numDisp);
    Evaluator::printDisparity(dispRes, cfg.processingScale);

    // --- Data needed only for the panels below ---
    cv::Mat nonBlackMask = rectL > 0;

    cv::Mat distC = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(res.inPtsL, rL, res.K, distC, res.R1, res.P1r);
    cv::undistortPoints(res.inPtsR, rR, res.K, distC, res.R2, res.P2r);

    cv::Mat warpedR(rectL.size(), rectL.type(), cv::Scalar(0));
    cv::Mat photoMask(rectL.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d >= lo && d < hi))
                continue;
            int xr = cvRound(x - d);
            if (xr < 0 || xr >= rectR.cols)
                continue;
            warpedR.at<uchar>(y, x) = rectR.at<uchar>(y, xr);
            photoMask.at<uchar>(y, x) = 255;
        }

    cv::Mat gradX, gradY, gradMag;
    cv::Sobel(rectL, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(rectL, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);
    const float textureThresh = 5.0f; // > threshold -> textureless
    cv::Mat texturelessMask = (gradMag < textureThresh) & nonBlackMask;

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
        s << "Disparity  cover=" << std::fixed << std::setprecision(0) << dispRes.coverage
          << "%  dErr=" << std::setprecision(2) << dispRes.agreementMean << "px";
        label(vizDisp, s.str());
    }
    {
        std::ostringstream s;
        s << "Photo error  MAE=" << std::fixed << std::setprecision(1) << dispRes.photometricMAE;
        label(vizErr, s.str());
    }
    {
        std::ostringstream s;
        s << "Textureless  " << std::fixed << std::setprecision(0)
          << (dispRes.nonBlackPixels ? 100.0 * dispRes.texturelessCount / dispRes.nonBlackPixels : 0.0)
          << "%  (cyan=matched, red=invalid)";
        label(vizTex, s.str());
    }

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{vizL, vizDisp, vizErr, vizTex}, combined);

    const std::string outPath = std::string("disparity_verification_") +
                                (cfg.disparity == DisparityMethod::OpenCVSGBM ? "opencv" : "custom") + ".png";
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
