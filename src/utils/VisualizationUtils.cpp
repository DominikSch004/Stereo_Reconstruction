#include "VisualizationUtils.hpp"
#include "Evaluator.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace VisualizationUtils
{
    void displayEpipolarMatches(const std::string &windowTitle,
                                const cv::Mat &imgL,
                                const cv::Mat &imgR,
                                const std::vector<cv::Point2f> &ptsL,
                                const std::vector<cv::Point2f> &ptsR,
                                const std::vector<bool> &inlierMask,
                                const Eigen::Matrix3d &F,
                                double rotErrorDeg,
                                double transErrorDeg,
                                double epipolarErrorPx,
                                int maxDrawn)
    {
        // Clone and convert to color for drawing
        cv::Mat vizL = imgL.clone();
        cv::Mat vizR = imgR.clone();

        if (vizL.channels() == 1)
            cv::cvtColor(vizL, vizL, cv::COLOR_GRAY2BGR);
        if (vizR.channels() == 1)
            cv::cvtColor(vizR, vizR, cv::COLOR_GRAY2BGR);

        int w = vizR.cols;
        int drawn = 0;

        // Deterministic seed for consistent point colors
        srand(1337);

        for (size_t i = 0; i < ptsL.size() && drawn < maxDrawn; ++i)
        {
            cv::Scalar color(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);

            if (!inlierMask[i])
                continue;

            // epipolar on right image
            Eigen::Vector3d pL(ptsL[i].x, ptsL[i].y, 1.0);
            Eigen::Vector3d lineR = F * pL;

            int y0_R = cvRound(-lineR(2) / lineR(1));
            int y1_R = cvRound(-(lineR(2) + lineR(0) * w) / lineR(1));

            // epipolar on left image
            Eigen::Vector3d pR(ptsR[i].x, ptsR[i].y, 1.0);
            Eigen::Vector3d lineL = F.transpose() * pR;

            int y0_L = cvRound(-lineL(2) / lineL(1));
            int y1_L = cvRound(-(lineL(2) + lineL(0) * w) / lineL(1));

            cv::line(vizR, cv::Point(0, y0_R), cv::Point(w, y1_R), color, 5);
            cv::line(vizL, cv::Point(0, y0_L), cv::Point(w, y1_L), color, 5);

            cv::circle(vizL, ptsL[i], 12, color, -1);
            cv::circle(vizR, ptsR[i], 12, color, -1);

            ++drawn;
        }

        cv::Mat combined;
        cv::hconcat(vizL, vizR, combined);

        // Add metrics overlay
        std::vector<std::string> metrics = {
            "Metrics:",
            "Rot Error:   " + std::to_string(rotErrorDeg).substr(0, 5) + " deg",
            "Trans Error: " + std::to_string(transErrorDeg).substr(0, 5) + " deg",
            "Epi Error:   " + std::to_string(epipolarErrorPx).substr(0, 5) + " px"};

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
        int thickness = 2;
        int baseline = 0;
        int y_offset = 35;
        int x_offset = 20;

        cv::Mat overlay = combined.clone();
        cv::rectangle(overlay, cv::Point(10, 10), cv::Point(350, 160), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.6, combined, 0.4, 0, combined); // 60% opacity

        for (size_t i = 0; i < metrics.size(); ++i)
        {
            cv::putText(combined, metrics[i],
                        cv::Point(x_offset, y_offset + (i * 30)),
                        font, fontScale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
        }

        cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
        cv::imshow(windowTitle, combined);
        cv::waitKey(0);
    }

    void visualizeOutliers(const std::vector<cv::Point2f> &ptsL,
                           const std::vector<cv::Point2f> &ptsR,
                           const cv::Mat &grayLeft,
                           const cv::Mat &grayRight,
                           const std::vector<bool> &inlierMask,
                           const std::string &leftWindowTitle,
                           const std::string &rightWindowTitle)
    {
        cv::Mat visLeft, visRight;
        cv::cvtColor(grayLeft, visLeft, cv::COLOR_GRAY2BGR);
        cv::cvtColor(grayRight, visRight, cv::COLOR_GRAY2BGR);
        for (size_t p = 0; p < ptsL.size(); p++)
        {
            if (!inlierMask[p])
            {

                cv::circle(visLeft, ptsL[p], 4, cv::Scalar(0, 0, 255), -1);
                cv::circle(visRight, ptsR[p], 4, cv::Scalar(0, 0, 255), -1);
            }
            else
            {
                cv::circle(visLeft, ptsL[p], 6, cv::Scalar(0, 255, 0), -1);
                cv::circle(visRight, ptsR[p], 6, cv::Scalar(0, 255, 0), -1);
            }
        }
        cv::Mat displayImgLeft, displayImgRight;
        cv::resize(visLeft, displayImgLeft, cv::Size(), 0.5, 0.5);
        cv::resize(visRight, displayImgRight, cv::Size(), 0.5, 0.5);

        cv::imshow(leftWindowTitle, displayImgLeft);
        cv::imshow(rightWindowTitle, displayImgRight);
        std::cout << "Press any key on the image window to continue to the experiments...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void fundamentalExplorationVideo(
        const cv::Mat &imgL,
        const cv::Mat &imgR,
        const VisualizationData &visualize,
        const std::string &windowName,
        int pauseInterval)
    {
        // 1. Convert base images to color
        cv::Mat baseL, baseR;
        if (imgL.channels() == 1)
            cv::cvtColor(imgL, baseL, cv::COLOR_GRAY2BGR);
        else
            imgL.copyTo(baseL);

        if (imgR.channels() == 1)
            cv::cvtColor(imgR, baseR, cv::COLOR_GRAY2BGR);
        else
            imgR.copyTo(baseR);

        // 2. Setup the window
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);

        // 3. Keep a "trail" image that accumulates past points in green
        cv::Mat accumL = baseL.clone();
        cv::Mat accumR = baseR.clone();

        // Playback loop
        for (size_t it = 0; it < visualize.history_L.size(); ++it)
        {

            cv::Mat frameL = accumL.clone();
            cv::Mat frameR = accumR.clone();

            for (size_t i = 0; i < visualize.history_L[it].size(); ++i)
            {
                // Draw the CURRENT 8 points in bright RED (active)
                cv::circle(frameL, visualize.history_L[it][i], 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                cv::circle(frameR, visualize.history_R[it][i], 4, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);

                // Add them to the permanent accumulation trail as tiny GREEN dots
                cv::circle(accumL, visualize.history_L[it][i], 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
                cv::circle(accumR, visualize.history_R[it][i], 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
            }

            // Stitch images together
            cv::Mat displayImg;
            cv::hconcat(frameL, frameR, displayImg);

            // --- FORMAT TEXT STRINGS ---
            std::string iterText = "Improvement Step: " + std::to_string(it + 1) + " / " + std::to_string(visualize.history_L.size());

            std::string inlierCountText = std::to_string(visualize.inLierCount[it]) + " inliers.";

            std::string bestInlierText = "Best Inlier Count: " + std::to_string(visualize.currBestInlier[it]);

            // --- DRAW TEXT OVERLAY ---
            // Make the background rectangle taller to fit two lines of text
            cv::rectangle(displayImg, cv::Point(10, 10), cv::Point(400, 130), cv::Scalar(0, 0, 0), cv::FILLED);

            // Draw Iteration Text (Line 1)
            cv::putText(displayImg, iterText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                        1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

            cv::putText(displayImg, inlierCountText, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

            // Draw Sampson Error Text (Line 2) in cyan or yellow so it stands out
            cv::putText(displayImg, bestInlierText, cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

            cv::imshow(windowName, displayImg);

            char c = 0;

            // If a pause interval is set and we've reached it
            if (pauseInterval > 0 && (it + 1) % pauseInterval == 0)
            {
                c = (char)cv::waitKey(0); // Wait infinitely for user input
            }
            else
            {
                c = (char)cv::waitKey(750); // Standard 10ms playback delay
            }

            if (c == 27)
            { // ESC key
                std::cout << "Playback aborted by user.\n";
                break;
            }
        }

        std::cout << windowName << " finished. Press any key to continue...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
        std::cout << "\nFinal " << windowName << " Sampson Error: " << visualize.final_sampson_err << " px.\n";
        std::cout << "Final " << windowName << " Inlier Count: " << visualize.final_inlier_count << "\n";
    }

    void fundamentalComparison(
        const Eigen::Matrix3d &F,
        const Eigen::Matrix3d R_gt,
        const Eigen::Vector3d t_gt,
        const cv::Mat K_cv)
    {
        // 1. Convert OpenCV Intrinsic Matrix (K) to Eigen::Matrix3d
        // Note: Assuming K_cv is of type CV_64F (double).
        Eigen::Matrix3d K;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                K(i, j) = K_cv.at<double>(i, j);
            }
        }

        // 2. Compute Ground Truth Essential Matrix (E_gt)
        // Create skew-symmetric matrix for t_gt
        Eigen::Matrix3d t_x;
        t_x << 0, -t_gt(2), t_gt(1),
            t_gt(2), 0, -t_gt(0),
            -t_gt(1), t_gt(0), 0;

        Eigen::Matrix3d E_gt = t_x * R_gt;

        // 3. Compute Ground Truth Fundamental Matrix (F_gt)
        Eigen::Matrix3d K_inv = K.inverse();
        Eigen::Matrix3d F_gt = K_inv.transpose() * E_gt * K_inv;

        // 4. Normalize both matrices by F(2,2)
        Eigen::Matrix3d F_norm = F / F(2, 2);
        Eigen::Matrix3d F_gt_norm = F_gt / F_gt(2, 2);

        // 5. Handle Sign Ambiguity
        // If subtracting them yields a larger error than adding them, the sign is flipped.
        if ((F_norm - F_gt_norm).norm() > (F_norm + F_gt_norm).norm())
        {
            F_gt_norm = -F_gt_norm;
        }

        Eigen::IOFormat SciFmt(4, 0, ", ", "\n", "[", "]");

        std::cout << "\n       Fundamental Matrix Comparison        \n";

        std::cout << std::scientific;

        std::cout << "[Estimated F]:\n"
                  << F_norm.format(SciFmt) << "\n\n";
        std::cout << "[Ground Truth F]:\n"
                  << F_gt_norm.format(SciFmt) << "\n\n";

        std::cout << std::defaultfloat;

        double frobenius_error = (F_norm - F_gt_norm).norm();
        std::cout << "--> Frobenius Distance Error: " << frobenius_error << "\n";
        std::cout << "\n";
    }

    void visualizeRectification(
        const RectifyResult &rect,
        const std::vector<cv::Point2f> &inL,
        const std::vector<cv::Point2f> &inR,
        const cv::Mat &K,
        const std::string &windowName)
    {
        // 1. Map points to rectified space
        cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
        std::vector<cv::Point2f> rL, rR;
        cv::undistortPoints(inL, rL, K, dist, rect.R1, rect.P1);
        cv::undistortPoints(inR, rR, K, dist, rect.R2, rect.P2);

        // 2. Prepare images
        cv::Mat vizL, vizR;
        cv::cvtColor(rect.rectLeft, vizL, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rect.rectRight, vizR, cv::COLOR_GRAY2BGR);

        // 3. Draw horizontal alignment grid
        for (int y = 0; y < vizL.rows; y += 50)
        {
            cv::line(vizL, {0, y}, {vizL.cols, y}, {80, 80, 80}, 1);
            cv::line(vizR, {0, y}, {vizR.cols, y}, {80, 80, 80}, 1);
        }

        // 4. Combine images to prepare for connectors and metrics
        cv::Mat combined;
        cv::hconcat(vizL, vizR, combined);
        int xOff = vizL.cols;

        // Variables for our metrics
        double sumErr = 0.0;
        double maxErr = 0.0;
        int goodCount = 0;

        // 5. Draw matches and calculate errors
        for (size_t i = 0; i < rL.size(); ++i)
        {
            double dy = std::abs(rL[i].y - rR[i].y);

            // Track metrics
            sumErr += dy;
            if (dy > maxErr)
                maxErr = dy;
            if (dy <= 1.0)
                goodCount++;

            // Determine color based on strict error thresholds
            cv::Scalar color = (dy <= 1.0) ? cv::Scalar(0, 255, 0) : (dy <= 3.0) ? cv::Scalar(0, 255, 255)
                                                                                 : cv::Scalar(0, 0, 255);

            // Cleanly round the floats to integers to prevent narrowing warnings
            cv::Point pt1(cvRound(rL[i].x), cvRound(rL[i].y));
            cv::Point pt2(cvRound(rR[i].x + xOff), cvRound(rR[i].y));

            // Draw points and connector lines
            cv::circle(combined, pt1, 3, color, -1);
            cv::circle(combined, pt2, 3, color, -1);
            cv::line(combined, pt1, pt2, color, 1, cv::LINE_AA);
        }

        // Calculate final metric values
        double meanErr = rL.empty() ? 0.0 : sumErr / rL.size();
        double pctWithin1px = rL.empty() ? 0.0 : (100.0 * goodCount) / rL.size();

        // 6. Construct HUD Text
        std::ostringstream hud;
        hud << "mean dy=" << std::fixed << std::setprecision(2) << meanErr << "px  "
            << "max=" << std::fixed << std::setprecision(2) << maxErr << "px  "
            << "within 1px=" << std::fixed << std::setprecision(2) << pctWithin1px << "%";

        // Draw text with a thick black outline for visibility over bright images
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 0}, 4, cv::LINE_AA);

        // Draw the actual text color based on the mean alignment quality
        cv::Scalar textColor = (meanErr <= 1.0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(combined, hud.str(), {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, textColor, 1, cv::LINE_AA);

        cv::imshow(windowName, combined);
        cv::waitKey(0);
    }
}