#include "VisualizationUtils.hpp"
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

        cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
        cv::imshow(windowTitle, combined);
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
                cv::circle(frameL, visualize.history_L[it][i], 6, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                cv::circle(frameR, visualize.history_R[it][i], 6, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

                // Add them to the permanent accumulation trail as tiny GREEN dots
                cv::circle(accumL, visualize.history_L[it][i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                cv::circle(accumR, visualize.history_R[it][i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
            }

            // Stitch images together
            cv::Mat displayImg;
            cv::hconcat(frameL, frameR, displayImg);

            // --- FORMAT TEXT STRINGS ---
            std::string iterText = "Iteration: " + std::to_string(it + 1) + " / " + std::to_string(visualize.history_L.size());

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
                c = (char)cv::waitKey(10); // Standard 10ms playback delay
            }

            if (c == 27)
            { // ESC key
                std::cout << "Playback aborted by user.\n";
                break;
            }
        }

        std::cout << windowName << " finished. Press any key to continue...\n";
        cv::waitKey(0);

        std::cout << "\nFinal " << windowName << " Sampson Error: " << visualize.final_sampson_err << " px.\n";
        std::cout << "Final " << windowName << " Inlier Count: " << visualize.final_inlier_count << "\n";
    }
}