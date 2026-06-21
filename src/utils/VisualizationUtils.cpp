#include "VisualizationUtils.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

            cv::line(vizR, cv::Point(0, y0_R), cv::Point(w, y1_R), color, 1);
            cv::line(vizL, cv::Point(0, y0_L), cv::Point(w, y1_L), color, 1);

            cv::circle(vizL, ptsL[i], 4, color, -1);
            cv::circle(vizR, ptsR[i], 4, color, -1);

            ++drawn;
        }

        cv::Mat combined;
        cv::hconcat(vizL, vizR, combined);

        cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
        cv::imshow(windowTitle, combined);
    }
}