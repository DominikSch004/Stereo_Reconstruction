#include "SparseKeyPointMatcher.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

SparseKeyPointMatcher::SparseKeyPointMatcher(float ratioThreshold, FeatureDetector detector)
    : ratioThreshold_(ratioThreshold), detector_(detector),
      sift_(cv::SIFT::create()), orb_(cv::ORB::create(100000)) // For Sift no cap but for orb there is a default cap of 500
{
}

MatchResult SparseKeyPointMatcher::match(const cv::Mat &grayLeft,
                                         const cv::Mat &grayRight) const
{
    MatchResult result;

    cv::Mat descLeft, descRight;

    if (detector_ == FeatureDetector::ORB)
    {
        orb_->detectAndCompute(grayLeft, cv::noArray(),
                               result.keypointsLeft, descLeft);
        orb_->detectAndCompute(grayRight, cv::noArray(),
                               result.keypointsRight, descRight);
    }
    else
    {
        sift_->detectAndCompute(grayLeft, cv::noArray(),
                                result.keypointsLeft, descLeft);
        sift_->detectAndCompute(grayRight, cv::noArray(),
                                result.keypointsRight, descRight);
    }

    if (descLeft.empty() || descRight.empty())
    {
        return result;
    }

    std::vector<std::vector<cv::DMatch>> knnMatches;

    if (detector_ == FeatureDetector::ORB)
    {
        // ORB descriptors are binary; Hamming distance via brute force.
        cv::BFMatcher bf(cv::NORM_HAMMING);
        bf.knnMatch(descLeft, descRight, knnMatches, 2);
    }
    else
    {
        cv::FlannBasedMatcher flann;
        flann.knnMatch(descLeft, descRight, knnMatches, 2);
    }

    std::vector<std::pair<float, cv::DMatch>> rankedMatches;
    for (const auto &m : knnMatches)
    { // Lowe's ratio test
        if (m.size() == 2 && m[0].distance < ratioThreshold_ * m[1].distance)
        {
            result.matches.push_back(m[0]);
        }
    }

    return result;
}

void SparseKeyPointMatcher::extractPoints(const MatchResult &result,
                                          std::vector<cv::Point2f> &ptsLeft,
                                          std::vector<cv::Point2f> &ptsRight)
{
    ptsLeft.clear();
    ptsRight.clear();
    ptsLeft.reserve(result.matches.size());
    ptsRight.reserve(result.matches.size());

    for (const auto &m : result.matches)
    {
        ptsLeft.push_back(result.keypointsLeft[m.queryIdx].pt);
        ptsRight.push_back(result.keypointsRight[m.trainIdx].pt);
    }
}

void SparseKeyPointMatcher::visualize(const MatchResult &result, const cv::Mat &grayLeft, const cv::Mat &grayRight)
{
    size_t good_matches = result.matches.size();
    size_t total_kpts_left = result.keypointsLeft.size();
    size_t total_kpts_right = result.keypointsRight.size();

    // Retention rate: What percentage of Left keypoints successfully found a unique, unambiguous match?
    float retention_ratio = total_kpts_left > 0 ? ((float)good_matches / total_kpts_left) * 100.0f : 0.0f;

    std::cout << "Keypoints — left: " << total_kpts_left
              << ", right: " << total_kpts_right << "\n"
              << "Matches passed ratio test: " << good_matches
              << " (" << std::fixed << std::setprecision(1) << retention_ratio << "% retention)\n";

    cv::Mat vis;
    cv::drawMatches(grayLeft, result.keypointsLeft,
                    grayRight, result.keypointsRight,
                    result.matches, vis,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Draw a semi-transparent HUD background box for text readability
    cv::Mat overlay;
    vis.copyTo(overlay);
    cv::Rect hudRect(10, 10, 420, 110);

    // Check if the image is large enough for the HUD
    if (vis.cols > hudRect.x + hudRect.width && vis.rows > hudRect.y + hudRect.height)
    {
        cv::rectangle(overlay, hudRect, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.6, vis, 0.4, 0, vis); // 60% opacity black box

        // Prepare metric strings
        std::string text1 = "# of Correspondences: " + std::to_string(good_matches);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << retention_ratio;
        std::string text2 = "Lowe's Retention Rate: " + ss.str() + "%";

        std::string text3 = "Keypoints Detected: L:" + std::to_string(total_kpts_left) + " R:" + std::to_string(total_kpts_right);

        // Print text to image
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.6;
        int thick = 2;
        cv::Scalar color(255, 255, 255); // White text

        // Make the success count green, and the stats white
        cv::putText(vis, text1, cv::Point(25, 40), font, scale, cv::Scalar(0, 255, 0), thick);
        cv::putText(vis, text2, cv::Point(25, 75), font, scale, color, thick);
        cv::putText(vis, text3, cv::Point(25, 110), font, scale, color, thick);
    }

    const std::string outPath = "sparse_matches.png";
    cv::imwrite(outPath, vis);
    std::cout << "Saved visualization to " << outPath << "\n";

    cv::imshow("Sparse Key Point Matching correspondences", vis);
    cv::waitKey(0);
}