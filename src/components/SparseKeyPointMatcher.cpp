#include "SparseKeyPointMatcher.hpp"
#include <algorithm>

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
    std::cout << "Keypoints — left: " << result.keypointsLeft.size()
              << ", right: " << result.keypointsRight.size() << "\n"
              << "Matches passed ratio test: " << result.matches.size() << "\n";

    cv::Mat vis;
    cv::drawMatches(grayLeft, result.keypointsLeft,
                    grayRight, result.keypointsRight,
                    result.matches, vis,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const std::string outPath = "sparse_matches.png";
    cv::imwrite(outPath, vis);
    std::cout << "Saved visualization to " << outPath
              << " (" << result.matches.size() << " of " << result.matches.size()
              << " matches drawn)\n";

    cv::imshow("Sparse Key Point Matching correspondences", vis);
    cv::waitKey(0);
}