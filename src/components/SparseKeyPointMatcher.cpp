#include "SparseKeyPointMatcher.hpp"

namespace {
// Flip this to switch feature detector/matcher used by SparseKeyPointMatcher::match().
enum class DetectorType { SIFT, ORB };
constexpr DetectorType kDetectorType = DetectorType::ORB;
}

SparseKeyPointMatcher::SparseKeyPointMatcher(float ratioThreshold)
    : ratioThreshold_(ratioThreshold),
      sift_(cv::SIFT::create()), orb_(cv::ORB::create(100000))
{
}

MatchResult SparseKeyPointMatcher::match(const cv::Mat &grayLeft,
                                         const cv::Mat &grayRight) const
{
    MatchResult result;

    cv::Mat descLeft, descRight;

    if (kDetectorType == DetectorType::ORB)
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

    if (kDetectorType == DetectorType::ORB)
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