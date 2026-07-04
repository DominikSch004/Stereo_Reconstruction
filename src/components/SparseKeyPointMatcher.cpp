#include "SparseKeyPointMatcher.hpp"
#include <algorithm>

SparseKeyPointMatcher::SparseKeyPointMatcher(float ratioThreshold)
    : ratioThreshold_(ratioThreshold), sift_(cv::SIFT::create())
{
}

MatchResult SparseKeyPointMatcher::match(const cv::Mat &grayLeft,
                                         const cv::Mat &grayRight) const
{
    MatchResult result;

    cv::Mat descLeft, descRight;
    sift_->detectAndCompute(grayLeft, cv::noArray(),
                            result.keypointsLeft, descLeft);
    sift_->detectAndCompute(grayRight, cv::noArray(),
                            result.keypointsRight, descRight);

    if (descLeft.empty() || descRight.empty())
    {
        return result;
    }

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descLeft, descRight, knnMatches, 2);

    std::vector<std::pair<float, cv::DMatch>> rankedMatches;
    for (const auto &m : knnMatches)
    { // Lowe's ratio test
        if (m.size() == 2)
        {
            const float ratio = m[0].distance / m[1].distance;
            if (ratio < ratioThreshold_)
            {
                rankedMatches.emplace_back(ratio, m[0]);
            }
        }
    }

    std::sort(rankedMatches.begin(), rankedMatches.end(),
              [](const auto &a, const auto &b) {
                  return a.first < b.first;
              });

    result.matches.reserve(rankedMatches.size());
    for (const auto &ranked : rankedMatches)
    {
        result.matches.push_back(ranked.second);
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
