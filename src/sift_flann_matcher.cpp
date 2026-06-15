#include "SiftFlannMatcher.hpp"

SiftFlannMatcher::SiftFlannMatcher(float ratioThreshold)
    : ratioThreshold_(ratioThreshold)
    , sift_(cv::SIFT::create())
{}

MatchResult SiftFlannMatcher::match(const cv::Mat& grayLeft,
                                    const cv::Mat& grayRight) const
{
    MatchResult result;

    cv::Mat descLeft, descRight;
    sift_->detectAndCompute(grayLeft,  cv::noArray(),
                            result.keypointsLeft,  descLeft);
    sift_->detectAndCompute(grayRight, cv::noArray(),
                            result.keypointsRight, descRight);

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descLeft, descRight, knnMatches, 2);

    for (const auto& m : knnMatches)
        if (m[0].distance < ratioThreshold_ * m[1].distance)
            result.matches.push_back(m[0]);

    return result;
}

void SiftFlannMatcher::extractPoints(const MatchResult& result,
                                     std::vector<cv::Point2f>& ptsLeft,
                                     std::vector<cv::Point2f>& ptsRight)
{
    ptsLeft.clear();
    ptsRight.clear();
    ptsLeft.reserve(result.matches.size());
    ptsRight.reserve(result.matches.size());

    for (const auto& m : result.matches) {
        ptsLeft.push_back(result.keypointsLeft[m.queryIdx].pt);
        ptsRight.push_back(result.keypointsRight[m.trainIdx].pt);
    }
}