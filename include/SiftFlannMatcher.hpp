#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

struct MatchResult {
    std::vector<cv::KeyPoint> keypointsLeft;
    std::vector<cv::KeyPoint> keypointsRight;
    std::vector<cv::DMatch>   matches;
};

class SiftFlannMatcher {
public:
    explicit SiftFlannMatcher(float ratioThreshold = 0.75f);

    // Detect keypoints, compute descriptors, run ratio-test filtered matching.
    MatchResult match(const cv::Mat& grayLeft, const cv::Mat& grayRight) const;

    // Extract matched point coordinates as parallel float vectors.
    // Output is ready to pass directly to next stage in pipeline
    static void extractPoints(const MatchResult& result,
                              std::vector<cv::Point2f>& ptsLeft,
                              std::vector<cv::Point2f>& ptsRight);

private:
    float ratioThreshold_;
    cv::Ptr<cv::SIFT> sift_;
};