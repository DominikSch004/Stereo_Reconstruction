#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "PipelineConfig.hpp"

struct MatchResult {
    std::vector<cv::KeyPoint> keypointsLeft;
    std::vector<cv::KeyPoint> keypointsRight;
    std::vector<cv::DMatch>   matches;
};

class SparseKeyPointMatcher {
public:
    explicit SparseKeyPointMatcher(float ratioThreshold = 0.75f,
                                   FeatureDetector detector = FeatureDetector::SIFT);

    // Detect keypoints, compute descriptors, run ratio-test filtered matching.
    // SIFT uses FLANN matching; ORB uses brute-force Hamming matching.
    MatchResult match(const cv::Mat& grayLeft, const cv::Mat& grayRight) const;

    // Extract matched point coordinates as parallel float vectors.
    static void extractPoints(const MatchResult& result,
                              std::vector<cv::Point2f>& ptsLeft,
                              std::vector<cv::Point2f>& ptsRight);

private:
    float ratioThreshold_;
    FeatureDetector detector_;
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::ORB> orb_;
};