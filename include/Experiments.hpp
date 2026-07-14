#pragma once

#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

struct ContaminatedPointSets
{
    std::vector<std::vector<cv::Point2f>> L;
    std::vector<std::vector<cv::Point2f>> R;
};

class Experiments
{
public:
    /**
     * @brief Experiment 1: Ratio Degradation (Uniform Noise)
     *
     * Injects uniformly distributed random noise into inliers to
     * reach a target outlier ratio (ex: from 0.0 to 0.90). Both left and
     * right point sets are processed simultaneously to ensure 1-to-1
     * point correspondences are strictly maintained.
     *
     * @param inliersL The original set of pristine inliers for the left image.
     * @param inliersR The synchronized set of pristine inliers for the right image.
     * @param imgSize The dimensions of the image (used for bounding the uniform noise).
     * @param rng A configured random number generator.
     * @param initialRatio The starting outlier-to-inlier ratio (default: 0.0).
     * @param finalRatio The maximum outlier-to-inlier ratio desired (default: 0.9).
     * @param intervalJump The step size to increase the ratio at each iteration (default: 0.01).
     * @param keepTotalConstant If true, randomly removes inliers as outliers are injected so the total number of points remains static.
     * @return A ContaminatedPointSets struct containing synchronized lists of progressively degraded 2D coordinate vectors.
     */
    static ContaminatedPointSets ratioDegradation(
        const std::vector<cv::Point2f> &inliersL,
        const std::vector<cv::Point2f> &inliersR,
        const cv::Size &imgSize,
        std::mt19937 &rng,
        double initialRatio = 0.0,
        double finalRatio = 0.9,
        double intervalJump = 0.01,
        bool keepTotalConstant = true);
};