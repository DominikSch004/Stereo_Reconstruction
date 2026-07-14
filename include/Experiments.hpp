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

    /**
     * @brief Experiment 2: Magnitude Degradation (Localization Error)
     *
     * Isolates a subset of the pristine inliers and applies a random
     * directional pixel offset of increasing magnitude to the right image coordinates.
     * Evaluates how sensitive the solver is to localized feature mismatches.
     *
     * @param inliersL The original set of pristine inliers for the left image.
     * @param inliersR The synchronized set of pristine inliers for the right image.
     * @param imgSize The dimensions of the image (used for clamping perturbations).
     * @param rng A configured random number generator.
     * @param subsetRatio The fraction of total points to perturb (default: 0.30).
     * @param initialMagnitude The starting noise magnitude in pixels (default: 1.0).
     * @param finalMagnitude The maximum noise magnitude in pixels (default: 50.0).
     * @param magnitudeJump The step size to increase the magnitude at each iteration (default: 2.0).
     * @return A ContaminatedPointSets struct containing synchronized lists of progressively perturbed 2D coordinate vectors.
     */
    static ContaminatedPointSets magnitudeDegradation(
        const std::vector<cv::Point2f> &inliersL,
        const std::vector<cv::Point2f> &inliersR,
        const cv::Size &imgSize,
        std::mt19937 &rng,
        double subsetRatio = 0.30,
        double initialMagnitude = 1.0,
        double finalMagnitude = 50.0,
        double magnitudeJump = 2.0);

    /**
     * @brief Experiment 3: Increase Inlier Count
     *
     * Iteratively samples subsets of size N from the pristine inlier pool.
     * To ensure a smooth curve and mitigate the risk of a specific subset
     * being geometrically degenerate, it generates multiple random samples
     * for each N step.
     *
     * @param inliersL The original set of pristine inliers for the left image.
     * @param inliersR The synchronized set of pristine inliers for the right image.
     * @param rng A configured random number generator.
     * @param iterationsPerStep The number of random subsets to generate for each N (default: 5).
     * @param startSampling The minimum number of points to sample (default: 8).
     * @param endSampling The maximum number of points to sample (default: -1, meaning use all available).
     * @param jump The step size between iterations (default: 5).
     * @return A ContaminatedPointSets struct containing synchronized lists of sampled inlier subsets.
     */
    static ContaminatedPointSets inlierSubsets(
        const std::vector<cv::Point2f> &inliersL,
        const std::vector<cv::Point2f> &inliersR,
        std::mt19937 &rng,
        int iterationsPerStep = 5,
        int startSampling = 8,
        int endSampling = -1,
        int jump = 5);
};