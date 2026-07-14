#include <Experiments.hpp>
#include <numeric>
#include <algorithm>
#include <cmath>

ContaminatedPointSets Experiments::ratioDegradation(
    const std::vector<cv::Point2f> &inliersL,
    const std::vector<cv::Point2f> &inliersR,
    const cv::Size &imgSize,
    std::mt19937 &rng,
    double initialRatio,
    double finalRatio,
    double intervalJump,
    bool keepTotalConstant)
{
    std::uniform_real_distribution<float> distX(0.0f, (float)imgSize.width);
    std::uniform_real_distribution<float> distY(0.0f, (float)imgSize.height);

    ContaminatedPointSets result;
    int total_original = inliersL.size();

    for (double target_ratio = initialRatio; target_ratio <= finalRatio; target_ratio += intervalJump)
    {
        std::vector<cv::Point2f> current_mixed_L;
        std::vector<cv::Point2f> current_mixed_R;

        int num_outliers = std::round(target_ratio * total_original);
        int num_inliers_to_keep = keepTotalConstant ? (total_original - num_outliers) : total_original;

        // create an index array to safely synchronize L and R
        std::vector<int> indices(total_original);
        std::iota(indices.begin(), indices.end(), 0);

        if (keepTotalConstant)
        {
            // shuffle the indices, NOT the point vectors directly
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        // add the synchronized inliers
        for (int k = 0; k < num_inliers_to_keep; k++)
        {
            int idx = indices[k];
            current_mixed_L.push_back(inliersL[idx]);
            current_mixed_R.push_back(inliersR[idx]);
        }

        // inject uniform random noise pairs
        for (int k = 0; k < num_outliers; k++)
        {
            current_mixed_L.push_back(cv::Point2f(distX(rng), distY(rng)));
            current_mixed_R.push_back(cv::Point2f(distX(rng), distY(rng)));
        }

        // synchronized final shuffle (to mix the new outliers into the inliers)
        int total_mixed = current_mixed_L.size();
        std::vector<int> final_indices(total_mixed);
        std::iota(final_indices.begin(), final_indices.end(), 0);
        std::shuffle(final_indices.begin(), final_indices.end(), rng);

        std::vector<cv::Point2f> shuffled_L(total_mixed);
        std::vector<cv::Point2f> shuffled_R(total_mixed);

        for (int k = 0; k < total_mixed; k++)
        {
            int sync_idx = final_indices[k];
            shuffled_L[k] = current_mixed_L[sync_idx];
            shuffled_R[k] = current_mixed_R[sync_idx];
        }

        result.L.push_back(shuffled_L);
        result.R.push_back(shuffled_R);
    }

    return result;
}
ContaminatedPointSets Experiments::magnitudeDegradation(
    const std::vector<cv::Point2f> &inliersL,
    const std::vector<cv::Point2f> &inliersR,
    const cv::Size &imgSize,
    std::mt19937 &rng,
    double subsetRatio,
    double initialMagnitude,
    double finalMagnitude,
    double magnitudeJump)
{
    ContaminatedPointSets result;

    int total_points = inliersL.size();

    // peturb a subset of the points
    int num_to_perturb = std::round(subsetRatio * total_points);

    // random angle distribution [0, 2*pi]
    std::uniform_real_distribution<float> distTheta(0.0f, 2.0f * (float)CV_PI);

    // shuffle a vector of indices to pick the random subset
    std::vector<int> indices(total_points);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (double magnitude = initialMagnitude; magnitude <= finalMagnitude; magnitude += magnitudeJump)
    {
        std::vector<cv::Point2f> perturbedL = inliersL;
        std::vector<cv::Point2f> perturbedR = inliersR;

        // apply directional offset to the fixed random subset
        for (int k = 0; k < num_to_perturb; k++)
        {
            int idx = indices[k]; // Use the shuffled index
            float theta = distTheta(rng);
            float dx = (float)(magnitude * std::cos(theta));
            float dy = (float)(magnitude * std::sin(theta));

            // inject error strictly into the right image coordinates
            perturbedR[idx].x += dx;
            perturbedR[idx].y += dy;

            // Clamp to image boundaries
            perturbedR[idx].x = std::max(0.0f, std::min((float)imgSize.width - 1.0f, perturbedR[idx].x));
            perturbedR[idx].y = std::max(0.0f, std::min((float)imgSize.height - 1.0f, perturbedR[idx].y));
        }
        result.L.push_back(perturbedL);
        result.R.push_back(perturbedR);
    }

    return result;
}