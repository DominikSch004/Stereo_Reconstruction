#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Hard-coded Upper Gamma for s = 1.5 from 4 DoF
inline double upper_gamma_1_5(double x)
{
    return 0.5 * std::sqrt(M_PI) * std::erfc(std::sqrt(x)) + std::sqrt(x) * std::exp(-x);
}

// Hard-coded Upper gamma s = 2.5 from 4 DoF
inline double upper_gamma_2_5(double x)
{
    return 1.5 * upper_gamma_1_5(x) + std::pow(x, 1.5) * std::exp(-x);
}

// MAGSAC++ scoring and weighting logic
class MagsacPlusPlus
{
public:
    double k = 3.64;          // 0.99 quantile multiplier of Chi-square for 4 DoFs
    double maximum_threshold; // user-defined max absolute error
    double maximum_sigma;     // max noise scale = max_threshold / k
    double maximum_sigma_2;
    double maximum_sigma_2_per_2;   // constant in magsac++ loss function
    double maximum_sigma_2_times_2; // denominator to normalize square residuals

    double k2_per_2;
    double gamma_1_5_k;
    double gamma_2_5_k;
    double gamma_2_5_complete;
    double lower_gamma_2_5_k;

    double outlier_loss;                          // max loss assigned to an outlier
    double two_ad_dof_plus_one_per_maximum_sigma; // scaling factor for loss equation
    double one_over_sigma;                        // scaling factor for IRLS
    double weight_zero;                           // weight assigned to a point that perfecly fits the model

    MagsacPlusPlus(double sigmaMax)
    {
        maximum_threshold = sigmaMax;
        maximum_sigma = maximum_threshold / k;
        maximum_sigma_2 = maximum_sigma * maximum_sigma;
        maximum_sigma_2_per_2 = maximum_sigma_2 / 2.0;
        maximum_sigma_2_times_2 = maximum_sigma_2 * 2.0;

        k2_per_2 = (k * k) / 2.0;
        gamma_1_5_k = upper_gamma_1_5(k2_per_2);
        gamma_2_5_k = upper_gamma_2_5(k2_per_2);

        gamma_2_5_complete = 1.329340388; // std::tgamma(2.5)
        lower_gamma_2_5_k = gamma_2_5_complete - gamma_2_5_k;

        double two_ad_dof_minus_one = std::pow(2.0, 1.5);
        double two_ad_dof_plus_one = std::pow(2.0, 2.5);

        outlier_loss = maximum_sigma * two_ad_dof_minus_one * lower_gamma_2_5_k;
        two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;

        double C = 0.25; // C(4) constant for MAGSAC
        one_over_sigma = (C * std::pow(2.0, 1.5)) / maximum_sigma;
        weight_zero = one_over_sigma * (0.886226925 - gamma_1_5_k); // std::tgamma(1.5) = 0.886...
    }

    // Calculates the analytical loss for a single point (Equation 3 variant)
    double calculateLoss(double e2)
    {
        double residual = std::sqrt(e2);
        if (residual > maximum_threshold)
        {
            return outlier_loss;
        }

        double squared_residual_per_sigma = e2 / maximum_sigma_2_times_2;
        double current_gamma_1_5 = upper_gamma_1_5(squared_residual_per_sigma);
        double current_gamma_2_5 = upper_gamma_2_5(squared_residual_per_sigma);
        double current_lower_gamma_2_5 = gamma_2_5_complete - current_gamma_2_5;

        double loss = maximum_sigma_2_per_2 * current_lower_gamma_2_5 +
                      (e2 / 4.0) * (current_gamma_1_5 - gamma_1_5_k);
        return loss * two_ad_dof_plus_one_per_maximum_sigma;
    }

    // Calculates the IRLS weight for a single point (Equation 2)
    double calculateWeight(double e2)
    {
        double residual = std::sqrt(e2);
        if (residual > maximum_threshold)
            return 0.0;
        if (residual < 1e-6)
            return weight_zero;

        double squared_residual_per_sigma = e2 / maximum_sigma_2_times_2;
        double current_gamma_1_5 = upper_gamma_1_5(squared_residual_per_sigma);

        return one_over_sigma * (current_gamma_1_5 - gamma_1_5_k);
    }
};