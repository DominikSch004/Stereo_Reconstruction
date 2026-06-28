#include "Evaluator.hpp"
#include <cmath>
#include <algorithm>
#include <opencv2/core/eigen.hpp>

EvaluatorRes Evaluator::evaluateMetrics(const EvaluatorParams &params)
{
    EvaluatorRes result;
    Eigen::Matrix3d R_est;
    Eigen::Vector3d t_est;

    if (!params.res.R_est.empty() && !params.res.t_est.empty())
    {
        cv::cv2eigen(params.res.R_est, R_est);
        cv::cv2eigen(params.res.t_est, t_est);
        evaluatePose(R_est, t_est, params.R_gt, params.t_gt,
                     result.rot_error_deg, result.trans_error_deg);
    }
    else
    {
        result.rot_error_deg = -1.0;
        result.trans_error_deg = -1.0;
    }

    if (!params.res.E.empty() && !params.res.K.empty() && !params.res.inPtsL.empty())
    {
        Eigen::Matrix3d E_est, K_est;
        cv::cv2eigen(params.res.E, E_est);
        cv::cv2eigen(params.res.K, K_est);

        Eigen::Matrix3d K_inv = K_est.inverse();
        Eigen::Matrix3d F_est = K_inv.transpose() * E_est * K_inv;

        cv::Mat F_cv;
        cv::eigen2cv(F_est, F_cv);

        result.epipolar_error = computeSymmetricEpipolarDistance(params.res.inPtsL, params.res.inPtsR, F_cv);
    }
    else
    {
        result.epipolar_error = -1.0;
    }

    result.inlier_ratio = computeInlierRatio(params.res.inlierMask);

    return result;
}

void Evaluator::printMetrics(const EvaluatorRes &res)
{
    std::cout << "\n Pipeline Evaluation Metrics \n";

    std::cout << std::fixed << std::setprecision(4);

    if (res.rot_error_deg >= 0)
    {
        std::cout << "Geodesic Rotation : " << res.rot_error_deg << " deg\n";
        std::cout << "Angular Translation: " << res.trans_error_deg << " deg\n";
    }
    else
    {
        std::cout << "Pose Error        : [Missing / Failed]\n";
    }

    if (res.epipolar_error >= 0)
    {
        std::cout << "Epipolar Error    : " << res.epipolar_error << " px\n";
    }
    else
    {
        std::cout << "Epipolar Error    : [Missing / Failed]\n";
    }

    if (res.inlier_ratio >= 0)
    {
        std::cout << "Inlier Ratio      : " << res.inlier_ratio << " %\n";
    }

    std::cout << std::defaultfloat;
}

void Evaluator::evaluatePose(const Eigen::Matrix3d &R_est, const Eigen::Vector3d &t_est,
                             const Eigen::Matrix3d &R_gt, const Eigen::Vector3d &t_gt,
                             double &rot_error_deg, double &trans_error_deg)
{
    const double PI = std::acos(-1.0);

    // calculate difference using Rotation matrix property.
    Eigen::Matrix3d R_diff = R_est * R_gt.transpose();
    double trace = R_diff.trace();

    // angle = arccos(Trace(R_diff) - 1 / 2)
    double cos_theta = (trace - 1.0) / 2.0;

    // clamp to avoid floating point inaccuracies
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    rot_error_deg = std::acos(cos_theta) * (180.0 / PI);

    // translation angular error
    Eigen::Vector3d t_est_norm = t_est.normalized();
    Eigen::Vector3d t_gt_norm = t_gt.normalized();
    double dot_prod = t_est_norm.dot(t_gt_norm);

    dot_prod = std::clamp(dot_prod, -1.0, 1.0);
    trans_error_deg = std::acos(dot_prod) * (180.0 / PI);
}

double Evaluator::computeSymmetricEpipolarDistance(const std::vector<cv::Point2f> &pts1,
                                                   const std::vector<cv::Point2f> &pts2,
                                                   const cv::Mat &F)
{
    if (pts1.empty() || pts1.size() != pts2.size() || F.empty())
        return -1.0;

    double total_error = 0.0;

    for (size_t i = 0; i < pts1.size(); ++i)
    {
        // conversion to homogeneus coordinates
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat pt2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

        // project point to get its epipolar line in image 2
        cv::Mat l2 = F * pt1;
        double a2 = l2.at<double>(0, 0);
        double b2 = l2.at<double>(1, 0);
        double c2 = l2.at<double>(2, 0);

        // perpendicular point-to-line distance
        double dist2 = std::abs(a2 * pts2[i].x + b2 * pts2[i].y + c2) / std::sqrt(a2 * a2 + b2 * b2);

        // same for image 1
        cv::Mat l1 = F.t() * pt2;
        double a1 = l1.at<double>(0, 0);
        double b1 = l1.at<double>(1, 0);
        double c1 = l1.at<double>(2, 0);

        double dist1 = std::abs(a1 * pts1[i].x + b1 * pts1[i].y + c1) / std::sqrt(a1 * a1 + b1 * b1);

        // accumulate the symmetric error
        total_error += (dist1 + dist2);
    }

    // return the average symmetric distance per point pair
    return total_error / (2.0 * pts1.size());
}

double Evaluator::evaluateEpipolarError(const Eigen::Matrix3d &F_eigen,
                                        const std::vector<cv::Point2f> &ptsL,
                                        const std::vector<cv::Point2f> &ptsR,
                                        const std::vector<bool> &inlierMask)
{
    // filter only the robust inliers
    std::vector<cv::Point2f> inL, inR;
    for (size_t i = 0; i < inlierMask.size(); ++i)
    {
        if (inlierMask[i])
        {
            inL.push_back(ptsL[i]);
            inR.push_back(ptsR[i]);
        }
    }

    if (inL.empty())
        return -1.0;

    cv::Mat F_cv;
    cv::eigen2cv(F_eigen, F_cv);

    return computeSymmetricEpipolarDistance(inL, inR, F_cv);
}
double Evaluator::computeInlierRatio(const std::vector<bool> &inlierMask)
{
    if (inlierMask.empty())
        return 0.0;
    int inlier_count = std::count(inlierMask.begin(), inlierMask.end(), true);
    return ((double)inlier_count / inlierMask.size()) * 100.0;
}