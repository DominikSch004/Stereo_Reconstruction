#include "Evaluator.hpp"
#include "ImgUtils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/flann.hpp>

EvaluatorRes Evaluator::evaluateMetrics(const EvaluatorParams &params)
{
    EvaluatorRes result;
    result.reprojection_error = -1.0;
    result.mean_absolute_distance = -1.0;
    result.chamfer_accuracy = -1.0;
    result.chamfer_completeness = -1.0;

    // reprojection error
    if (!params.res.R_est.empty() && !params.res.t_est.empty())
    {
        Eigen::Matrix3d R_est = toEigenMat(params.res.R_est);
        Eigen::Vector3d t_est = toEigenVec(params.res.t_est);

        // Pose Error
        evaluatePose(R_est, t_est, params.R_gt, params.t_gt,
                     result.rot_error_deg, result.trans_error_deg);

        // Reprojection Error (Triangulation Consistency)
        if (!params.res.K.empty() && !params.res.inPtsL.empty())
        {
            result.reprojection_error = computeReprojectionError(
                params.res.inPtsL, params.res.inPtsR,
                params.res.K, params.res.R_est, params.res.t_est);
        }
    }
    else
    {
        result.rot_error_deg = -1.0;
        result.trans_error_deg = -1.0;
    }

    // epipolar error
    if (!params.res.E.empty() && !params.res.K.empty() && !params.res.inPtsL.empty())
    {
        Eigen::Matrix3d E_est = toEigenMat(params.res.E);
        Eigen::Matrix3d K_est = toEigenMat(params.res.K);
        Eigen::Matrix3d K_inv = K_est.inverse();
        Eigen::Matrix3d F_est = K_inv.transpose() * E_est * K_inv;
        cv::Mat F_cv = toCvMat(F_est);

        result.epipolar_error = computeSymmetricEpipolarDistance(
            params.res.inPtsL, params.res.inPtsR, F_cv);
    }
    else
    {
        result.epipolar_error = -1.0;
    }

    // inlier ratio
    result.inlier_ratio = computeInlierRatio(params.res.inlierMask);

    // champfer and mean absolute distance
    if (!params.res.dense3DPoints.empty() && !params.gt_pointcloud.empty())
    {
        computePointCloudMetrics(params.res.dense3DPoints, params.gt_pointcloud,
                                 result.chamfer_accuracy, result.chamfer_completeness);

        result.mean_absolute_distance = result.chamfer_accuracy;
    }

    return result;
}

void Evaluator::printMetrics(const EvaluatorRes &res)
{
    std::cout << "\n Pipeline Evaluation Metrics \n";

    std::cout << " 8-Point Metrics \n";

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

    std::cout << " 3D Reconstruction Quality \n";
    if (res.mean_absolute_distance >= 0)
    {
        std::cout << "Mean Abs Dist (Accuracy): " << res.mean_absolute_distance << " units\n";
        std::cout << "Chamfer (Completeness): " << res.chamfer_completeness << " units\n";
    }
    else
    {
        std::cout << "Metrics            : [Missing GT or Dense Cloud]\n";
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

    cv::Mat F_cv = toCvMat(F_eigen);

    return computeSymmetricEpipolarDistance(inL, inR, F_cv);
}

double Evaluator::computeInlierRatio(const std::vector<bool> &inlierMask)
{
    if (inlierMask.empty())
        return 0.0;
    int inlier_count = std::count(inlierMask.begin(), inlierMask.end(), true);
    return ((double)inlier_count / inlierMask.size()) * 100.0;
}

double Evaluator::computeReprojectionError(const std::vector<cv::Point2f> &ptsL,
                                           const std::vector<cv::Point2f> &ptsR,
                                           const cv::Mat &K, const cv::Mat &R, const cv::Mat &t)
{
    if (ptsL.empty() || ptsL.size() != ptsR.size())
        return -1.0;

    // build unrectified projection matrices
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    R.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    P1 = K * P1;
    P2 = K * P2;

    // triangulate points
    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, ptsL, ptsR, pts4D);

    double total_err = 0.0;
    for (size_t i = 0; i < ptsL.size(); ++i)
    {
        cv::Mat X = pts4D.col(i);
        X /= X.at<double>(3, 0);

        // project back to image 1
        cv::Mat p1_proj = P1 * X;
        cv::Point2f pt1_est(p1_proj.at<double>(0, 0) / p1_proj.at<double>(2, 0),
                            p1_proj.at<double>(1, 0) / p1_proj.at<double>(2, 0));

        // project back to image 2
        cv::Mat p2_proj = P2 * X;
        cv::Point2f pt2_est(p2_proj.at<double>(0, 0) / p2_proj.at<double>(2, 0),
                            p2_proj.at<double>(1, 0) / p2_proj.at<double>(2, 0));

        // accumulate euclidean distance
        double err1 = cv::norm(pt1_est - ptsL[i]);
        double err2 = cv::norm(pt2_est - ptsR[i]);
        total_err += (err1 + err2) / 2.0;
    }

    return total_err / ptsL.size();
}

void Evaluator::computePointCloudMetrics(const cv::Mat &est_dense_pts,
                                         const std::vector<cv::Point3f> &gt_cloud,
                                         double &mad_accuracy,
                                         double &completeness)
{
    // filter out invalid/background points from the estimated dense matrix
    std::vector<cv::Point3f> est_cloud;
    for (int y = 0; y < est_dense_pts.rows; ++y)
    {
        for (int x = 0; x < est_dense_pts.cols; ++x)
        {
            cv::Vec3f pt = est_dense_pts.at<cv::Vec3f>(y, x);
            if (std::isfinite(pt[2]) && pt[2] > 0.1 && pt[2] < 10000.0)
            {
                est_cloud.push_back(cv::Point3f(pt[0], pt[1], pt[2]));
            }
        }
    }

    if (est_cloud.empty() || gt_cloud.empty())
    {
        mad_accuracy = -1.0;
        completeness = -1.0;
        return;
    }

    // convert to raw 2D matrices for FLANN KD-Tree processing
    cv::Mat est_mat(est_cloud.size(), 3, CV_32F, est_cloud.data());
    cv::Mat gt_mat(gt_cloud.size(), 3, CV_32F, (void *)gt_cloud.data());

    cv::Mat indices, dists;

    // computing accuracy: how close is the closest point to the GT?
    cv::flann::Index kdtree_gt(gt_mat, cv::flann::KDTreeIndexParams(4));
    kdtree_gt.knnSearch(est_mat, indices, dists, 1);

    double acc_sum = 0.0;
    for (int i = 0; i < dists.rows; ++i)
    {
        acc_sum += std::sqrt(dists.at<float>(i, 0));
    }
    mad_accuracy = acc_sum / dists.rows;

    // computing completeness: How much of the GT is covered by our estimation?
    cv::flann::Index kdtree_est(est_mat, cv::flann::KDTreeIndexParams(4));
    kdtree_est.knnSearch(gt_mat, indices, dists, 1);

    double comp_sum = 0.0;
    for (int i = 0; i < dists.rows; ++i)
    {
        comp_sum += std::sqrt(dists.at<float>(i, 0));
    }
    completeness = comp_sum / dists.rows;
}
