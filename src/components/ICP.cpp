#include "ICP.hpp"
#include <opencv2/flann.hpp>
#include <vector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Functor for calculating confidence-weighted point-to-point residual
struct WeightedPointToPointError {
    WeightedPointToPointError(const Eigen::Vector3f& src, const Eigen::Vector3f& tgt, float weight)
        : src_(src.cast<double>()), tgt_(tgt.cast<double>()), weight_(static_cast<double>(weight)) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const {
        // camera[0,1,2] is Angle-Axis rotation vector (omega)
        // camera[3,4,5] is Translation vector (t)
        T p[3];
        T src_pt[3] = { T(src_[0]), T(src_[1]), T(src_[2]) };

        // Rotate point using Ceres's internal angle-axis rotation utility
        ceres::AngleAxisRotatePoint(camera, src_pt, p); // applies Rodrigues formula internally

        // Add translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Residual = sqrt(weight) * (R*s + t - tgt)
        // Using sqrt(weight) because Ceres minimizes the square of the residual:
        // [sqrt(w) * error]^2 = w * error^2
        T sqrt_w = T(std::sqrt(weight_));
        residuals[0] = sqrt_w * (p[0] - T(tgt_[0]));
        residuals[1] = sqrt_w * (p[1] - T(tgt_[1]));
        residuals[2] = sqrt_w * (p[2] - T(tgt_[2]));

        return true;
    }

    const Eigen::Vector3d src_;
    const Eigen::Vector3d tgt_;
    const double weight_;
};

Eigen::Matrix4f ICP::align(PointCloud& source, const PointCloud& target,
                            int maxIter, float distThresh, bool useWeights)
{
    // Build FLANN KD-tree on target point sets
    int n = int(target.pts.size());
    cv::Mat targetMat(n, 3, CV_32F);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            targetMat.at<float>(i, j) = target.pts[i](j);
        }
    }

    cv::flann::Index kdtree(targetMat, cv::flann::KDTreeIndexParams(4));

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    for (int iter = 0; iter < maxIter; ++iter)
    {
        int m = int(source.pts.size());
        cv::Mat queryMat(m, 3, CV_32F);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < 3; ++j) {
                queryMat.at<float>(i, j) = source.pts[i](j);
            }
        }

        cv::Mat indices(m, 1, CV_32S);
        cv::Mat dists(m, 1, CV_32F);
        kdtree.knnSearch(queryMat, indices, dists, 1);

        // Collect matching tracking pairs within Euclidean distance parameters
        std::vector<Eigen::Vector3f> src, tgt;
        std::vector<float> srcWeights;
        for (int i = 0; i < m; ++i)
        {
            if (dists.at<float>(i, 0) > distThresh * distThresh) continue;
            src.push_back(source.pts[i]);
            tgt.push_back(target.pts[indices.at<int>(i, 0)]);
            if (useWeights && i < source.weights.size())
                srcWeights.push_back(source.weights[i]);
            else
                srcWeights.push_back(1.0f);
        }
        if ((int)src.size() < 6) break;
        
        // Initialize optimization params: 3 for angle-axis, 3 for translation
        // start with an identity transformation (0 rotation vector, 0 translation)
        double camera_params[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        ceres::Problem problem;
        for (size_t i = 0; i < src.size(); ++i) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<WeightedPointToPointError, 3, 6>(
                    new WeightedPointToPointError(src[i], tgt[i], srcWeights[i]));
            
            problem.AddResidualBlock(cost_function, nullptr, camera_params);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 10; // Low iteration count per ICP step as KD-tree re-associates
        options.minimizer_progress_to_stdout = false; // Keep console clean

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Convert the optimization results back into an Eigen 4x4 Transformation Matrix
        Eigen::Vector3d omega(camera_params[0], camera_params[1], camera_params[2]);
        Eigen::Vector3d trans(camera_params[3], camera_params[4], camera_params[5]);

        double R_arr[9];
        ceres::AngleAxisToRotationMatrix(camera_params, R_arr);

        Eigen::Matrix4f T_iter = Eigen::Matrix4f::Identity();
        T_iter(0,0) = static_cast<float>(R_arr[0]); T_iter(0,1) = static_cast<float>(R_arr[3]); T_iter(0,2) = static_cast<float>(R_arr[6]);
        T_iter(1,0) = static_cast<float>(R_arr[1]); T_iter(1,1) = static_cast<float>(R_arr[4]); T_iter(1,2) = static_cast<float>(R_arr[7]);
        T_iter(2,0) = static_cast<float>(R_arr[2]); T_iter(2,1) = static_cast<float>(R_arr[5]); T_iter(2,2) = static_cast<float>(R_arr[8]);
        
        T_iter(0,3) = static_cast<float>(trans.x());
        T_iter(1,3) = static_cast<float>(trans.y());
        T_iter(2,3) = static_cast<float>(trans.z());

        // Update the source point clouds using the transformation computed
        for (auto& pt : source.pts) {
            Eigen::Vector4f p_h(pt.x(), pt.y(), pt.z(), 1.0f);
            pt = (T_iter * p_h).head<3>();
        }

        // Accumulate global incremental transform matrix
        T = T_iter * T;

        // Check for early termination if incremental update step is minuscule
        if (trans.norm() < 1e-5 && omega.norm() < 1e-5) {
            break;
        }
    }

    return T;
}