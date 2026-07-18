#include "GeometryUtils.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <cmath>

// Row-major 3x3 matrix multiply: C = A * B
template <typename T>
void multiply3x3(const T A[9], const T B[9], T C[9])
{
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            T sum = T(0);
            for (int k = 0; k < 3; ++k)
                sum += A[r * 3 + k] * B[k * 3 + c];
            C[r * 3 + c] = sum;
        }
    }
}

// Row-major 3x3 matrix transpose: At = A^T
template <typename T>
void transpose3x3(const T A[9], T At[9])
{
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
            At[c * 3 + r] = A[r * 3 + c];
    }
}

// Sampson error residual for relative pose (angle-axis, unit t). Same distance
// (Hartley & Zisserman, Multiple View Geometry, 2nd ed., Sec. 11.4.3) used for
// inlier scoring in FundamentalMatrix::sampsonError, here minimized directly
// over (R, t)
struct SampsonResidual
{
    SampsonResidual(const cv::Point2f &pl, const cv::Point2f &pr, const Eigen::Matrix3d &Kinv)
        : pl_(pl), pr_(pr)
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Kinv_[r * 3 + c] = Kinv(r, c);
    }

    template <typename T>
    bool operator()(const T *const angleAxis, const T *const tDir, T *residual) const
    {
        // Ceres outputs rotation in column-major order
        T Rcol[9];
        ceres::AngleAxisToRotationMatrix(angleAxis, Rcol);

        // Convert to row-major for standard [row * 3 + col] indexing
        T R[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R[r * 3 + c] = Rcol[c * 3 + r];

        // Unit translation direction from 2 spherical parameters (theta=azimuth, phi=elevation).
        const T &theta = tDir[0];
        const T &phi = tDir[1];
        const T t0 = cos(phi) * cos(theta);
        const T t1 = cos(phi) * sin(theta);
        const T t2 = sin(phi);

        const T Tx[9] = {
            T(0), -t2, t1,
            t2, T(0), -t0,
            -t1, t0, T(0)};

        T E[9];
        multiply3x3(Tx, R, E);

        T Kinv[9], KinvT[9];
        for (int i = 0; i < 9; ++i)
            Kinv[i] = T(Kinv_[i]);
        transpose3x3(Kinv, KinvT);

        // F = Kinv^T * E * Kinv
        T M[9], F[9];
        multiply3x3(E, Kinv, M);
        multiply3x3(KinvT, M, F);

        // Same Sampson as in FundamentalMatrix::sampsonError (num / sqrt(den)
        // Ceres squares the residual internally, reproducing num^2/den).
        const T lx = T(pl_.x), ly = T(pl_.y); // [lx ly 1]
        const T rx = T(pr_.x), ry = T(pr_.y); // [rx ry 1]

        // left epipolar line: l' = F * xl = [Fl0 Fl1 Fl2]
        const T Fl0 = F[0] * lx + F[1] * ly + F[2];
        const T Fl1 = F[3] * lx + F[4] * ly + F[5];
        const T Fl2 = F[6] * lx + F[7] * ly + F[8];

        // right epipolar line: l = Ft * xr = [Ftr0 Ftr1 not_computed]
        const T Ftr0 = F[0] * rx + F[3] * ry + F[6];
        const T Ftr1 = F[1] * rx + F[4] * ry + F[7];

        const T num = rx * Fl0 + ry * Fl1 + Fl2; // xr^t * F * xl
        const T den = Fl0 * Fl0 + Fl1 * Fl1 + Ftr0 * Ftr0 + Ftr1 * Ftr1;

        residual[0] = num / sqrt(den + T(1e-12));
        return true;
    }

    cv::Point2f pl_, pr_;
    double Kinv_[9];
};

namespace GeometryUtils
{

    bool extractPoseFromFundamental(const Eigen::Matrix3d &F_eigen,
                                    const std::vector<cv::Point2f> &ptsL,
                                    const std::vector<cv::Point2f> &ptsR,
                                    const std::vector<bool> &inlierMask,
                                    const cv::Mat &K,
                                    cv::Mat &R_est,
                                    cv::Mat &t_est)
    {
        // filter points to only include the robust inliers
        std::vector<cv::Point2f> inL, inR;
        for (size_t i = 0; i < inlierMask.size(); ++i)
        {
            if (inlierMask[i])
            {
                inL.push_back(ptsL[i]);
                inR.push_back(ptsR[i]);
            }
        }

        if (inL.size() < 5)
            return false;

        // Convert Eigen F to OpenCV F
        cv::Mat F_cv;
        cv::eigen2cv(F_eigen, F_cv);

        // E = K^T * F * K
        cv::Mat E = K.t() * F_cv * K;

        cv::recoverPose(E, inL, inR, K, R_est, t_est);

        return true;
    }

    bool refinePose(const cv::Mat &K,
                    const std::vector<cv::Point2f> &ptsL,
                    const std::vector<cv::Point2f> &ptsR,
                    cv::Mat &R,
                    cv::Mat &t)
    {
        if (ptsL.size() < 8 || ptsL.size() != ptsR.size() || R.empty() || t.empty())
            return false;

        cv::Mat Kinv_cv = K.inv();
        Eigen::Matrix3d Kinv;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Kinv(r, c) = Kinv_cv.at<double>(r, c);

        // Initial angle-axis from R (column-major, matching Ceres).
        double Rcol[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Rcol[c * 3 + r] = R.at<double>(r, c);
        double angleAxis[3];
        ceres::RotationMatrixToAngleAxis(Rcol, angleAxis);

        // Initial spherical direction from t. Only direction matters here.
        // The input's magnitude is preserved on output.
        cv::Vec3d tvec(t.at<double>(0), t.at<double>(1), t.at<double>(2));
        double originalScale = cv::norm(tvec);
        if (originalScale < 1e-9)
            return false;
        tvec /= originalScale;
        double tDir[2];
        tDir[1] = std::asin(std::clamp(tvec[2], -1.0, 1.0)); // phi (elevation)
        tDir[0] = std::atan2(tvec[1], tvec[0]);               // theta (azimuth)

        ceres::Problem problem;
        for (size_t i = 0; i < ptsL.size(); ++i)
        {
            ceres::CostFunction *cost =
                new ceres::AutoDiffCostFunction<SampsonResidual, 1, 3, 2>(
                    new SampsonResidual(ptsL[i], ptsR[i], Kinv));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), angleAxis, tDir);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 50;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double RcolRefined[9];
        ceres::AngleAxisToRotationMatrix(angleAxis, RcolRefined);
        cv::Mat Rrefined(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Rrefined.at<double>(r, c) = RcolRefined[c * 3 + r];

        const double phi = tDir[1], theta = tDir[0];
        cv::Mat tRefined(3, 1, CV_64F);
        tRefined.at<double>(0) = std::cos(phi) * std::cos(theta) * originalScale;
        tRefined.at<double>(1) = std::cos(phi) * std::sin(theta) * originalScale;
        tRefined.at<double>(2) = std::sin(phi) * originalScale;

        R = Rrefined;
        t = tRefined;
        return true;
    }

}