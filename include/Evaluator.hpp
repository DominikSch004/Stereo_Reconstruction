#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>
#include <Pipeline.hpp>

struct EvaluatorParams
{
    PipelineResult &res;
    const Eigen::Matrix3d &R_gt;
    const Eigen::Vector3d &t_gt;
    const std::vector<cv::Point3f> &gt_pointcloud;
};

struct EightPointParams
{
    const std::vector<cv::Point2f> &ptsL, &ptsR;
    const cv::Mat &F_est;
    const std::vector<bool> &inlierMask;
    const Eigen::Matrix3d &R_gt;
    const Eigen::Vector3d &t_gt;
    Eigen::Matrix3d R_est;
    Eigen::Vector3d t_est;
};

struct EightPointRes
{
    double rot_error_deg;
    double trans_error_deg;
    double epipolar_error;
    double inlier_ratio;
};

struct EvaluatorRes
{
    // 8-point metrics
    EightPointRes eightRes;
    // Point cloud & mesh metrics
    double reprojection_error;
    double mean_absolute_distance;
    double chamfer_accuracy;
    double chamfer_completeness;
};

struct DisparityRes
{
    int minDisp = 0, numDisp = 0;

    // Coverage / range over valid pixels
    double coverage = 0.0;
    long nonBlackPixels = 0;
    long validPixels = 0;
    double dispMin = 0.0, dispMax = 0.0, dispMean = 0.0;

    // Sparse inlier disparity distribution (xL_rect - xR_rect)
    size_t sparseCount = 0;
    double sparseMin = 0.0, sparseP2 = 0.0, sparseMean = 0.0, sparseMedian = 0.0, sparseP98 = 0.0, sparseMax = 0.0;
    int negativeCount = 0;

    // Dense-vs-sparse disparity agreement
    size_t checkedCount = 0;
    double agreementMean = 0.0, agreementMedian = 0.0, agreementMax = 0.0;
    int within2px = 0;

    // Photometric reconstruction error (right -> left warp)
    double photometricMAE = 0.0;
    long photometricSamples = 0;

    // Textureless region analysis (gradient magnitude < 5.0)
    long texturelessCount = 0;
    long invalidNonBlack = 0;
    long invalidAndTextureless = 0;

    bool pass = false;
};

struct RectificationRes
{
    size_t correspondences = 0;
    double meanErr = 0.0, medianErr = 0.0, maxErr = 0.0;
    int within1px = 0;
    bool pass = false;
};

class Evaluator
{
public:
    // High-level orchestrator for computing all metrics
    static EvaluatorRes evaluateMetrics(const EvaluatorParams &params);

    // Evaluate only 8-point metrics
    static EightPointRes evaluateEightPoint(const EightPointParams &eightParams);

    // New cleanly formatted print function
    static void printMetrics(const EvaluatorRes &res);

    static void printEightPoint(const EightPointRes &res);

    // Computes the absolute geometric error between an estimated pose and ground truth.
    // Rotation Error: The geodesic distance (angle in degrees) required to align R_est with R_gt.
    // Translation Error: The scale-invariant angular difference between the directional vectors.
    static void evaluatePose(const Eigen::Matrix3d &R_est, const Eigen::Vector3d &t_est,
                             const Eigen::Matrix3d &R_gt, const Eigen::Vector3d &t_gt,
                             double &rot_error_deg, double &trans_error_deg);

    // Computes the Symmetric Epipolar Distance (in pixels) for a set of point correspondences.
    // This measures the geometric 2D fit by calculating the perpendicular distance from
    // each point to its corresponding epipolar line in both images.
    static double computeSymmetricEpipolarDistance(const std::vector<cv::Point2f> &pts1,
                                                   const std::vector<cv::Point2f> &pts2,
                                                   const cv::Mat &F);

    static double evaluateEpipolarError(const cv::Mat &F,
                                        const std::vector<cv::Point2f> &ptsL,
                                        const std::vector<cv::Point2f> &ptsR,
                                        const std::vector<bool> &inlierMask);

    // Calculates the percentage of matched points that survived the RANSAC filtering process.
    static double computeInlierRatio(const std::vector<bool> &inlierMask);

    static double computeReprojectionError(const std::vector<cv::Point2f> &ptsL,
                                           const std::vector<cv::Point2f> &ptsR,
                                           const cv::Mat &K, const cv::Mat &R, const cv::Mat &t);

    static void computePointCloudMetrics(const cv::Mat &est_dense_pts,
                                         const std::vector<cv::Point3f> &gt_cloud,
                                         double &mad_accuracy,
                                         double &completeness);

    // Dense disparity quality: coverage, sparse-vs-dense agreement, photometric
    // consistency and textureless-region correlation. inPtsL/inPtsR are the
    // (unrectified) sparse RANSAC inlier correspondences; R1/P1/R2/P2 are the
    // rectifying transforms used to bring them into the same frame as disp.
    static DisparityRes evaluateDisparity(const cv::Mat &disp,
                                          const cv::Mat &rectL, const cv::Mat &rectR,
                                          const std::vector<cv::Point2f> &inPtsL,
                                          const std::vector<cv::Point2f> &inPtsR,
                                          const cv::Mat &K,
                                          const cv::Mat &R1, const cv::Mat &P1,
                                          const cv::Mat &R2, const cv::Mat &P2,
                                          int minDisp, int numDisp);

    // Error is reported in native pixels plus a full-res equivalent (÷ scale) line when
    // scale != 1.0, since 1px of error does not mean the same real-world distance at
    // different processing resolutions
    static void printDisparity(const DisparityRes &res, double scale = 0.5);

    // Rectification vertical-alignment error. inL/inR are the (unrectified) sparse
    // RANSAC inlier correspondences; R1/P1/R2/P2 are the rectifying transforms.
    static RectificationRes evaluateRectification(const std::vector<cv::Point2f> &inL,
                                                   const std::vector<cv::Point2f> &inR,
                                                   const cv::Mat &K,
                                                   const cv::Mat &R1, const cv::Mat &P1,
                                                   const cv::Mat &R2, const cv::Mat &P2);

    static void printRectification(const RectificationRes &res);
};