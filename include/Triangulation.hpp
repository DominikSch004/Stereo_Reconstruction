#pragma once

#include <vector>
#include <opencv2/core.hpp>

/**
 * @enum TriangulationMethod
 * @brief Selects how a dense disparity map is turned into 3D points.
 */
enum class TriangulationMethod {
    OpenCV,   // cv::reprojectImageTo3D using the disparity-to-depth matrix Q
    Manual    // Per-pixel linear (DLT) triangulation from the rectified projection matrices
};

/**
 * @class Triangulation
 * @brief Math engine for sparse triangulation and dense 3D coordinate reprojections.
 */
class Triangulation
{
public:
    // TODO: This was method was left here but not sure where it's used in code
    /**
     * @brief Sparse multi-point triangulation helper.
     */
    static void triangulatePoints(
        const cv::Mat& p1, 
        const cv::Mat& p2,
        const std::vector<cv::Vec2d>& pts1,
        const std::vector<cv::Vec2d>& pts2,
        std::vector<cv::Vec3d>& pts3D
    );

    /**
     * @brief Transforms a rectified floating-point disparity map into a dense grid of 3D points.
     * @param disp32f The input floating-point disparity map.
     * @param Q The 4x4 disparity-to-depth re-projection matrix.
     * @param P1r The 3x4 rectified left projection matrix (required for Manual method).
     * @param P2r The 3x4 rectified right projection matrix (required for Manual method).
     * @param minDisp Disparity search range start; disp32f values <= minDisp are the
     *        invalid/no-match indicator (both OpenCV's SGBM and the custom disparity
     *        implementations use minDisp - 1) and must not be triangulated (required for
     *        Manual method only as OpenCV's own reprojectImageTo3D handles this internally).
     * @param method Selection between OpenCV baseline and custom per-pixel DLT loops.
     * @return CV_32FC3 matrix containing local spatial camera frame positions.
     */
    static cv::Mat reprojectDisparityTo3D(
        const cv::Mat& disp32f,
        const cv::Mat& Q,
        const cv::Mat& P1r,
        const cv::Mat& P2r,
        int minDisp,
        TriangulationMethod method = TriangulationMethod::OpenCV
    );

private:
    /**
     * @brief Linear Direct Linear Transform (DLT) triangulation for a single point pair.
     * @return cv::Vec3d 3D scene point relative to the left camera's coordinate reference system.
     */
    static cv::Vec3d triangulate(
        const cv::Mat& p1, 
        const cv::Mat& p2,
        const cv::Vec2d& u1, 
        const cv::Vec2d& u2
    );

    /**
     * @brief OpenCV's built-in reprojection function using the Q matrix.
     * @return matrix of 3D points in the left camera's coordinate frame
     */
    static cv::Mat reprojectOpenCV(
        const cv::Mat& disp32f,
        const cv::Mat& Q
    );

    /**
     * @brief Custom manual reprojection using per-pixel DLT triangulation with the rectified projection matrices.
     * @return matrix of 3D points in the left camera's coordinate frame
     */
    static cv::Mat reprojectManual(
        const cv::Mat& disp32f,
        const cv::Mat& P1r,
        const cv::Mat& P2r,
        int minDisp
    );
};