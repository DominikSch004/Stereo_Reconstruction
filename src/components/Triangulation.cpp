#include "Triangulation.hpp"
#include <cmath>
#include <opencv2/calib3d.hpp>

cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2)
{
    cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0),
                  u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1),
                  u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
                  u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0),
                  u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1),
                  u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
                  u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0),
                  u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1),
                  u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
                  u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0),
                  u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1),
                  u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

    cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2, 3),
                  p1.at<double>(1, 3) - u1(1)*p1.at<double>(2, 3),
                  p2.at<double>(0, 3) - u2(0)*p2.at<double>(2, 3),
                  p2.at<double>(1, 3) - u2(1)*p2.at<double>(2, 3));

    cv::Vec3d X;
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

void triangulate_points(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D)
{
    for (size_t i = 0; i < pts1.size(); i++) {
        pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
    }
}

// OpenCV backend: disparity-to-depth reprojection via Q.
static cv::Mat reprojectOpenCV(const cv::Mat &disp32f, const PipelineResult &res)
{
    if (res.Q.empty() || res.Q.rows != 4 || res.Q.cols != 4) {
        std::cerr << "ERROR: PipelineResult::Q is invalid ("
                  << res.Q.rows << "x" << res.Q.cols << "). Cannot reproject.\n";
        return cv::Mat();
    }
    cv::Mat pts3D;
    cv::reprojectImageTo3D(disp32f, pts3D, res.Q, true);
    return pts3D;
}

// Manual backend: per-pixel DLT triangulation using the rectified projections.
// In rectified geometry a pixel (x, y) on the left matches (x - d, y) on the right.
static cv::Mat reprojectManual(const cv::Mat &disp32f, const PipelineResult &res)
{
    if (res.P1r.empty() || res.P2r.empty()) {
        std::cerr << "ERROR: PipelineResult::P1r/P2r are empty. "
                     "Manual triangulation needs the rectified projection matrices.\n";
        return cv::Mat();
    }

    cv::Mat pts3D(disp32f.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < disp32f.rows; ++y) {
        for (int x = 0; x < disp32f.cols; ++x) {
            float d = disp32f.at<float>(y, x);
            if (!std::isfinite(d)) continue;

            cv::Vec2d u1(x, y);
            cv::Vec2d u2(x - d, y);
            cv::Vec3d X = triangulate(res.P1r, res.P2r, u1, u2);
            pts3D.at<cv::Vec3f>(y, x) = cv::Vec3f((float)X[0], (float)X[1], (float)X[2]);
        }
    }
    return pts3D;
}

cv::Mat reprojectDisparityTo3D(const cv::Mat &disp32f, const PipelineResult &res, TriangulationMethod method)
{
    switch (method) {
        case TriangulationMethod::Manual: return reprojectManual(disp32f, res);
        case TriangulationMethod::OpenCV:
        default:                          return reprojectOpenCV(disp32f, res);
    }
}
