#include "Triangulation.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/calib3d.hpp>

cv::Vec3d Triangulation::triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2)
{
    // Construct the linear system for DLT triangulation Ax = B
    // Matrix A: each row corresponds to the cross product of the image point with the projection matrix rows
    // Vector B: the difference between the projection matrix's last column and the scaled image point
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

void Triangulation::triangulatePoints(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D)
{
    pts3D.reserve(pts3D.size() + pts1.size());
    for (size_t i = 0; i < pts1.size(); i++) {
        pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
    }
}

cv::Mat Triangulation::reprojectOpenCV(const cv::Mat &disp32f, const cv::Mat &Q)
{
    if (Q.empty() || Q.rows != 4 || Q.cols != 4) {
        std::cerr << "ERROR: Q matrix is invalid. Reprojection aborted.\n";
        return cv::Mat();
    }
    cv::Mat pts3D;
    cv::reprojectImageTo3D(disp32f, pts3D, Q, true);
    return pts3D;
}

cv::Mat Triangulation::reprojectManual(const cv::Mat &disp32f, const cv::Mat &P1r, const cv::Mat &P2r, int minDisp)
{
    if (P1r.empty() || P2r.empty()) {
        std::cerr << "ERROR: Manual projection requires populated rectified matrices P1r/P2r.\n";
        return cv::Mat();
    }

    cv::Mat pts3D(disp32f.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < disp32f.rows; ++y) {
        for (int x = 0; x < disp32f.cols; ++x) {
            float d = disp32f.at<float>(y, x);
            // Invalid/no-match value is minDisp - 1
            // invalid pixels are skipped and left as (0,0,0) and so callers should handle them appropriately
            if (!std::isfinite(d) || d <= (float)minDisp) continue;

            cv::Vec2d u1(x, y);
            cv::Vec2d u2(x - d, y);
            cv::Vec3d X = Triangulation::triangulate(P1r, P2r, u1, u2);
            pts3D.at<cv::Vec3f>(y, x) = cv::Vec3f((float)X[0], (float)X[1], (float)X[2]);
        }
    }
    return pts3D;
}

cv::Mat Triangulation::reprojectDisparityTo3D(
    const cv::Mat &disp32f,
    const cv::Mat &Q,
    const cv::Mat &P1r,
    const cv::Mat &P2r,
    int minDisp,
    TriangulationMethod method)
{
    switch (method) {
        case TriangulationMethod::Manual: return reprojectManual(disp32f, P1r, P2r, minDisp);
        case TriangulationMethod::OpenCV:
        default:                          return reprojectOpenCV(disp32f, Q);
    }
}
