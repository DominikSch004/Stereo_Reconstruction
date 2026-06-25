#include "Rectification.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>
#include <array>
#include <algorithm>

namespace {

// --- Minimal hand-rolled linear algebra (no OpenCV algorithms) ---------------
// The custom rectification path does all of its geometry with these plain
// helpers; cv::Mat is only ever used as a number/pixel buffer at the interface.
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

// Read element (r,c) from a small cv::Mat regardless of float/double depth.
inline double el(const cv::Mat& m, int r, int c)
{
    return m.depth() == CV_64F ? m.at<double>(r, c) : m.at<float>(r, c);
}

inline Vec3 matVec(const Mat3& M, const Vec3& v)
{
    return { M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
             M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
             M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] };
}

inline Mat3 matMul(const Mat3& A, const Mat3& B)
{
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
    return C;
}

inline Mat3 transpose(const Mat3& M)
{
    return {{ {M[0][0], M[1][0], M[2][0]},
              {M[0][1], M[1][1], M[2][1]},
              {M[0][2], M[1][2], M[2][2]} }};
}

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
}

inline double norm(const Vec3& v)
{
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline Mat3 inv3(const Mat3& M)
{
    const double det =
        M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
        M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
        M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    const double idet = (std::abs(det) > 1e-12) ? 1.0 / det : 0.0;
    Mat3 R;
    R[0][0] =  (M[1][1]*M[2][2] - M[1][2]*M[2][1]) * idet;
    R[0][1] = -(M[0][1]*M[2][2] - M[0][2]*M[2][1]) * idet;
    R[0][2] =  (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * idet;
    R[1][0] = -(M[1][0]*M[2][2] - M[1][2]*M[2][0]) * idet;
    R[1][1] =  (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * idet;
    R[1][2] = -(M[0][0]*M[1][2] - M[0][2]*M[1][0]) * idet;
    R[2][0] =  (M[1][0]*M[2][1] - M[1][1]*M[2][0]) * idet;
    R[2][1] = -(M[0][0]*M[2][1] - M[0][1]*M[2][0]) * idet;
    R[2][2] =  (M[0][0]*M[1][1] - M[0][1]*M[1][0]) * idet;
    return R;
}

// Store a 3x3 into a fresh CV_64F cv::Mat for the result struct.
cv::Mat toCv(const Mat3& M)
{
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.at<double>(i, j) = M[i][j];
    return out;
}

// Warp src into dst using the rectified->original homography H, with hand-rolled
// bilinear sampling. Replaces cv::remap; cv::Mat is used only as a pixel buffer.
// Out-of-bounds destinations are left black (constant border).
void warpInverse(const cv::Mat& src, cv::Mat& dst, const Mat3& H, const cv::Size& size)
{
    const int ch = src.channels();
    dst.create(size, src.type());
    dst.setTo(cv::Scalar::all(0));

    for (int v = 0; v < size.height; ++v) {
        uchar* drow = dst.ptr<uchar>(v);
        for (int u = 0; u < size.width; ++u) {
            const double w = H[2][0]*u + H[2][1]*v + H[2][2];
            if (std::abs(w) < 1e-12) continue;
            const double sx = (H[0][0]*u + H[0][1]*v + H[0][2]) / w;
            const double sy = (H[1][0]*u + H[1][1]*v + H[1][2]) / w;
            if (sx < 0.0 || sy < 0.0 || sx > src.cols - 1.0 || sy > src.rows - 1.0)
                continue;

            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const int x1 = std::min(x0 + 1, src.cols - 1);
            const int y1 = std::min(y0 + 1, src.rows - 1);
            const double ax = sx - x0;
            const double ay = sy - y0;

            const uchar* r0 = src.ptr<uchar>(y0);
            const uchar* r1 = src.ptr<uchar>(y1);
            uchar* d = drow + u * ch;
            for (int c = 0; c < ch; ++c) {
                const double top = r0[x0*ch + c] * (1.0 - ax) + r0[x1*ch + c] * ax;
                const double bot = r1[x0*ch + c] * (1.0 - ax) + r1[x1*ch + c] * ax;
                d[c] = static_cast<uchar>(std::lround(top * (1.0 - ay) + bot * ay));
            }
        }
    }
}

} // namespace

bool Rectification::computeCalibrated(
    const cv::Mat& K, const cv::Mat& R, const cv::Mat& t,
    const cv::Size& imageSize, const cv::Mat& grayL, const cv::Mat& grayR,
    const cv::Mat& colorL, RectifyResult& out, RectificationMethod method)
{
    switch (method) {
        case RectificationMethod::CalibratedCustom:
            return computeCalibratedCustom(K, R, t, imageSize, grayL, grayR, colorL, out);
        case RectificationMethod::CalibratedOpenCV:
        default:
            return computeCalibratedOpenCV(K, R, t, imageSize, grayL, grayR, colorL, out);
    }
}

bool Rectification::computeCalibratedOpenCV(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Size& imageSize,
    const cv::Mat& grayL,
    const cv::Mat& grayR,
    const cv::Mat& colorL,
    RectifyResult& out)
{
    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

    // Compute rectifying rotations (R1, R2), projection matrices (P1, P2) and disparity-to-depth mapping (Q)
    cv::stereoRectify(K, dist, K, dist, imageSize, R, t,
                      out.R1, out.R2, out.P1, out.P2, out.Q,
                      cv::CALIB_ZERO_DISPARITY, -1);

    // Generate coordinate mapping lookups storing the pixel correspondences between the original and rectified image planes
    cv::Mat mapAx, mapAy, mapBx, mapBy;
    cv::initUndistortRectifyMap(K, dist, out.R1, out.P1, imageSize, CV_16SC2, mapAx, mapAy);
    cv::initUndistortRectifyMap(K, dist, out.R2, out.P2, imageSize, CV_16SC2, mapBx, mapBy);

    // Warp image planes into row-aligned configurations. Pixels from the original image planes are mapped to the rectified planes via the lookups
    cv::remap(grayL,  out.rectLeft,  mapAx, mapAy, cv::INTER_LINEAR);
    cv::remap(grayR,  out.rectRight, mapBx, mapBy, cv::INTER_LINEAR);
    cv::remap(colorL, out.rectColor, mapAx, mapAy, cv::INTER_LINEAR);
    return true;
}

bool Rectification::computeCalibratedCustom(
    const cv::Mat& K,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Size& imageSize,
    const cv::Mat& grayL,
    const cv::Mat& grayR,
    const cv::Mat& colorL,
    RectifyResult& out)
{
    // Calibrated rectification (Fusiello / Bouguet), hand-rolled with no OpenCV
    // algorithms. Everything is expressed in the LEFT camera frame, where the
    // left camera sits at the origin with identity orientation and the right
    // camera satisfies X_right = R*X_left + t. We synthesise a single rotation
    // that makes both image planes coplanar with the baseline, turning epipolar
    // lines into image rows.

    // Pull the small calibrated quantities out of their cv::Mat containers.
    const Mat3 Kmat = {{ {el(K,0,0), el(K,0,1), el(K,0,2)},
                         {el(K,1,0), el(K,1,1), el(K,1,2)},
                         {el(K,2,0), el(K,2,1), el(K,2,2)} }};
    const Mat3 Rmat = {{ {el(R,0,0), el(R,0,1), el(R,0,2)},
                         {el(R,1,0), el(R,1,1), el(R,1,2)},
                         {el(R,2,0), el(R,2,1), el(R,2,2)} }};
    const Vec3 tvec = (t.rows >= 3) ? Vec3{ el(t,0,0), el(t,1,0), el(t,2,0) }
                                    : Vec3{ el(t,0,0), el(t,0,1), el(t,0,2) };

    // Right camera centre in the left frame: solve R*C + t = 0  =>  C = -R^T t.
    const Mat3 Rt = transpose(Rmat);
    Vec3 C = matVec(Rt, tvec);
    C = { -C[0], -C[1], -C[2] };           // baseline vector (left centre -> right centre)
    const double baseline = norm(C);
    if (baseline < 1e-9) {
        std::cerr << "ERROR: Degenerate baseline; cameras share an optical centre.\n";
        return false;
    }

    // Common rectified orientation. The rows are the new camera axes expressed in
    // left-frame coordinates: x along the baseline, y orthogonal to x and the old
    // optical axis, z completing a right-handed frame.
    const Vec3 e1 = { C[0]/baseline, C[1]/baseline, C[2]/baseline };
    const Vec3 k  = { 0.0, 0.0, 1.0 };     // old left optical axis
    Vec3 e2 = cross(k, e1);
    const double n2 = norm(e2);
    if (n2 < 1e-9) {
        std::cerr << "ERROR: Baseline parallel to optical axis; cannot rectify.\n";
        return false;
    }
    e2 = { e2[0]/n2, e2[1]/n2, e2[2]/n2 };
    const Vec3 e3 = cross(e1, e2);

    const Mat3 Rrect = {{ {e1[0], e1[1], e1[2]},
                          {e2[0], e2[1], e2[2]},
                          {e3[0], e3[1], e3[2]} }};

    // Rectifying rotations: the left camera is already at identity, the right one
    // must be undone (R^T) before applying the shared rotation.
    const Mat3 R1 = Rrect;
    const Mat3 R2 = matMul(Rrect, Rt);
    out.R1 = toCv(R1);
    out.R2 = toCv(R2);

    // Shared rectified intrinsics (reuse K) so both views align row-for-row.
    const Mat3 Knew = Kmat;
    const double fx = Knew[0][0];
    const double cx = Knew[0][2];
    const double cy = Knew[1][2];

    // Rectified projection matrices. In the rectified frame the right camera
    // centre sits at +baseline along x, i.e. Tx = -baseline in OpenCV terms.
    const double Tx = -baseline;
    out.P1 = cv::Mat::zeros(3, 4, CV_64F);
    out.P2 = cv::Mat::zeros(3, 4, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            out.P1.at<double>(i, j) = Knew[i][j];
            out.P2.at<double>(i, j) = Knew[i][j];
        }
    out.P2.at<double>(0, 3) = fx * Tx;     // = -fx * baseline

    // Disparity-to-depth mapping: [X Y Z W]^T = Q [x y d 1]^T, giving the usual
    // depth = fx * baseline / disparity. Principal points are identical here, so
    // the (cx - cx')/Tx term vanishes.
    out.Q = cv::Mat::zeros(4, 4, CV_64F);
    out.Q.at<double>(0, 0) = 1.0;
    out.Q.at<double>(1, 1) = 1.0;
    out.Q.at<double>(0, 3) = -cx;
    out.Q.at<double>(1, 3) = -cy;
    out.Q.at<double>(2, 3) = fx;
    out.Q.at<double>(3, 2) = -1.0 / Tx;    // = 1 / baseline
    out.Q.at<double>(3, 3) = 0.0;

    // Rectified->original homographies. A rectified pixel's viewing ray is traced
    // back into the original image: src = K * R_rect^T * Knew^{-1} * [u v 1].
    const Mat3 KnewInv = inv3(Knew);
    const Mat3 Hl = matMul(matMul(Kmat, transpose(R1)), KnewInv); // rect-left  -> orig-left
    const Mat3 Hr = matMul(matMul(Kmat, transpose(R2)), KnewInv); // rect-right -> orig-right

    warpInverse(grayL,  out.rectLeft,  Hl, imageSize);
    warpInverse(grayR,  out.rectRight, Hr, imageSize);
    warpInverse(colorL, out.rectColor, Hl, imageSize);
    return true;
}


bool Rectification::computeUncalibrated(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    const cv::Size& imageSize,
    const cv::Mat& F,
    cv::Mat& H1,
    cv::Mat& H2)
{
    if (ptsL.size() < 8) {
        std::cerr << "ERROR: Insufficient points passed for uncalibrated calculation routines\n";
        return false;
    }

    return cv::stereoRectifyUncalibrated(ptsL, ptsR, F, imageSize, H1, H2);
}

void Rectification::warp(
    const cv::Mat& imgL,
    const cv::Mat& imgR,
    const cv::Mat& H1,
    const cv::Mat& H2,
    cv::Mat& rectL,
    cv::Mat& rectR)
{
    cv::warpPerspective(imgL, rectL, H1, imgL.size());
    cv::warpPerspective(imgR, rectR, H2, imgR.size());
}