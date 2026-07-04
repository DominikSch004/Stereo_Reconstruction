#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include "DTULoader.hpp"
#include "Pipeline.hpp"

// Triangulate a set of 2D correspondences with two 3x4 projection matrices.
static std::vector<cv::Vec3d> triPts(const cv::Mat &P1, const cv::Mat &P2,
                                     const std::vector<cv::Point2f> &a,
                                     const std::vector<cv::Point2f> &b)
{
    cv::Mat X4;
    cv::triangulatePoints(P1, P2, a, b, X4); // 4 x N homogeneous
    std::vector<cv::Vec3d> out(a.size());
    for (int i = 0; i < X4.cols; ++i)
    {
        float w = X4.at<float>(3, i);
        out[i] = cv::Vec3d(X4.at<float>(0, i) / w, X4.at<float>(1, i) / w, X4.at<float>(2, i) / w);
    }
    return out;
}

static void depthStats(const std::vector<cv::Vec3d> &p, const char *tag)
{
    std::vector<double> z;
    for (auto &v : p) if (std::isfinite(v[2])) z.push_back(v[2]);
    if (z.empty()) { std::cout << "  " << tag << ": no finite points\n"; return; }
    std::sort(z.begin(), z.end());
    double mn = z.front(), mx = z.back(), md = z[z.size() / 2];
    std::cout << "  " << tag << " : Z range [" << std::fixed << std::setprecision(3)
              << mn << ", " << mx << "]  median " << md
              << "  spread(max/min)=" << (mn != 0 ? mx / mn : 0) << "\n";
}

// Umeyama similarity alignment src -> dst; returns RMS residual after alignment.
static double umeyamaRMS(const std::vector<cv::Vec3d> &src, const std::vector<cv::Vec3d> &dst)
{
    int n = 0;
    Eigen::Vector3d mu_s = Eigen::Vector3d::Zero(), mu_d = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < src.size(); ++i)
        if (std::isfinite(src[i][2]) && std::isfinite(dst[i][2]))
        {
            mu_s += Eigen::Vector3d(src[i][0], src[i][1], src[i][2]);
            mu_d += Eigen::Vector3d(dst[i][0], dst[i][1], dst[i][2]);
            ++n;
        }
    if (n < 3) return -1;
    mu_s /= n; mu_d /= n;
    Eigen::Matrix3d Sigma = Eigen::Matrix3d::Zero();
    double varS = 0;
    for (size_t i = 0; i < src.size(); ++i)
        if (std::isfinite(src[i][2]) && std::isfinite(dst[i][2]))
        {
            Eigen::Vector3d xs = Eigen::Vector3d(src[i][0], src[i][1], src[i][2]) - mu_s;
            Eigen::Vector3d xd = Eigen::Vector3d(dst[i][0], dst[i][1], dst[i][2]) - mu_d;
            Sigma += xd * xs.transpose();
            varS += xs.squaredNorm();
        }
    Sigma /= n; varS /= n;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) D(2, 2) = -1;
    Eigen::Matrix3d Rot = svd.matrixU() * D * svd.matrixV().transpose();
    double scale = (svd.singularValues().asDiagonal() * D).trace() / varS;
    Eigen::Vector3d tr = mu_d - scale * Rot * mu_s;

    double sse = 0;
    for (size_t i = 0; i < src.size(); ++i)
        if (std::isfinite(src[i][2]) && std::isfinite(dst[i][2]))
        {
            Eigen::Vector3d xs(src[i][0], src[i][1], src[i][2]);
            Eigen::Vector3d xd(dst[i][0], dst[i][1], dst[i][2]);
            sse += (scale * Rot * xs + tr - xd).squaredNorm();
        }
    return std::sqrt(sse / n);
}

int main()
{
    DTULoader loader("../data/dtu/");
    StereoPair pair = loader.loadPair(1, 2);
    cv::Mat Kfull = loader.loadIntrinsicCV(1);

    PipelineResult res;
    if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, Kfull, res))
    {
        std::cerr << "ERROR: pipeline failed\n";
        return -1;
    }

    const double scale = res.K.at<double>(0, 0) / Kfull.at<double>(0, 0); // processing scale (e.g. 0.5)
    std::cout << "\nProcessing scale: " << scale << "\n";

    // --- Map sparse inliers into rectified coordinates ---
    cv::Mat distC = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(res.inPtsL, rL, res.K, distC, res.R1, res.P1r);
    cv::undistortPoints(res.inPtsR, rR, res.K, distC, res.R2, res.P2r);

    // --- (A) Sparse triangulation via Q (the OpenCV reproject path) ---
    std::vector<cv::Vec3d> X_Q(rL.size());
    for (size_t i = 0; i < rL.size(); ++i)
    {
        double d = rL[i].x - rR[i].x;
        cv::Mat v = res.Q * (cv::Mat_<double>(4, 1) << rL[i].x, rL[i].y, d, 1.0);
        double w = v.at<double>(3);
        X_Q[i] = cv::Vec3d(v.at<double>(0) / w, v.at<double>(1) / w, v.at<double>(2) / w);
    }

    // --- (B) Sparse triangulation via cv::triangulatePoints on rectified P1r/P2r ---
    std::vector<cv::Vec3d> X_tri = triPts(res.P1r, res.P2r, rL, rR);

    // --- (C) Ground-truth metric triangulation via the DTU full-res P matrices ---
    auto loadP = [&](int id) {
        char b[8]; snprintf(b, sizeof(b), "%03d", id);
        std::string p = std::string("../data/dtu/SampleSet/MVS Data/Calibration/cal18/pos_") + b + ".txt";
        std::ifstream f(p);
        cv::Mat P(3, 4, CV_64F);
        for (int i = 0; i < 12; ++i) f >> P.at<double>(i / 4, i % 4);
        return P;
    };
    cv::Mat P1full = loadP(1), P2full = loadP(2);
    std::vector<cv::Point2f> oL(res.inPtsL.size()), oR(res.inPtsR.size());
    for (size_t i = 0; i < oL.size(); ++i)
    {
        oL[i] = res.inPtsL[i] / scale; // back to full-res pixels
        oR[i] = res.inPtsR[i] / scale;
    }
    std::vector<cv::Vec3d> X_gt = triPts(P1full, P2full, oL, oR);

    std::cout << "\n--- Sparse depth statistics (" << rL.size() << " points) ---\n";
    depthStats(X_Q,   "Q reproject  ");
    depthStats(X_tri, "triangulate  ");
    depthStats(X_gt,  "ground truth ");

    // Agreement between the two OpenCV reconstructions (should be ~0).
    double qVsTri = 0; int m = 0;
    for (size_t i = 0; i < X_Q.size(); ++i)
        if (std::isfinite(X_Q[i][2]) && std::isfinite(X_tri[i][2]))
        { qVsTri += cv::norm(X_Q[i] - X_tri[i]); ++m; }
    std::cout << "\n  mean |Q - triangulatePoints| = " << (m ? qVsTri / m : 0)
              << "  (should be ~0 if Q is consistent)\n";

    // Shape agreement vs ground truth (similarity-invariant).
    std::cout << "  Umeyama RMS  Q-path  vs GT = " << umeyamaRMS(X_Q, X_gt) << "\n";
    std::cout << "  GT median depth (mm) = "; depthStats(X_gt, "");

    // --- Dense OpenCV (Q) vs Manual (DLT) reconstruction: are the two triangulation
    //     backends equivalent? The PLY export uses Manual; the pipeline stores OpenCV. ---
    cv::Mat dOpenCV = Triangulation::reprojectDisparityTo3D(res.denseDisparity, res.Q, res.P1r, res.P2r, TriangulationMethod::OpenCV);
    cv::Mat dManual = Triangulation::reprojectDisparityTo3D(res.denseDisparity, res.Q, res.P1r, res.P2r, TriangulationMethod::Manual);
    double dsum = 0; long dn = 0;
    for (int y = 0; y < dOpenCV.rows; ++y)
        for (int x = 0; x < dOpenCV.cols; ++x)
        {
            float d = res.denseDisparity.at<float>(y, x);
            if (d <= res.minDisp) continue;
            cv::Vec3f a = dOpenCV.at<cv::Vec3f>(y, x), b = dManual.at<cv::Vec3f>(y, x);
            if (!std::isfinite(a[2]) || !std::isfinite(b[2]) || a[2] <= 0 || b[2] <= 0) continue;
            dsum += cv::norm(a - b); ++dn;
        }
    std::cout << "  mean |OpenCV - Manual| dense = " << (dn ? dsum / dn : 0)
              << " over " << dn << " px (should be ~0)\n";

    // The sparse inliers are geometrically verified, so their depth band is the
    // object's true Z extent. Anything far outside it is a disparity outlier.
    std::vector<double> zSparse;
    for (auto &p : X_Q) if (std::isfinite(p[2])) zSparse.push_back(p[2]);
    std::sort(zSparse.begin(), zSparse.end());
    double zBandLo = zSparse[(size_t)(0.02 * zSparse.size())];
    double zBandHi = zSparse[(size_t)(0.98 * (zSparse.size() - 1))];
    std::cout << "  sparse (object) Z band = [" << zBandLo << ", " << zBandHi << "]\n";

    // --- Visualization: two top-down (X-Z) scatters, shared scale ---
    // LEFT  = raw dense cloud (green=in object band, red=outlier outside band)
    // RIGHT = dense cloud restricted to the validated band (the real object)
    cv::Mat dense3D = res.dense3DPoints; // CV_32FC3 from reprojectImageTo3D
    struct Pt { float x, z; bool in; };
    std::vector<Pt> pts;
    float xmn = 1e9, xmx = -1e9, zmn = 1e9, zmx = -1e9;
    for (int y = 0; y < dense3D.rows; ++y)
        for (int x = 0; x < dense3D.cols; ++x)
        {
            cv::Vec3f p = dense3D.at<cv::Vec3f>(y, x);
            float d = res.denseDisparity.at<float>(y, x);
            if (d <= res.minDisp) continue;
            if (!std::isfinite(p[0]) || !std::isfinite(p[2]) || p[2] <= 0 || p[2] > 9000) continue;
            bool in = (p[2] >= zBandLo && p[2] <= zBandHi);
            pts.push_back({p[0], p[2], in});
            xmn = std::min(xmn, p[0]); xmx = std::max(xmx, p[0]);
            zmn = std::min(zmn, p[2]); zmx = std::max(zmx, p[2]);
        }

    const int W = 760, H = 820;
    auto panel = [&](bool onlyIn, const std::string &title) {
        cv::Mat plot(H, W, CV_8UC3, cv::Scalar(20, 20, 20));
        auto toPx = [&](float X, float Z) {
            int px = (int)((X - xmn) / (xmx - xmn + 1e-6f) * (W - 40)) + 20;
            int py = H - 40 - (int)((Z - zmn) / (zmx - zmn + 1e-6f) * (H - 80));
            return cv::Point(px, py);
        };
        for (auto &p : pts)
        {
            if (onlyIn && !p.in) continue;
            cv::Scalar c = onlyIn ? cv::Scalar(0, 220, 0)
                                  : (p.in ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 230));
            cv::circle(plot, toPx(p.x, p.z), 1, c, -1);
        }
        cv::putText(plot, title, {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1, cv::LINE_AA);
        return plot;
    };

    long nIn = 0; for (auto &p : pts) if (p.in) ++nIn;
    std::ostringstream l, r;
    l << "RAW dense (X-Z)  red=outlier  Z ratio=" << std::fixed << std::setprecision(1)
      << zmx / std::max(zmn, 1e-3f);
    r << "BAND-FILTERED  kept " << (pts.empty() ? 0 : 100 * nIn / (long)pts.size()) << "%";
    cv::Mat combined;
    cv::hconcat(panel(false, l.str()), panel(true, r.str()), combined);

    const std::string outPath = "triangulation_verification.png";
    cv::imwrite(outPath, combined);
    std::cout << "\nSaved top-down scatter to: " << outPath << "\n";

    if (std::getenv("DISPLAY") != nullptr)
    {
        cv::namedWindow("Triangulation Verification", cv::WINDOW_NORMAL);
        cv::imshow("Triangulation Verification", combined);
        cv::waitKey(0);
    }
    return 0;
}
