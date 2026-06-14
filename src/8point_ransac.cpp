#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "DTULoader.hpp"

// Hartley normalization: zero mean, average distance sqrt(2)
// Returns T such that p_norm = T * p (homogeneous)
static Eigen::Matrix3d normalizePoints(
    const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& ptsNorm)
{
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= pts.size(); cy /= pts.size();

    double scale = 0;
    for (const auto& p : pts)
        scale += std::sqrt((p.x - cx)*(p.x - cx) + (p.y - cy)*(p.y - cy));
    scale = std::sqrt(2.0) * pts.size() / scale;

    ptsNorm.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
        ptsNorm[i] = { float((pts[i].x - cx) * scale), float((pts[i].y - cy) * scale) };

    Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
    T(0,0) = scale; T(0,2) = -cx * scale;
    T(1,1) = scale; T(1,2) = -cy * scale;
    T(2,2) = 1.0;
    return T;
}

// Normalized 8-point algorithm → Fundamental matrix
static Eigen::Matrix3d eightPoint(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR)
{
    std::vector<cv::Point2f> ptsLn, ptsRn;
    Eigen::Matrix3d Tl = normalizePoints(ptsL, ptsLn);
    Eigen::Matrix3d Tr = normalizePoints(ptsR, ptsRn);

    int n = ptsLn.size();
    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; ++i)
    {
        double xl = ptsLn[i].x, yl = ptsLn[i].y;
        double xr = ptsRn[i].x, yr = ptsRn[i].y;
        A.row(i) << xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1.0;
    }

    // Smallest right-singular vector of A → f
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullV);
    Eigen::VectorXd f = svdA.matrixV().col(8);
    Eigen::Matrix3d F;
    F << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    // Enforce rank-2: zero out smallest singular value
    Eigen::JacobiSVD<Eigen::Matrix3d> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = svdF.singularValues();
    s(2) = 0.0;
    F = svdF.matrixU() * s.asDiagonal() * svdF.matrixV().transpose();

    // Denormalize: F = Tr^T * F_norm * Tl
    F = Tr.transpose() * F * Tl;
    if (std::abs(F(2,2)) > 1e-10) F /= F(2,2);
    return F;
}

// Sampson distance (first-order approximation of epipolar error)
static double sampsonError(
    const Eigen::Matrix3d& F,
    const cv::Point2f& pl, const cv::Point2f& pr)
{
    Eigen::Vector3d l(pl.x, pl.y, 1.0), r(pr.x, pr.y, 1.0);
    Eigen::Vector3d Fl  = F * l;
    Eigen::Vector3d Ftr = F.transpose() * r;
    double num = r.dot(Fl);
    double den = Fl(0)*Fl(0) + Fl(1)*Fl(1) + Ftr(0)*Ftr(0) + Ftr(1)*Ftr(1);
    return (num * num) / den;
}

// RANSAC + 8-point: returns best F and fills inlierMask
static Eigen::Matrix3d ransacFundamental(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    std::vector<bool>& inlierMask,
    double threshold = 1.0,
    int maxIter = 1000)
{
    const int N = ptsL.size();
    Eigen::Matrix3d bestF = Eigen::Matrix3d::Identity();
    int bestInliers = 0;
    inlierMask.assign(N, false);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int iter = 0; iter < maxIter; ++iter)
    {
        // Sample 8 unique point pairs
        std::vector<int> idx;
        idx.reserve(8);
        while ((int)idx.size() < 8)
        {
            int s = dist(rng);
            if (std::find(idx.begin(), idx.end(), s) == idx.end())
                idx.push_back(s);
        }

        std::vector<cv::Point2f> sL(8), sR(8);
        for (int i = 0; i < 8; ++i) { sL[i] = ptsL[idx[i]]; sR[i] = ptsR[idx[i]]; }

        Eigen::Matrix3d F = eightPoint(sL, sR);

        std::vector<bool> mask(N);
        int inliers = 0;
        for (int i = 0; i < N; ++i)
        {
            mask[i] = sampsonError(F, ptsL[i], ptsR[i]) < threshold;
            if (mask[i]) ++inliers;
        }

        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestF = F;
            inlierMask = mask;
        }
    }

    // Refit on all inliers for a more accurate F
    std::vector<cv::Point2f> inL, inR;
    for (int i = 0; i < N; ++i)
        if (inlierMask[i]) { inL.push_back(ptsL[i]); inR.push_back(ptsR[i]); }

    if ((int)inL.size() >= 8)
    {
        bestF = eightPoint(inL, inR);
        for (int i = 0; i < N; ++i)
            inlierMask[i] = sampsonError(bestF, ptsL[i], ptsR[i]) < threshold;
    }

    return bestF;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <left_image> <right_image>\n";
        return -1;
    }

    DTULoader loader("");
    StereoPair pair = loader.loadPair(std::string(argv[1]), std::string(argv[2]));

    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load images\n";
        return -1;
    }

    auto toGray = [](const FreeImageB& fi) {
        cv::Mat rgba(fi.h, fi.w, CV_8UC4, fi.data);
        cv::Mat gray;
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
        return gray;
    };

    cv::Mat grayLeft  = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    // SIFT + FLANN correspondences
    auto sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kpLeft, kpRight;
    cv::Mat descLeft, descRight;
    sift->detectAndCompute(grayLeft,  cv::noArray(), kpLeft,  descLeft);
    sift->detectAndCompute(grayRight, cv::noArray(), kpRight, descRight);

    cv::FlannBasedMatcher flann;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    flann.knnMatch(descLeft, descRight, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : knnMatches)
        if (m[0].distance < ratioThresh * m[1].distance)
        {
            ptsL.push_back(kpLeft[m[0].queryIdx].pt);
            ptsR.push_back(kpRight[m[0].trainIdx].pt);
        }

    std::cout << "Correspondences after ratio test: " << ptsL.size() << "\n";

    if ((int)ptsL.size() < 8)
    {
        std::cerr << "Not enough correspondences for 8-point algorithm\n";
        return -1;
    }

    // RANSAC + 8-point → Fundamental matrix
    std::vector<bool> inlierMask;
    Eigen::Matrix3d F = ransacFundamental(ptsL, ptsR, inlierMask);

    int nInliers = std::count(inlierMask.begin(), inlierMask.end(), true);
    std::cout << "Inliers: " << nInliers << " / " << ptsL.size() << "\n";
    std::cout << "Fundamental matrix F:\n" << F << "\n";

    // Visualize: epipolar lines in right image for first 20 inliers
    cv::Mat vizLeft, vizRight;
    cv::cvtColor(grayLeft,  vizLeft,  cv::COLOR_GRAY2BGR);
    cv::cvtColor(grayRight, vizRight, cv::COLOR_GRAY2BGR);

    std::mt19937 colorRng(0);
    std::uniform_int_distribution<int> colorDist(50, 255);
    int drawn = 0;
    for (size_t i = 0; i < ptsL.size() && drawn < 20; ++i)
    {
        if (!inlierMask[i]) continue;
        cv::Scalar color(colorDist(colorRng), colorDist(colorRng), colorDist(colorRng));

        Eigen::Vector3d p(ptsL[i].x, ptsL[i].y, 1.0);
        Eigen::Vector3d line = F * p;         // epipolar line in right image: ax + by + c = 0
        int w = grayRight.cols, h = grayRight.rows;
        float y0 = (float)(-(line(2))               / line(1));
        float y1 = (float)(-(line(2) + line(0) * w) / line(1));
        y0 = std::clamp(y0, 0.f, (float)h);
        y1 = std::clamp(y1, 0.f, (float)h);

        cv::line(vizRight, {0, (int)y0}, {w, (int)y1}, color, 1);
        cv::circle(vizLeft,  ptsL[i], 4, color, -1);
        cv::circle(vizRight, ptsR[i], 4, color, -1);
        ++drawn;
    }

    cv::Mat combined;
    cv::hconcat(vizLeft, vizRight, combined);
    cv::namedWindow("Epipolar lines (RANSAC + 8-point)", cv::WINDOW_AUTOSIZE);
    cv::imshow("Epipolar lines (RANSAC + 8-point)", combined);
    std::cout << "Press any key to exit...\n";
    cv::waitKey(0);

    return 0;
}
