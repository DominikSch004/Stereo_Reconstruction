#include "Evaluator.hpp"
#include "ImgUtils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
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
                     result.eightRes.rot_error_deg, result.eightRes.trans_error_deg);

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
        result.eightRes.rot_error_deg = -1.0;
        result.eightRes.trans_error_deg = -1.0;
    }

    // epipolar error
    if (!params.res.E.empty() && !params.res.K.empty() && !params.res.inPtsL.empty())
    {
        Eigen::Matrix3d E_est = toEigenMat(params.res.E);
        Eigen::Matrix3d K_est = toEigenMat(params.res.K);
        Eigen::Matrix3d K_inv = K_est.inverse();
        Eigen::Matrix3d F_est = K_inv.transpose() * E_est * K_inv;
        cv::Mat F_cv = toCvMat(F_est);

        result.eightRes.epipolar_error = computeSymmetricEpipolarDistance(
            params.res.inPtsL, params.res.inPtsR, F_cv);
    }
    else
    {
        result.eightRes.epipolar_error = -1.0;
    }

    // inlier ratio
    result.eightRes.inlier_ratio = computeInlierRatio(params.res.inlierMask);

    // champfer and mean absolute distance
    if (!params.res.dense3DPoints.empty() && !params.gt_pointcloud.empty())
    {
        computePointCloudMetrics(params.res.dense3DPoints, params.gt_pointcloud,
                                 result.chamfer_accuracy, result.chamfer_completeness);
        result.mean_absolute_distance = result.chamfer_accuracy;
    }

    return result;
}

EightPointRes Evaluator::evaluateEightPoint(const EightPointParams &eightParams)
{
    EightPointRes result;
    Eigen::Matrix3d R_est = eightParams.R_est;
    Eigen::Vector3d t_est = eightParams.t_est;

    // reprojection error
    if (R_est.size() != 0 && t_est.size() != 0)
    {
        // Pose Error
        evaluatePose(R_est, t_est, eightParams.R_gt, eightParams.t_gt,
                     result.rot_error_deg, result.trans_error_deg);
    }
    else
    {
        result.rot_error_deg = -1.0;
        result.trans_error_deg = -1.0;
    }

    // epipolar error
    if (!eightParams.F_est.empty() && !eightParams.ptsL.empty())
    {
        result.epipolar_error = evaluateEpipolarError(eightParams.F_est, eightParams.ptsL,
                                                      eightParams.ptsR, eightParams.inlierMask);
    }
    else
    {
        result.epipolar_error = -1.0;
    }

    // inlier ratio
    result.inlier_ratio = computeInlierRatio(eightParams.inlierMask);
    return result;
}

void Evaluator::printEightPoint(const EightPointRes &res)
{
    std::cout << " 8-Point Evaluation Metrics \n";

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
}

void Evaluator::printMetrics(const EvaluatorRes &res)
{
    std::cout << "\n Pipeline Evaluation Metrics \n";

    std::cout << " 8-Point Metrics \n";

    std::cout << std::fixed << std::setprecision(4);

    if (res.eightRes.rot_error_deg >= 0)
    {
        std::cout << "Geodesic Rotation : " << res.eightRes.rot_error_deg << " deg\n";
        std::cout << "Angular Translation: " << res.eightRes.trans_error_deg << " deg\n";
    }
    else
    {
        std::cout << "Pose Error        : [Missing / Failed]\n";
    }

    if (res.eightRes.epipolar_error >= 0)
    {
        std::cout << "Epipolar Error    : " << res.eightRes.epipolar_error << " px\n";
    }
    else
    {
        std::cout << "Epipolar Error    : [Missing / Failed]\n";
    }

    if (res.eightRes.inlier_ratio >= 0)
    {
        std::cout << "Inlier Ratio      : " << res.eightRes.inlier_ratio << " %\n";
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

double Evaluator::evaluateEpipolarError(const cv::Mat &F,
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

    return computeSymmetricEpipolarDistance(inL, inR, F);
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

    cv::Mat K64, R64, t64;
    K.convertTo(K64, CV_64F);
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);

    // build unrectified projection matrices
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    R64.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    t64.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    P1 = K64 * P1;
    P2 = K64 * P2;

    // triangulate points
    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, ptsL, ptsR, pts4D);

    pts4D.convertTo(pts4D, CV_64F);

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
    cv::Mat est_pts_float;
    if (est_dense_pts.type() != CV_32FC3)
    {
        est_dense_pts.convertTo(est_pts_float, CV_32FC3);
    }
    else
    {
        est_pts_float = est_dense_pts;
    }

    // filter out invalid/background points from the estimated dense matrix
    std::vector<cv::Point3f> est_cloud;
    for (int y = 0; y < est_pts_float.rows; ++y)
    {
        for (int x = 0; x < est_pts_float.cols; ++x)
        {
            cv::Vec3f pt = est_pts_float.at<cv::Vec3f>(y, x);
            if (std::isfinite(pt[0]) && std::isfinite(pt[1]) && std::isfinite(pt[2]) &&
                pt[2] > 0.1 && pt[2] < 10000.0)
            {
                est_cloud.push_back(cv::Point3f(pt[0], pt[1], pt[2]));
            }
        }
    }

    std::vector<cv::Point3f> clean_gt;
    clean_gt.reserve(gt_cloud.size());
    for (const auto &pt : gt_cloud)
    {
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
        {
            clean_gt.push_back(pt);
        }
    }

    if (est_cloud.empty() || clean_gt.empty())
    {
        mad_accuracy = -1.0;
        completeness = -1.0;
        return;
    }

    // convert to raw 2D matrices for FLANN KD-Tree processing
    cv::Mat est_mat(est_cloud.size(), 3, CV_32F, est_cloud.data());
    cv::Mat gt_mat(clean_gt.size(), 3, CV_32F, (void *)clean_gt.data());

    cv::Mat indices, dists;

    // computing accuracy: how close is the closest point to the GT?
    cv::flann::Index kdtree_gt(gt_mat, cv::flann::KDTreeIndexParams(4));

    cv::Mat indices_acc(est_mat.rows, 1, CV_32S);
    cv::Mat dists_acc(est_mat.rows, 1, CV_32F);

    kdtree_gt.knnSearch(est_mat, indices_acc, dists_acc, 1, cv::flann::SearchParams(32));

    double acc_sum = 0.0;
    for (int i = 0; i < dists_acc.rows; ++i)
    {
        acc_sum += std::sqrt(dists_acc.at<float>(i, 0));
    }
    mad_accuracy = acc_sum / dists_acc.rows;

    // computing completeness: How much of the GT is covered by our estimation?
    cv::flann::Index kdtree_est(est_mat, cv::flann::KDTreeIndexParams(4));

    cv::Mat indices_comp(gt_mat.rows, 1, CV_32S);
    cv::Mat dists_comp(gt_mat.rows, 1, CV_32F);

    kdtree_est.knnSearch(gt_mat, indices_comp, dists_comp, 1, cv::flann::SearchParams(32));

    double comp_sum = 0.0;
    for (int i = 0; i < dists_comp.rows; ++i)
    {
        comp_sum += std::sqrt(dists_comp.at<float>(i, 0));
    }
    completeness = comp_sum / dists_comp.rows;
}

DisparityRes Evaluator::evaluateDisparity(const cv::Mat &disp,
                                          const cv::Mat &rectL, const cv::Mat &rectR,
                                          const std::vector<cv::Point2f> &inPtsL,
                                          const std::vector<cv::Point2f> &inPtsR,
                                          const cv::Mat &K,
                                          const cv::Mat &R1, const cv::Mat &P1,
                                          const cv::Mat &R2, const cv::Mat &P2,
                                          int minDisp, int numDisp)
{
    DisparityRes res;
    res.minDisp = minDisp;
    res.numDisp = numDisp;

    const float lo = float(minDisp);
    const float hi = float(minDisp + numDisp);

    // Coverage + range stats over valid pixels.
    long valid = 0;
    double dSum = 0.0, dMin = 1e9, dMax = -1e9;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (d >= lo && d < hi)
            {
                ++valid;
                dSum += d;
                dMin = std::min(dMin, static_cast<double>(d));
                dMax = std::max(dMax, static_cast<double>(d));
            }
        }

    // Measure coverage only over the valid rectified image region. Rectification
    // fills pixels outside the source image with zero, where no disparity can exist.
    cv::Mat nonBlackMask = rectL > 0;
    long nonBlackPixels = cv::countNonZero(nonBlackMask);
    res.nonBlackPixels = nonBlackPixels;
    res.validPixels = valid;
    res.coverage = nonBlackPixels > 0 ? 100.0 * valid / nonBlackPixels : 0.0;
    if (valid > 0)
    {
        res.dispMin = dMin;
        res.dispMax = dMax;
        res.dispMean = dSum / valid;
    }

    // Ground-truth check: at each sparse inlier correspondence we know the true
    // rectified disparity (xL_rect - xR_rect). The dense disparity sampled at that
    // pixel should match it.
    cv::Mat distC = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(inPtsL, rL, K, distC, R1, P1);
    cv::undistortPoints(inPtsR, rR, K, distC, R2, P2);

    std::vector<double> rawDisp;
    rawDisp.reserve(rL.size());
    for (size_t i = 0; i < rL.size(); ++i)
        rawDisp.push_back(rL[i].x - rR[i].x);

    std::vector<double> sorted = rawDisp;
    std::sort(sorted.begin(), sorted.end());
    res.sparseCount = sorted.size();
    for (double d : rawDisp)
        if (d < 0.0)
            ++res.negativeCount;

    if (!sorted.empty())
    {
        double sum = 0.0;
        for (double d : rawDisp)
            sum += d;
        res.sparseMin = sorted.front();
        res.sparseP2 = sorted[(size_t)(0.02 * sorted.size())];
        res.sparseMean = sum / sorted.size();
        res.sparseMedian = sorted[sorted.size() / 2];
        res.sparseP98 = sorted[(size_t)(0.98 * (sorted.size() - 1))];
        res.sparseMax = sorted.back();
    }

    std::vector<double> sparseErr;
    for (size_t i = 0; i < rL.size(); ++i)
    {
        int x = cvRound(rL[i].x), y = cvRound(rL[i].y);
        if (x < 0 || y < 0 || x >= disp.cols || y >= disp.rows)
            continue;
        float dDense = disp.at<float>(y, x);
        if (!(std::isfinite(dDense) && dDense >= lo && dDense < hi))
            continue; // dense matcher produced no valid value here
        double dTrue = rL[i].x - rR[i].x;
        sparseErr.push_back(std::abs(dDense - dTrue));
    }

    res.checkedCount = sparseErr.size();
    double sMean = 0, sMax = 0;
    for (double e : sparseErr)
    {
        sMean += e;
        sMax = std::max(sMax, e);
    }
    res.agreementMean = sparseErr.empty() ? 0 : sMean / sparseErr.size();
    res.agreementMax = sMax;
    std::vector<double> ss = sparseErr;
    std::sort(ss.begin(), ss.end());
    res.agreementMedian = ss.empty() ? 0 : ss[ss.size() / 2];
    for (double e : sparseErr)
        if (e <= 2.0)
            ++res.within2px;

    // Photometric consistency check: warp the right image into the left frame using
    // the dense disparity (a left pixel (x,y) matches right pixel (x-d, y)). Correct
    // disparities should reconstruct the left image with low intensity error.
    double photoSum = 0;
    long photoN = 0;
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            float d = disp.at<float>(y, x);
            if (!(d >= lo && d < hi))
                continue;
            int xr = cvRound(x - d);
            if (xr < 0 || xr >= rectR.cols)
                continue;
            photoSum += std::abs((int)rectL.at<uchar>(y, x) - (int)rectR.at<uchar>(y, xr));
            ++photoN;
        }
    res.photometricSamples = photoN;
    res.photometricMAE = photoN ? photoSum / photoN : 0;

    // Textureless region analysis: quantify how much of the non-black rectified image
    // has near-zero gradient, and whether invalid disparity pixels correlate with it.
    cv::Mat gradX, gradY, gradMag;
    cv::Sobel(rectL, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(rectL, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);
    const float textureThresh = 5.0f; // gradient magnitude < 5 -> textureless
    cv::Mat texturelessMask = (gradMag < textureThresh) & nonBlackMask;

    res.texturelessCount = cv::countNonZero(texturelessMask);
    for (int y = 0; y < disp.rows; ++y)
        for (int x = 0; x < disp.cols; ++x)
        {
            if (!nonBlackMask.at<uchar>(y, x))
                continue;
            float d = disp.at<float>(y, x);
            bool isInvalid = !(d >= lo && d < hi);
            bool isTextureless = texturelessMask.at<uchar>(y, x) != 0;
            if (isInvalid)
                ++res.invalidNonBlack;
            if (isInvalid && isTextureless)
                ++res.invalidAndTextureless;
        }

    res.pass = (res.checkedCount > 0 && res.agreementMean <= 2.5 && res.coverage > 40.0 &&
                (100.0 * res.within2px / res.checkedCount) >= 80.0);
    return res;
}

void Evaluator::printDisparity(const DisparityRes &res, double scale)
{
    std::cout << "\n--- Disparity map summary ---\n";
    std::cout << "  search range    : [" << res.minDisp << ", " << res.minDisp + res.numDisp << ")\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  coverage        : " << res.coverage << "% (of " << res.nonBlackPixels
              << " non black rectified pixels)\n";
    if (res.validPixels > 0)
        std::cout << "  disparity range : [" << res.dispMin << ", " << res.dispMax << "], mean "
                  << res.dispMean << "\n";

    std::cout << "\n--- Sparse inlier disparity distribution (xL_rect - xR_rect, " << res.sparseCount << " pts) ---\n";
    if (res.sparseCount > 0)
    {
        std::cout << "  min=" << res.sparseMin << "  p2=" << res.sparseP2 << "  mean=" << res.sparseMean
                  << "  median=" << res.sparseMedian << "  p98=" << res.sparseP98 << "  max=" << res.sparseMax << "\n";
        std::cout << "  negative disparities : " << res.negativeCount << " / " << res.sparseCount
                  << " (sign-convention check; should be 0 for a standard left-right pair)\n";
        std::cout << "  pipeline chose search range [" << res.minDisp << ", " << (res.minDisp + res.numDisp)
                  << ")  vs  sparse [p2, p98] = [" << res.sparseP2 << ", " << res.sparseP98 << "]\n";
    }

    std::cout << "\n--- Dense-vs-sparse disparity agreement (pixels) ---\n";
    std::cout << "  checked points  : " << res.checkedCount << " / " << res.sparseCount
              << " (rest invalid/out-of-bounds)\n";
    std::cout << "  mean |d_dense - d_true| : " << res.agreementMean << "\n";
    std::cout << "  median                  : " << res.agreementMedian << "\n";
    std::cout << "  max                     : " << res.agreementMax << "\n";
    std::cout << "  within 2px              : " << res.within2px << " / " << res.checkedCount << " ("
              << (res.checkedCount ? 100.0 * res.within2px / res.checkedCount : 0.0) << "%)\n";
    // 1px of error means a different real-world distance at different scales.
    // Project back to full-resolution-equivalent pixels for runs that used a downscale
    if (scale > 0.0 && scale != 1.0)
        std::cout << "  full-res equivalent (/ processing_scale=" << scale << "):"
                  << "  mean="   << (res.agreementMean / scale)   << "px"
                  << "  median=" << (res.agreementMedian / scale) << "px"
                  << "  max="    << (res.agreementMax / scale)    << "px\n";

    std::cout << "\n--- Photometric reconstruction error (right -> left warp; intensity 0-255) ---\n";
    std::cout << "  mean abs error  : " << res.photometricMAE << " over " << res.photometricSamples << " px\n";

    std::cout << "\n--- Textureless region analysis (grad magnitude < 5.0) ---\n";
    std::cout << "  textureless    : " << res.texturelessCount << " / " << res.nonBlackPixels << " non-black px ("
              << (res.nonBlackPixels ? 100.0 * res.texturelessCount / res.nonBlackPixels : 0.0) << "%)\n";
    std::cout << "  invalid disparity pixels that are textureless : "
              << (res.invalidNonBlack ? 100.0 * res.invalidAndTextureless / res.invalidNonBlack : 0.0) << "% of "
              << res.invalidNonBlack << " invalid non-black px\n";
    std::cout << "  textureless pixels that are invalid           : "
              << (res.texturelessCount ? 100.0 * res.invalidAndTextureless / res.texturelessCount : 0.0) << "%\n";

    std::cout << "\n  verdict         : "
              << (res.pass ? "PASS (dense disparity agrees with geometry)"
                            : "CHECK (disparity disagrees with sparse matches / low coverage)")
              << "\n";
    std::cout << std::defaultfloat;
}

RectificationRes Evaluator::evaluateRectification(const std::vector<cv::Point2f> &inL,
                                                   const std::vector<cv::Point2f> &inR,
                                                   const cv::Mat &K,
                                                   const cv::Mat &R1, const cv::Mat &P1,
                                                   const cv::Mat &R2, const cv::Mat &P2)
{
    RectificationRes res;

    cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Point2f> rL, rR;
    cv::undistortPoints(inL, rL, K, dist, R1, P1);
    cv::undistortPoints(inR, rR, K, dist, R2, P2);

    std::vector<double> errs;
    errs.reserve(rL.size());
    double sumErr = 0.0, maxErr = 0.0;
    for (size_t i = 0; i < rL.size(); ++i)
    {
        double e = std::abs(rL[i].y - rR[i].y);
        errs.push_back(e);
        sumErr += e;
        maxErr = std::max(maxErr, e);
    }

    res.correspondences = errs.size();
    res.meanErr = errs.empty() ? 0.0 : sumErr / errs.size();
    res.maxErr = maxErr;

    std::vector<double> sorted = errs;
    std::sort(sorted.begin(), sorted.end());
    res.medianErr = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];

    for (double e : errs)
        if (e <= 1.0)
            ++res.within1px;

    // Sub-pixel / single-pixel mean residual indicates a correct calibrated rectification.
    res.pass = res.meanErr < 1.0;
    return res;
}

void Evaluator::printRectification(const RectificationRes &res)
{
    std::cout << "\n--- Rectification Vertical-Alignment Error (pixels) ---\n";
    std::cout << "  correspondences : " << res.correspondences << "\n";
    std::cout << "  mean   |yL-yR|  : " << res.meanErr << "\n";
    std::cout << "  median |yL-yR|  : " << res.medianErr << "\n";
    std::cout << "  max    |yL-yR|  : " << res.maxErr << "\n";
    std::cout << "  within 1px      : " << res.within1px << " / " << res.correspondences
              << " (" << (res.correspondences ? 100.0 * res.within1px / res.correspondences : 0.0) << "%)\n";
    std::cout << "  verdict         : "
              << (res.pass ? "PASS (rectification row-aligned)" : "CHECK (residual too large)")
              << "\n";
}
