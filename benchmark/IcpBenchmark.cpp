// Benchmark: unweighted vs confidence-weighted ICP on real DTU stereo clouds.
//
// Two clouds are built with the full stereo pipeline (pairs (1,2) and (3,4)),
// placed in the DTU world frame via the calibrated poses, and normalized.
// Being calibrated into the same frame, they start co-registered; each trial
// applies a known random rigid perturbation to cloud 1 and asks each ICP
// variant to undo it, mirroring the production fusion procedure
// (coarse subsampled stage + finer stage).
//
// Ground truths per trial:
//   1. the applied perturbation itself -> rotation / translation recovery error
//   2. the DTU structured-light scan   -> chamfer accuracy (mm) of the aligned
//      cloud, an error measure independent of the ICP objective
//
// Variants: {point-to-point, point-to-plane} x {unweighted, weighted}.
// All variants of a trial see byte-identical inputs (same perturbation, same
// subsampled point sets). Results go to icp_benchmark.csv plus a summary table.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <opencv2/flann.hpp>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"

namespace {

using IcpUtils::applyRigid;
using IcpUtils::randomRigid;

// Mean / median nearest-neighbor distance (mm) from the (denormalized) cloud
// points to the DTU ground-truth scan.
void chamferToGT(const std::vector<Eigen::Vector3f> &pts,
                 const Eigen::Vector3f &mean0, float scale0,
                 cv::flann::Index &gtIndex,
                 double &meanMm, double &medianMm)
{
    const int n = (int)pts.size();
    cv::Mat query(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector3f w = pts[i] * scale0 + mean0;
        query.at<float>(i, 0) = w.x();
        query.at<float>(i, 1) = w.y();
        query.at<float>(i, 2) = w.z();
    }
    cv::Mat indices(n, 1, CV_32S), dists(n, 1, CV_32F);
    gtIndex.knnSearch(query, indices, dists, 1);

    std::vector<double> d(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        d[i] = std::sqrt((double)dists.at<float>(i, 0));
        sum += d[i];
    }
    std::sort(d.begin(), d.end());
    meanMm = sum / n;
    medianMm = d[n / 2];
}

struct Row
{
    double magDeg, magTrans;
    int trial;
    std::string variant;
    double rotErrDeg, transErrMm, chamferMeanMm, chamferMedianMm, seconds;
};

} // namespace

int main(int argc, char **argv)
{
    const std::string configPath = (argc > 1) ? argv[1] : "../config.yaml";
    PipelineConfig config;
    try
    {
        config = PipelineConfig::load(configPath);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    const int numTrials = 3;
    const size_t coarseSamples = 4000;
    const size_t fineSamples = 15000;
    const size_t targetSamples = 30000;
    const int coarseIter = 30, fineIter = 15;
    const float distThresh = 0.1f;

    std::mt19937 rng(42);
    DTULoader loader("../data/dtu/");

    // --- Build the two stereo clouds, placed in the DTU world frame ---
    std::vector<PointCloud> clouds;
    for (int i = 1; i <= 3; i += 2)
    {
        std::cout << "\n=== Building cloud from pair (" << i << ", " << i + 1 << ") ===\n";
        StereoPair pair = loader.loadPair(i, i + 1);
        cv::Mat K = loader.loadIntrinsicCV(i);
        CameraPose poseLeft = loader.loadCameraPose(i);
        CameraPose poseRight = loader.loadCameraPose(i + 1);

        PipelineResult res;
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config,
                                   poseLeft.t, poseRight.t))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            return 1;
        }
        PointCloud cloud = PlyUtils::buildPointCloud(res.denseDisparity, res.Q, res.P1r,
                                                     res.P2r, res.camToWorld, res.rectColor,
                                                     res.minDisp, res.globalConfidence,
                                                     config.triangulation,
                                                     res.disparityConfidence,
                                                     config.confidenceWeights());

        // rectified-left frame -> DTU world frame (see IcpUtils::transformCloudToWorld)
        IcpUtils::transformCloudToWorld(cloud, res.R1, poseLeft);

        std::cout << "Cloud: " << cloud.pts.size() << " points.\n";
        clouds.push_back(std::move(cloud));
    }
    if (clouds.size() < 2)
    {
        std::cerr << "Need two clouds.\n";
        return 1;
    }

    // --- Ground-truth structured-light scan + KD-tree ---
    std::cout << "\nLoading DTU ground-truth scan...\n";
    std::vector<cv::Point3f> gt = loader.loadPointCloud(1);
    if (gt.empty())
    {
        std::cerr << "Failed to load GT scan.\n";
        return 1;
    }
    cv::Mat gtMat((int)gt.size(), 3, CV_32F);
    for (int i = 0; i < (int)gt.size(); ++i)
    {
        gtMat.at<float>(i, 0) = gt[i].x;
        gtMat.at<float>(i, 1) = gt[i].y;
        gtMat.at<float>(i, 2) = gt[i].z;
    }
    std::cout << "Building KD-tree on " << gt.size() << " GT points...\n";
    cv::flann::Index gtIndex(gtMat, cv::flann::KDTreeIndexParams(4));

    // --- Normalize both clouds by cloud-0 statistics (as in fusion) ---
    auto [mean0, scale0] = PlyUtils::normalise(clouds[0]);
    for (auto &p : clouds[1].pts)
        p = (p - mean0) / scale0;
    std::cout << "Normalization: scale0 = " << scale0 << " mm per unit.\n";

    PointCloud targetSub = PlyUtils::subsample(clouds[0], targetSamples, rng);

    // Reference chamfer of the UNPERTURBED cloud 1 -- the floor any ICP variant
    // can at best return to.
    {
        PointCloud evalRef = PlyUtils::subsample(clouds[1], fineSamples, rng);
        double m, md;
        chamferToGT(evalRef.pts, mean0, scale0, gtIndex, m, md);
        std::cout << "\nChamfer floor (unperturbed cloud 1 vs GT): mean=" << m
                  << " mm, median=" << md << " mm\n";
    }

    struct Variant { const char *name; bool w; ICPMode m; };
    const std::vector<Variant> variants = {
        {"p2point_unweighted", false, ICPMode::PointToPoint},
        {"p2point_weighted",   true,  ICPMode::PointToPoint},
        {"p2plane_unweighted", false, ICPMode::PointToPlane},
        {"p2plane_weighted",   true,  ICPMode::PointToPlane}};

    const std::vector<std::pair<double, double>> magnitudes = {
        {2.0, 0.02}, {5.0, 0.05}, {10.0, 0.10}}; // (deg, normalized translation)

    std::vector<Row> rows;

    for (const auto &[magDeg, magTrans] : magnitudes)
    {
        for (int trial = 0; trial < numTrials; ++trial)
        {
            const Eigen::Matrix4f T_pert = randomRigid(magDeg, magTrans, rng);

            // identical subsamples for every variant of this trial
            PointCloud srcPert = clouds[1];
            applyRigid(srcPert, T_pert);
            PointCloud srcCoarse = PlyUtils::subsample(srcPert, coarseSamples, rng);
            PointCloud srcFine = PlyUtils::subsample(srcPert, fineSamples, rng);

            // no-ICP reference row: how bad is the perturbation by itself
            {
                double m, md;
                chamferToGT(srcFine.pts, mean0, scale0, gtIndex, m, md);
                rows.push_back({magDeg, magTrans, trial, "no_icp",
                                magDeg, T_pert.block<3, 1>(0, 3).norm() * scale0,
                                m, md, 0.0});
            }

            for (const auto &v : variants)
            {
                PointCloud coarseCopy = srcCoarse;
                CeresICPOptimizer icp;
                icp.setMode(v.m);
                icp.useWeights(v.w);
                icp.setMatchingMaxDistance(distThresh);
                icp.setVerbose(false);

                const auto t0 = std::chrono::steady_clock::now();
                icp.setNbOfIterations(coarseIter);
                Eigen::Matrix4f T_coarse = icp.estimatePose(coarseCopy, targetSub);
                PointCloud fineCopy = srcFine;
                applyRigid(fineCopy, T_coarse);
                icp.setNbOfIterations(fineIter);
                Eigen::Matrix4f T_fine = icp.estimatePose(fineCopy, targetSub);
                applyRigid(fineCopy, T_fine); // materialize aligned points for the chamfer measure
                const auto t1 = std::chrono::steady_clock::now();

                const Eigen::Matrix4f T_total = T_fine * T_coarse;
                const Eigen::Matrix4f E = T_total * T_pert; // identity if perfect
                const Eigen::Matrix3f R_err = E.block<3, 3>(0, 0);
                const double c = std::min(1.0, std::max(-1.0, (double)((R_err.trace() - 1.0f) / 2.0f)));

                Row row;
                row.magDeg = magDeg;
                row.magTrans = magTrans;
                row.trial = trial;
                row.variant = v.name;
                row.rotErrDeg = std::acos(c) * 180.0 / M_PI;
                row.transErrMm = E.block<3, 1>(0, 3).norm() * scale0;
                chamferToGT(fineCopy.pts, mean0, scale0, gtIndex,
                            row.chamferMeanMm, row.chamferMedianMm);
                row.seconds = std::chrono::duration<double>(t1 - t0).count();
                rows.push_back(row);

                std::cout << "  [" << magDeg << " deg, trial " << trial << "] "
                          << std::left << std::setw(20) << v.name << std::right
                          << " rotErr=" << std::fixed << std::setprecision(3) << row.rotErrDeg
                          << " deg, transErr=" << row.transErrMm
                          << " mm, chamfer(mean)=" << row.chamferMeanMm
                          << " mm, t=" << std::setprecision(1) << row.seconds << " s\n";
            }
        }
    }

    // --- CSV ---
    std::ofstream csv("icp_benchmark.csv");
    csv << "mag_deg,mag_trans_norm,trial,variant,rot_err_deg,trans_err_mm,"
           "chamfer_mean_mm,chamfer_median_mm,seconds\n";
    for (const auto &r : rows)
        csv << r.magDeg << "," << r.magTrans << "," << r.trial << "," << r.variant << ","
            << r.rotErrDeg << "," << r.transErrMm << "," << r.chamferMeanMm << ","
            << r.chamferMedianMm << "," << r.seconds << "\n";
    csv.close();

    // --- Summary: mean over trials per (magnitude, variant) ---
    std::cout << "\n===== SUMMARY (mean over " << numTrials << " trials) =====\n";
    std::cout << std::left << std::setw(10) << "mag" << std::setw(22) << "variant"
              << std::right << std::setw(14) << "rotErr[deg]" << std::setw(15)
              << "transErr[mm]" << std::setw(18) << "chamfer[mm]" << std::setw(10)
              << "t[s]" << "\n";
    std::vector<std::string> names = {"no_icp", "p2point_unweighted", "p2point_weighted",
                                      "p2plane_unweighted", "p2plane_weighted"};
    for (const auto &[magDeg, magTrans] : magnitudes)
    {
        for (const auto &name : names)
        {
            double sr = 0, st = 0, sc = 0, ss = 0;
            int cnt = 0;
            for (const auto &r : rows)
                if (r.magDeg == magDeg && r.variant == name)
                {
                    sr += r.rotErrDeg; st += r.transErrMm;
                    sc += r.chamferMeanMm; ss += r.seconds; ++cnt;
                }
            if (!cnt) continue;
            std::cout << std::left << std::fixed << std::setprecision(1)
                      << std::setw(10) << magDeg << std::setw(22) << name
                      << std::right << std::setprecision(3) << std::setw(14) << sr / cnt
                      << std::setw(15) << st / cnt << std::setw(18) << sc / cnt
                      << std::setprecision(1) << std::setw(10) << ss / cnt << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Per-trial rows written to icp_benchmark.csv\n";
    return 0;
}
