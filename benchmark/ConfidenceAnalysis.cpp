// Confidence-ICP analysis, diagnostic core (Layers L0 + L1).
//
// This tool does NOT run ICP. It answers the two questions that gate the whole
// "does confidence-weighted ICP help?" investigation, and that the end-to-end
// benchmark (benchmark/IcpBenchmark.cpp) cannot separate:
//
//   L0 - Is the confidence informative?  For every reconstructed point it dumps
//        the confidence weight w and its three per-point factors alongside the
//        point's TRUE error (nearest-neighbour distance, in mm, to the DTU
//        structured-light ground-truth scan). Python then measures whether w
//        actually ranks points by accuracy (reliability diagram, Spearman, AUC).
//        If w is uncorrelated with error, weighting by it cannot help - and that
//        explains the null result in IcpBenchmark without any ICP being run.
//
//   L1 - Does the weight have leverage?  The same dump lets Python quantify the
//        per-point variance / dynamic range of w (a constant weight is inert for
//        a single registration) and the SPATIAL layout of high- vs low-weight
//        points (if they are co-located, reweighting cannot rotate a rigid fit).
//
// Clouds are built with the production stereo stack and the production pair
// selection (config.yaml), then placed in the DTU world frame via calibration so
// the per-point GT distance is meaningful. GT is used ONLY as an evaluation ruler
// here, never as pipeline input. The confidence weight is analysed PRE-cull (the
// full distribution, including the low-confidence tail the 10% cull would drop);
// the cull threshold is recorded per pair so Python can mark it.
//
// Outputs (into --out, default the working directory):
//   confidence_points.csv  one row per (sub-sampled) point
//   confidence_pairs.csv   one row per reconstructed pair (metadata)

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <opencv2/flann.hpp>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "IcpUtils.hpp"

namespace {

// Nearest-neighbour distance (mm) from each world-frame point to the GT scan.
std::vector<double> gtErrorMm(const std::vector<Eigen::Vector3f> &pts, cv::flann::Index &gtIndex)
{
    const int n = (int)pts.size();
    cv::Mat query(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        query.at<float>(i, 0) = pts[i].x();
        query.at<float>(i, 1) = pts[i].y();
        query.at<float>(i, 2) = pts[i].z();
    }
    cv::Mat indices(n, 1, CV_32S), dists(n, 1, CV_32F);
    gtIndex.knnSearch(query, indices, dists, 1);
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = std::sqrt((double)dists.at<float>(i, 0)); // flann returns squared L2
    return out;
}

// RMS radius (mm) of a cloud about its centroid -- a scene-scale used by Python
// to normalize spatial-leverage distances.
double rmsRadius(const std::vector<Eigen::Vector3f> &pts)
{
    if (pts.empty()) return 0.0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto &p : pts) mean += p.cast<double>();
    mean /= (double)pts.size();
    double acc = 0.0;
    for (const auto &p : pts) acc += (p.cast<double>() - mean).squaredNorm();
    return std::sqrt(acc / (double)pts.size());
}

// q-quantile of a copy of v (q in [0,1]).
float quantile(std::vector<float> v, float q)
{
    if (v.empty()) return 0.0f;
    q = std::clamp(q, 0.0f, 1.0f);
    const size_t k = std::min(v.size() - 1, (size_t)(q * v.size()));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

std::string cliValue(int argc, char **argv, const std::string &flag, const std::string &def)
{
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string configPath = (argc > 1 && argv[1][0] != '-') ? argv[1] : "../config.yaml";
    const std::string outDir = cliValue(argc, argv, "--out", ".");
    const size_t maxPointsPerPair = (size_t)std::stoul(cliValue(argc, argv, "--max-points", "50000"));
    const unsigned seed = (unsigned)std::stoul(cliValue(argc, argv, "--seed", "42"));
    // Compose the weight w from a subset of factors, overriding config for a one-command
    // A/B. 'config' honors config.yaml; 'full' = all factors; 'depth_only' = w = c_depth
    // (the fix the diagnostic core motivates). The raw per-factor columns are unaffected.
    const std::string weightMode = cliValue(argc, argv, "--weight-mode", "config");
    const float cullFraction = 0.10f; // matches IcpFusion's confidenceDiscardFraction

    PipelineConfig config;
    try { config = PipelineConfig::load(configPath); }
    catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
    config.print();

    ConfidenceWeightConfig weightCfg = config.confidenceWeights();
    if (weightMode == "full")            weightCfg = {true, true, true, true};
    else if (weightMode == "depth_only") weightCfg = {false, true, false, false};
    else if (weightMode != "config")
    { std::cerr << "Unknown --weight-mode '" << weightMode << "' (config|full|depth_only)\n"; return 1; }
    std::cout << "Weight composition (mode=" << weightMode << "): global="
              << weightCfg.useGlobal << " depth=" << weightCfg.useDepth
              << " edge=" << weightCfg.useEdge << " stereo=" << weightCfg.useStereo << "\n";

    // --- Production pair selection (mirror IcpFusion.cpp) ---
    std::vector<std::pair<int, int>> selectedPairs;
    if (config.icpPairMode == IcpPairMode::Consecutive)
    {
        for (int v = config.icpViewFirst; v < config.icpViewLast; v += 1)
            selectedPairs.emplace_back(v, v + 1);
        std::cout << "Pair source: sliding window over views " << config.icpViewFirst << ".."
                  << config.icpViewLast << " (" << selectedPairs.size() << " candidates).\n";
    }
    else
    {
        selectedPairs = {{1, 2}, {4, 5}, {7, 8}, {12, 13}, {18, 19}, {26, 27}, {32, 33}, {37, 38}};
        std::cout << "Pair source: curated selection (" << selectedPairs.size() << " pairs).\n";
    }

    DTULoader loader("../data/dtu/");

    // --- Ground-truth structured-light scan + KD-tree (world frame, mm) ---
    std::cout << "Loading DTU ground-truth scan...\n";
    std::vector<cv::Point3f> gt = loader.loadPointCloud(1);
    if (gt.empty()) { std::cerr << "Failed to load GT scan.\n"; return 1; }
    cv::Mat gtMat((int)gt.size(), 3, CV_32F);
    for (int i = 0; i < (int)gt.size(); ++i)
    {
        gtMat.at<float>(i, 0) = gt[i].x;
        gtMat.at<float>(i, 1) = gt[i].y;
        gtMat.at<float>(i, 2) = gt[i].z;
    }
    std::cout << "Building KD-tree on " << gt.size() << " GT points...\n";
    cv::flann::Index gtIndex(gtMat, cv::flann::KDTreeIndexParams(4));

    std::ofstream pts(outDir + "/confidence_points.csv");
    std::ofstream meta(outDir + "/confidence_pairs.csv");
    if (!pts.is_open() || !meta.is_open())
    {
        std::cerr << "Cannot open output CSVs in '" << outDir << "'.\n";
        return 1;
    }
    pts << "pair_id,left,right,wx,wy,wz,cam_depth,w,c_depth,c_edge,c_stereo,valid_normal,gt_err_mm\n";
    meta << "pair_id,left,right,global_confidence,n_points_total,n_points_dumped,"
            "cull_threshold_w,rms_radius_mm\n";
    pts << std::fixed << std::setprecision(6);

    std::mt19937 rng(seed);
    int pairId = 0;

    for (const auto &[a, b] : selectedPairs)
    {
        int leftView = a, rightView = b;
        CameraPose poseLeft, poseRight;
        if (!IcpUtils::orderPair(loader, a, b, leftView, rightView, poseLeft, poseRight))
        {
            std::cerr << "Skipping pair (" << a << "," << b << "): vertical baseline.\n";
            continue;
        }

        std::cout << "\n=== Pair (" << leftView << ", " << rightView << ") ===\n";
        StereoPair pair = loader.loadPair(leftView, rightView);
        cv::Mat K(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                K.at<double>(r, c) = poseLeft.K(r, c);

        PipelineResult res;
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config,
                                   poseLeft.t, poseRight.t))
        {
            std::cerr << "Pipeline failed for pair (" << leftView << "," << rightView << ").\n";
            continue;
        }
        if (std::abs(res.P2r.at<double>(0, 3)) < std::abs(res.P2r.at<double>(1, 3)))
        {
            std::cerr << "Pair rectified to a vertical baseline -- skipping.\n";
            continue;
        }

        PointConfidenceBreakdown bd;
        PointCloud cloud = PlyUtils::buildPointCloud(
            res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld, res.rectColor,
            res.minDisp, res.globalConfidence, config.triangulation, res.disparityConfidence,
            weightCfg, &bd);

        const size_t n = cloud.pts.size();
        if (n < 1000 || bd.depthConf.size() != n)
        {
            std::cerr << "Pair produced " << n << " points (or breakdown size mismatch) -- skipping.\n";
            continue;
        }

        // Place in the DTU world frame (calibration) so GT distances are meaningful.
        // This is the world placement IcpFusion applies; it does not reorder points,
        // so the breakdown arrays stay aligned with cloud.pts.
        IcpUtils::transformCloudToWorld(cloud, res.R1, poseLeft);

        const std::vector<double> err = gtErrorMm(cloud.pts, gtIndex);
        const float cullThresh = quantile(cloud.weights, cullFraction);
        const double rms = rmsRadius(cloud.pts);

        // Deterministic sub-sample of indices for the dump (all analysis is
        // distribution-level, so 50k points/pair is ample and keeps the CSV small).
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        if (n > maxPointsPerPair)
        {
            std::shuffle(idx.begin(), idx.end(), rng);
            idx.resize(maxPointsPerPair);
        }

        for (size_t i : idx)
        {
            const auto &p = cloud.pts[i];
            pts << pairId << ',' << leftView << ',' << rightView << ','
                << p.x() << ',' << p.y() << ',' << p.z() << ','
                << bd.camDepth[i] << ',' << cloud.weights[i] << ','
                << bd.depthConf[i] << ',' << bd.edgeConf[i] << ',' << bd.stereoConf[i] << ','
                << (i < cloud.validNormal.size() && cloud.validNormal[i] ? 1 : 0) << ','
                << err[i] << '\n';
        }

        meta << pairId << ',' << leftView << ',' << rightView << ','
             << res.globalConfidence << ',' << n << ',' << idx.size() << ','
             << cullThresh << ',' << rms << '\n';

        std::cout << "  points=" << n << " (dumped " << idx.size() << "), global_conf="
                  << res.globalConfidence << ", cull@10%=" << cullThresh
                  << ", GT err median=" << err[err.size() / 2] << " mm (unsorted sample).\n";
        ++pairId;
    }

    pts.close();
    meta.close();

    if (pairId == 0) { std::cerr << "No pairs reconstructed.\n"; return 1; }
    std::cout << "\nWrote " << pairId << " pairs to " << outDir
              << "/confidence_points.csv and confidence_pairs.csv\n"
              << "Next: python3 scripts/analyze_confidence.py --in " << outDir << "\n";
    return 0;
}
