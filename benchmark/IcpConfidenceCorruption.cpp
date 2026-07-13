// L4 - controlled-corruption mechanism test for confidence-weighted ICP.
//
// The diagnostic core (L0/L1) showed the real stereo confidence does not rank
// points by error. That leaves an unanswered question the end-to-end benchmark
// cannot separate: is confidence-weighting a dead MECHANISM, or a live mechanism
// fed a bad SIGNAL? L4 answers it by self-registration with a KNOWN perturbation
// and KNOWN injected corruption, comparing weight regimes:
//
//   uniform  - no weighting (baseline)
//   real     - the actual product confidence w
//   depth    - c_depth only (the L0 "fix")
//   oracle   - w = 0 on exactly the corrupted points (perfect confidence)
//
// The oracle vs uniform gap is the MECHANISM CEILING: the most weighting could
// ever buy. real/depth vs uniform is what our signal actually buys. If oracle
// helps a lot but real/depth track uniform, the mechanism is fine and the signal
// is the problem -- the definitive Track-C statement.
//
// Two robustness conditions, because the ICP already owns a robustness stack
// (reciprocal + adaptive gate + trimming + Cauchy):
//   stack=off - weight is the ONLY defense against outliers (pure mechanism)
//   stack=on  - production robustness on; does weighting add anything on top?
//
// Design choices for a clean isolation:
//   * self-registration: source = perturbed+corrupted copy of ONE cloud, target =
//     that same cloud. Ideal answer is exactly T_pert^{-1}; no surface mismatch.
//   * point-to-point residual: an outlier displacement is fully visible to the
//     objective (point-to-plane would hide tangential displacement).
//   * corruption sits within the matching gate, so it BIASES the solve rather
//     than being trivially gated out -- otherwise the gate, not the weight, wins.
//
// Output: icp_corruption.csv (one row per pair x seed x fraction x stack x regime),
// analysed by scripts/analyze_corruption.py.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include "DTULoader.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"

namespace {

using IcpUtils::applyRigid;
using IcpUtils::randomRigid;

std::string cliValue(int argc, char **argv, const std::string &flag, const std::string &def)
{
    for (int i = 1; i < argc - 1; ++i)
        if (flag == argv[i]) return argv[i + 1];
    return def;
}

// A clean base cloud plus the per-point c_depth factor kept index-aligned (needed
// for the depth-only regime, which PlyUtils::subsample would not carry).
struct BaseCloud
{
    PointCloud cloud;             // pts/colors/normals/validNormal/weights (full product w)
    std::vector<float> depthConf; // c_depth per point, aligned with cloud.pts
    std::pair<int, int> pair;
};

// Index-based gather so full w and c_depth stay aligned through subsampling.
BaseCloud subsampleBase(const PointCloud &full, const std::vector<float> &depth,
                        size_t n, std::mt19937 &rng)
{
    std::vector<size_t> idx(full.pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (full.pts.size() > n) { std::shuffle(idx.begin(), idx.end(), rng); idx.resize(n); }

    BaseCloud b;
    auto &c = b.cloud;
    c.pts.reserve(idx.size()); c.colors.reserve(idx.size()); c.weights.reserve(idx.size());
    c.normals.reserve(idx.size()); c.validNormal.reserve(idx.size());
    b.depthConf.reserve(idx.size());
    for (size_t i : idx)
    {
        c.pts.push_back(full.pts[i]);
        if (i < full.colors.size()) c.colors.push_back(full.colors[i]);
        c.weights.push_back(full.weights[i]);
        if (i < full.normals.size()) c.normals.push_back(full.normals[i]);
        if (i < full.validNormal.size()) c.validNormal.push_back(full.validNormal[i]);
        b.depthConf.push_back(i < depth.size() ? depth[i] : 1.0f);
    }
    return b;
}

// coarse-to-fine ICP, either with the production robustness stack or with it off.
Eigen::Matrix4f runIcp(const PointCloud &source, const PointCloud &target,
                       bool stackOn, bool useWeights, ICPMode mode, int &matchesOut)
{
    struct Level { float maxDist; unsigned iters; };
    // stack off: loose gates so corrupted points stay matched (weight is the only defense)
    // stack on : production-like tight gates
    // 3-level pyramid; the coarse gate must exceed the ~0.2 displacement a several-degree
    // perturbation induces on a unit-RMS cloud, or the fit never closes (a convergence-floor
    // artifact seen with a too-tight coarse gate). stack=off keeps loose gates so corrupted
    // points stay matched (weight is the only defense); stack=on tightens after the coarse pull.
    const std::vector<Level> levels = stackOn
        ? std::vector<Level>{{0.300f, 12}, {0.120f, 10}, {0.050f, 8}}
        : std::vector<Level>{{0.500f, 12}, {0.250f, 10}, {0.120f, 8}};

    Eigen::Matrix4f total = Eigen::Matrix4f::Identity();
    PointCloud work = source;
    int lastMatches = 0;
    for (const auto &lvl : levels)
    {
        CeresICPOptimizer icp;
        icp.setMode(mode);
        icp.setVerbose(false);
        icp.useWeights(useWeights);
        icp.setMatchingMaxDistance(lvl.maxDist);
        icp.setNbOfIterations(lvl.iters);
        if (stackOn)
        {
            icp.useReciprocalCorrespondences(true);
            icp.useAdaptiveDistanceGate(true);
            icp.setTrimFraction(0.80f);
            icp.setRobustLoss(ICPRobustLoss::Cauchy, std::max(0.5f * lvl.maxDist, 0.003f));
        }
        else
        {
            icp.useReciprocalCorrespondences(false);
            icp.useAdaptiveDistanceGate(false);
            icp.setTrimFraction(1.0f);
            icp.setRobustLoss(ICPRobustLoss::None);
        }
        Eigen::Matrix4f T = icp.estimatePose(work, target);
        applyRigid(work, T);
        total = T * total;
        lastMatches = icp.finalMetrics().matchCount;
    }
    matchesOut = lastMatches;
    return total;
}

void poseError(const Eigen::Matrix4f &Test, const Eigen::Matrix4f &Tpert,
               float scaleMm, double &rotDeg, double &transMm)
{
    const Eigen::Matrix4f E = Test * Tpert; // identity if perfect
    const Eigen::Matrix3f R = E.block<3, 3>(0, 0);
    const double c = std::clamp<double>((R.trace() - 1.0) * 0.5, -1.0, 1.0);
    rotDeg = std::acos(c) * 180.0 / M_PI;
    transMm = E.block<3, 1>(0, 3).norm() * scaleMm;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string configPath = (argc > 1 && argv[1][0] != '-') ? argv[1] : "../config.yaml";
    const std::string outDir = cliValue(argc, argv, "--out", ".");
    const int nPairs   = std::stoi(cliValue(argc, argv, "--pairs", "3"));
    const int nSeeds   = std::stoi(cliValue(argc, argv, "--seeds", "8"));
    const size_t sampleN = (size_t)std::stoul(cliValue(argc, argv, "--sample", "5000"));
    const double perturbDeg   = std::stod(cliValue(argc, argv, "--perturb-deg", "6.0"));
    const double perturbTrans = std::stod(cliValue(argc, argv, "--perturb-trans", "0.05"));
    const float outlierMag = std::stof(cliValue(argc, argv, "--outlier-mag", "0.03"));
    // bias    = systematic shift of corrupted points along a fixed direction (the failure
    //           mode weighting addresses: it BIASES the rigid fit; zero-mean noise does not)
    // outlier = random-direction fixed-magnitude blunder (zero-mean)
    // noise   = Gaussian per-axis (zero-mean)
    const std::string corruptMode = cliValue(argc, argv, "--mode", "bias"); // bias | outlier | noise
    const std::string icpModeStr = cliValue(argc, argv, "--icp-mode", "point_to_point");
    const ICPMode icpMode = (icpModeStr == "point_to_plane") ? ICPMode::PointToPlane
                                                             : ICPMode::PointToPoint;

    PipelineConfig config;
    try { config = PipelineConfig::load(configPath); }
    catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
    config.print();

    // --- production pair selection (mirror IcpFusion) ---
    std::vector<std::pair<int, int>> selectedPairs;
    if (config.icpPairMode == IcpPairMode::Consecutive)
        for (int v = config.icpViewFirst; v < config.icpViewLast; ++v)
            selectedPairs.emplace_back(v, v + 1);
    else
        selectedPairs = {{1, 2}, {4, 5}, {7, 8}, {12, 13}, {18, 19}, {26, 27}, {32, 33}, {37, 38}};

    DTULoader loader("../data/dtu/");

    // --- build the clean base clouds (self-normalized), keeping c_depth ---
    std::vector<BaseCloud> bases;
    std::mt19937 sampleRng(12345);
    for (const auto &[a, b] : selectedPairs)
    {
        if ((int)bases.size() >= nPairs) break;
        int leftView = a, rightView = b;
        CameraPose poseLeft, poseRight;
        if (!IcpUtils::orderPair(loader, a, b, leftView, rightView, poseLeft, poseRight)) continue;

        StereoPair pair = loader.loadPair(leftView, rightView);
        cv::Mat K(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) K.at<double>(r, c) = poseLeft.K(r, c);

        PipelineResult res;
        if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, K, res, config, poseLeft.t, poseRight.t))
            continue;
        if (std::abs(res.P2r.at<double>(0, 3)) < std::abs(res.P2r.at<double>(1, 3))) continue;

        PointConfidenceBreakdown bd;
        PointCloud cloud = PlyUtils::buildPointCloud(
            res.denseDisparity, res.Q, res.P1r, res.P2r, res.camToWorld, res.rectColor,
            res.minDisp, res.globalConfidence, config.triangulation, res.disparityConfidence,
            config.confidenceWeights(), &bd);
        if (cloud.pts.size() < sampleN * 2) continue;

        auto [mean, scale] = PlyUtils::normalise(cloud); // normalized units for gates/perturbation
        (void)mean;
        BaseCloud base = subsampleBase(cloud, bd.depthConf, sampleN, sampleRng);
        base.pair = {leftView, rightView};
        base.cloud.colors.clear(); // colors unused downstream; free memory
        std::cout << "Base cloud pair (" << leftView << "," << rightView << "): "
                  << base.cloud.pts.size() << " pts, scale=" << scale << " mm/unit "
                  << "(errors reported in normalized units).\n";
        bases.push_back(std::move(base));
    }
    if (bases.empty()) { std::cerr << "No base clouds built.\n"; return 1; }

    const std::vector<double> fractions = {0.0, 0.05, 0.10, 0.20, 0.40};
    struct Regime { const char *name; };
    const std::vector<std::string> regimes = {"uniform", "real", "depth", "oracle"};

    std::ofstream csv(outDir + "/icp_corruption.csv");
    if (!csv.is_open()) { std::cerr << "Cannot open " << outDir << "/icp_corruption.csv\n"; return 1; }
    csv << "pair,seed,corrupt_frac,mode,outlier_mag,stack,regime,icp_mode,"
           "rot_err_deg,trans_err_norm,matches,seconds\n";
    csv << std::fixed << std::setprecision(6);

    std::mt19937 rng(2026);
    int done = 0, total = (int)bases.size() * nSeeds * (int)fractions.size() * 2 * (int)regimes.size();

    for (size_t bi = 0; bi < bases.size(); ++bi)
    {
        const PointCloud &target = bases[bi].cloud;
        const std::vector<float> &depth = bases[bi].depthConf;
        const auto pr = bases[bi].pair;

        for (int seed = 0; seed < nSeeds; ++seed)
        {
            const Eigen::Matrix4f Tpert = randomRigid(perturbDeg, perturbTrans, rng);

            for (double f : fractions)
            {
                // deterministic corruption mask for (bi, seed, f)
                std::mt19937 crng((unsigned)(1000 * bi + 100 * seed + (int)std::round(f * 100)));
                const size_t n = target.pts.size();
                std::vector<char> corrupt(n, 0);
                const size_t k = (size_t)std::round(f * n);
                std::vector<size_t> idx(n); std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), crng);
                for (size_t i = 0; i < k; ++i) corrupt[idx[i]] = 1;

                // source = perturbed clean copy, then corrupt the masked points
                PointCloud src = target;
                applyRigid(src, Tpert);
                std::normal_distribution<float> gauss(0.0f, 1.0f);
                // one fixed shift direction per trial, used by the systematic bias mode
                Eigen::Vector3f biasDir(gauss(crng), gauss(crng), gauss(crng));
                biasDir = (biasDir.norm() < 1e-6f) ? Eigen::Vector3f(1, 0, 0) : biasDir.normalized();
                for (size_t i = 0; i < n; ++i)
                {
                    if (!corrupt[i]) continue;
                    Eigen::Vector3f d(gauss(crng), gauss(crng), gauss(crng));
                    if (corruptMode == "noise")
                        src.pts[i] += outlierMag * d;                 // zero-mean Gaussian
                    else if (corruptMode == "bias")
                        src.pts[i] += outlierMag * biasDir;           // systematic shift (biases the fit)
                    else
                    {
                        if (d.norm() < 1e-6f) d = Eigen::Vector3f(1, 0, 0);
                        src.pts[i] += outlierMag * d.normalized();    // random-direction blunder (zero-mean)
                    }
                }

                for (bool stackOn : {false, true})
                {
                    for (const std::string &regime : regimes)
                    {
                        PointCloud s = src;
                        bool useWeights = true;
                        if (regime == "uniform") useWeights = false;
                        else if (regime == "real") { /* s.weights already = full product w */ }
                        else if (regime == "depth") s.weights = depth;
                        else if (regime == "oracle")
                        {
                            s.weights.assign(n, 1.0f);
                            for (size_t i = 0; i < n; ++i) if (corrupt[i]) s.weights[i] = 0.0f;
                        }

                        const auto t0 = std::chrono::steady_clock::now();
                        int matches = 0;
                        Eigen::Matrix4f Test = runIcp(s, target, stackOn, useWeights, icpMode, matches);
                        const auto t1 = std::chrono::steady_clock::now();

                        double rotDeg, transMm;
                        poseError(Test, Tpert, 1.0f, rotDeg, transMm); // scale 1 -> normalized units

                        csv << pr.first << '_' << pr.second << ',' << seed << ',' << f << ','
                            << corruptMode << ',' << outlierMag << ','
                            << (stackOn ? "on" : "off") << ',' << regime << ',' << icpModeStr << ','
                            << rotDeg << ',' << transMm << ',' << matches << ','
                            << std::chrono::duration<double>(t1 - t0).count() << '\n';
                        ++done;
                    }
                }
            }
            std::cout << "  pair (" << pr.first << "," << pr.second << ") seed " << seed + 1
                      << "/" << nSeeds << "  [" << done << "/" << total << " runs]\n";
        }
    }
    csv.close();
    std::cout << "\nWrote " << done << " runs to " << outDir << "/icp_corruption.csv\n"
              << "Next: python3 scripts/analyze_corruption.py --in " << outDir << "\n";
    return 0;
}
