// Paired ablation of the confidence terms used by the production ICP stage.
//
// The benchmark reconstructs overlapping DTU stereo pairs once, places them in
// the calibrated world frame, and applies known rigid perturbations to each
// source cloud.  Every weight recipe sees the same target, source points,
// perturbation, coarse ICP result, correspondences, and solver settings.  Only
// the relative fine-stage source weights change.
//
// Outputs:
//   icp_weight_ablation.csv  one row per case/trial/perturbation/recipe
//   confidence_cues.csv      per-point cues and independent DTU GT error

// The plotting and statistical analysis live in scripts/analyze_icp_ablation.py.

// Methodological notes:
//   * A common unweighted coarse stage mirrors IcpFusion and puts every recipe
//     in the same basin of attraction.
//   * Recipe weights are mean-normalized.  Multiplying all residuals by a
//     constant cannot change the mathematical minimizer and must not affect a
//     result through numerical stopping tolerances.
//   * Transform recovery is the primary metric: T_fine T_coarse T_pert should
//     be identity.  Unweighted source-to-target distance and distance to the
//     structured-light scan are reported as independent secondary metrics.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/flann.hpp>

#include "DTULoader.hpp"
#include "ICP.hpp"
#include "IcpUtils.hpp"
#include "Pipeline.hpp"
#include "PipelineConfig.hpp"
#include "PlyUtils.hpp"

namespace {

struct ConfidenceCloud
{
    PointCloud cloud;
    PointConfidenceBreakdown confidence;
    float globalConfidence = 1.0f;
    int left = -1;
    int right = -1;
};

struct Perturbation
{
    double rotationDeg = 0.0;
    double translationMm = 0.0;
};

struct Recipe
{
    const char *name;
    bool global;
    bool depth;
    bool edge;
    bool stereo;
};

struct DistanceStats
{
    double median = 0.0;
    double p90 = 0.0;
};

std::string cliValue(int argc, char **argv, const std::string &flag,
                     const std::string &fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (flag == argv[i])
            return argv[i + 1];
    return fallback;
}

std::vector<Perturbation> parsePerturbations(const std::string &text)
{
    std::vector<Perturbation> result;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const size_t separator = token.find(':');
        if (separator == std::string::npos)
            throw std::runtime_error(
                "Each --perturbations entry must be ROT_DEG:TRANS_MM.");
        result.push_back({std::stod(token.substr(0, separator)),
                          std::stod(token.substr(separator + 1))});
    }
    if (result.empty())
        throw std::runtime_error("--perturbations must not be empty.");
    return result;
}

std::vector<size_t> sampleIndices(size_t count, size_t requested,
                                  std::mt19937 &rng)
{
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    if (indices.size() > requested)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(requested);
    }
    return indices;
}

ConfidenceCloud gather(const ConfidenceCloud &input, size_t requested,
                       std::mt19937 &rng)
{
    const std::vector<size_t> indices =
        sampleIndices(input.cloud.pts.size(), requested, rng);

    ConfidenceCloud output;
    output.globalConfidence = input.globalConfidence;
    output.left = input.left;
    output.right = input.right;
    output.cloud.pts.reserve(indices.size());
    output.cloud.colors.reserve(indices.size());
    output.cloud.weights.reserve(indices.size());
    output.cloud.normals.reserve(indices.size());
    output.cloud.validNormal.reserve(indices.size());
    output.confidence.cameraDepth.reserve(indices.size());
    output.confidence.depthConfidence.reserve(indices.size());
    output.confidence.edgeConfidence.reserve(indices.size());
    output.confidence.stereoConfidence.reserve(indices.size());

    for (size_t index : indices)
    {
        output.cloud.pts.push_back(input.cloud.pts[index]);
        if (index < input.cloud.colors.size())
            output.cloud.colors.push_back(input.cloud.colors[index]);
        if (index < input.cloud.weights.size())
            output.cloud.weights.push_back(input.cloud.weights[index]);
        if (index < input.cloud.normals.size())
            output.cloud.normals.push_back(input.cloud.normals[index]);
        if (index < input.cloud.validNormal.size())
            output.cloud.validNormal.push_back(input.cloud.validNormal[index]);

        output.confidence.cameraDepth.push_back(
            input.confidence.cameraDepth[index]);
        output.confidence.depthConfidence.push_back(
            input.confidence.depthConfidence[index]);
        output.confidence.edgeConfidence.push_back(
            input.confidence.edgeConfidence[index]);
        output.confidence.stereoConfidence.push_back(
            input.confidence.stereoConfidence[index]);
    }
    return output;
}

double setRecipeWeights(ConfidenceCloud &source, const Recipe &recipe)
{
    const size_t n = source.cloud.pts.size();
    source.cloud.weights.assign(n, 1.0f);
    double sum = 0.0;
    double sumSquares = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double weight = 1.0;
        if (recipe.global)
            weight *= source.globalConfidence;
        if (recipe.depth)
            weight *= source.confidence.depthConfidence[i];
        if (recipe.edge)
            weight *= source.confidence.edgeConfidence[i];
        if (recipe.stereo)
            weight *= source.confidence.stereoConfidence[i];
        if (!std::isfinite(weight) || weight < 0.0)
            weight = 0.0;
        source.cloud.weights[i] = static_cast<float>(weight);
        sum += weight;
    }

    // Remove a recipe's arbitrary global scale while preserving zero weights.
    const double mean = n > 0 ? sum / static_cast<double>(n) : 0.0;
    if (!(mean > 1e-12))
    {
        std::fill(source.cloud.weights.begin(), source.cloud.weights.end(), 1.0f);
        return static_cast<double>(n);
    }
    sum = 0.0;
    for (float &weight : source.cloud.weights)
    {
        weight = static_cast<float>(weight / mean);
        sum += weight;
        sumSquares += static_cast<double>(weight) * weight;
    }
    return sumSquares > 0.0 ? sum * sum / sumSquares : 0.0;
}

cv::Mat pointsToMat(const std::vector<Eigen::Vector3f> &points)
{
    cv::Mat matrix(static_cast<int>(points.size()), 3, CV_32F);
    for (int i = 0; i < matrix.rows; ++i)
    {
        matrix.at<float>(i, 0) = points[i].x();
        matrix.at<float>(i, 1) = points[i].y();
        matrix.at<float>(i, 2) = points[i].z();
    }
    return matrix;
}

std::vector<double> nearestDistances(const std::vector<Eigen::Vector3f> &query,
                                     cv::flann::Index &index,
                                     double distanceScale)
{
    if (query.empty())
        return {};
    cv::Mat queryMatrix = pointsToMat(query);
    cv::Mat indices(static_cast<int>(query.size()), 1, CV_32S);
    cv::Mat squaredDistances(static_cast<int>(query.size()), 1, CV_32F);
    index.knnSearch(queryMatrix, indices, squaredDistances, 1,
                    cv::flann::SearchParams(64));

    std::vector<double> distances(query.size());
    for (size_t i = 0; i < query.size(); ++i)
        distances[i] = std::sqrt(std::max(
            0.0, static_cast<double>(squaredDistances.at<float>(static_cast<int>(i), 0))))
            * distanceScale;
    return distances;
}

DistanceStats summarizeDistances(std::vector<double> distances)
{
    if (distances.empty())
        return {};
    std::sort(distances.begin(), distances.end());
    const size_t medianIndex = distances.size() / 2;
    const size_t p90Index = static_cast<size_t>(
        std::floor(0.9 * static_cast<double>(distances.size() - 1)));
    return {distances[medianIndex], distances[p90Index]};
}

void poseError(const Eigen::Matrix4f &estimated,
               const Eigen::Matrix4f &perturbation,
               float scaleMm, double &rotationDeg, double &translationMm)
{
    const Eigen::Matrix4f error = estimated * perturbation;
    const Eigen::Matrix3f rotation = error.block<3, 3>(0, 0);
    const double cosine = std::clamp(
        0.5 * static_cast<double>(rotation.trace() - 1.0f), -1.0, 1.0);
    rotationDeg = std::acos(cosine) * 180.0 / M_PI;
    translationMm = error.block<3, 1>(0, 3).norm() * scaleMm;
}

bool insideBox(const Eigen::Vector3f &point, const Eigen::Vector3f &lower,
               const Eigen::Vector3f &upper)
{
    return (point.array() >= lower.array()).all() &&
           (point.array() <= upper.array()).all();
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::string configPath =
            (argc > 1 && argv[1][0] != '-') ? argv[1] : "../config.yaml";
        const std::string outputDirectory =
            cliValue(argc, argv, "--out", "../results/icp_weight_ablation");
        const int numberOfCases =
            std::stoi(cliValue(argc, argv, "--cases", "3"));
        const int numberOfTrials =
            std::stoi(cliValue(argc, argv, "--trials", "5"));
        const size_t sourceSamples = static_cast<size_t>(
            std::stoul(cliValue(argc, argv, "--source-samples", "4000")));
        const size_t coarseSamples = static_cast<size_t>(
            std::stoul(cliValue(argc, argv, "--coarse-samples", "2000")));
        const size_t targetSamples = static_cast<size_t>(
            std::stoul(cliValue(argc, argv, "--target-samples", "12000")));
        const size_t cueSamples = static_cast<size_t>(
            std::stoul(cliValue(argc, argv, "--cue-samples", "20000")));
        const int viewFirst =
            std::stoi(cliValue(argc, argv, "--view-first", "6"));
        const unsigned randomSeed = static_cast<unsigned>(
            std::stoul(cliValue(argc, argv, "--seed", "2026")));
        const std::string icpModeName =
            cliValue(argc, argv, "--icp-mode", "config");
        const std::vector<Perturbation> perturbations = parsePerturbations(
            cliValue(argc, argv, "--perturbations", "1:5,3:15,5:25"));

        if (numberOfCases < 1 || numberOfTrials < 1 || sourceSamples < 6 ||
            coarseSamples < 6 || targetSamples < 6)
            throw std::runtime_error("Cases, trials, and sample counts must be positive.");

        PipelineConfig config = PipelineConfig::load(configPath);
        config.print();
        ICPMode icpMode = config.icpMode;
        std::string reportedIcpMode =
            icpMode == ICPMode::PointToPoint ? "point_to_point" : "point_to_plane";
        if (icpModeName == "point_to_point")
        {
            icpMode = ICPMode::PointToPoint;
            reportedIcpMode = icpModeName;
        }
        else if (icpModeName == "point_to_plane")
        {
            icpMode = ICPMode::PointToPlane;
            reportedIcpMode = icpModeName;
        }
        else if (icpModeName != "config")
        {
            throw std::runtime_error(
                "--icp-mode must be config, point_to_point, or point_to_plane.");
        }
        std::cout << "[Ablation] ICP residual: " << reportedIcpMode << '\n';
        if (!config.stereoConfidenceFilter)
            std::cerr << "WARNING: stereo_confidence_filter is off; c_stereo will be constant.\n";

        std::filesystem::create_directories(outputDirectory);
        std::ofstream csv(outputDirectory + "/icp_weight_ablation.csv");
        std::ofstream cueCsv(outputDirectory + "/confidence_cues.csv");
        if (!csv || !cueCsv)
            throw std::runtime_error("Could not create result CSVs in " + outputDirectory);

        csv << "case,target_pair,source_pair,icp_mode,trial,perturb_deg,perturb_trans_mm,"
               "variant,rot_err_deg,trans_err_mm,coarse_rot_err_deg,"
               "coarse_trans_err_mm,alignment_median_mm,alignment_p90_mm,"
               "gt_median_mm,gt_p90_mm,matches,weight_ess,seconds\n";
        cueCsv << "pair,left,right,c_global,c_depth,c_edge,c_stereo,w_full,"
                  "gt_error_mm,inside_gt_bbox\n";
        csv << std::fixed << std::setprecision(9);
        cueCsv << std::fixed << std::setprecision(9);

        DTULoader loader("../data/dtu/");

        std::cout << "\nLoading DTU structured-light reference...\n";
        const std::vector<cv::Point3f> gtPoints = loader.loadPointCloud(1);
        if (gtPoints.empty())
            throw std::runtime_error("Could not load the DTU ground-truth scan.");
        std::vector<Eigen::Vector3f> gtEigen;
        gtEigen.reserve(gtPoints.size());
        Eigen::Vector3f gtLower = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::max());
        Eigen::Vector3f gtUpper = Eigen::Vector3f::Constant(
            std::numeric_limits<float>::lowest());
        for (const cv::Point3f &point : gtPoints)
        {
            const Eigen::Vector3f p(point.x, point.y, point.z);
            gtEigen.push_back(p);
            gtLower = gtLower.cwiseMin(p);
            gtUpper = gtUpper.cwiseMax(p);
        }
        const Eigen::Vector3f bboxMargin = Eigen::Vector3f::Constant(10.0f);
        gtLower -= bboxMargin;
        gtUpper += bboxMargin;
        cv::Mat gtMatrix = pointsToMat(gtEigen);
        cv::flann::Index gtIndex(gtMatrix, cv::flann::KDTreeIndexParams(4));

        // Build one more cloud than registration cases. Consecutive stereo pairs
        // overlap by one camera, matching the production fusion setting.
        std::vector<ConfidenceCloud> clouds;
        std::mt19937 cueRng(randomSeed + 1);
        for (int view = viewFirst;
             static_cast<int>(clouds.size()) < numberOfCases + 1 && view < 49;
             ++view)
        {
            int left = view;
            int right = view + 1;
            CameraPose poseLeft;
            CameraPose poseRight;
            if (!IcpUtils::orderPair(loader, view, view + 1, left, right,
                                     poseLeft, poseRight))
                continue;

            std::cout << "\n=== Reconstructing pair (" << left << ',' << right << ") ===\n";
            const StereoPair pair = loader.loadPair(left, right);
            cv::Mat intrinsic(3, 3, CV_64F);
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    intrinsic.at<double>(row, column) = poseLeft.K(row, column);

            PipelineResult result;
            if (!Pipeline::runPipeline(pair.imageLeft, pair.imageRight, intrinsic,
                                       result, config, poseLeft.t, poseRight.t))
            {
                std::cerr << "Skipping failed pair (" << left << ',' << right << ").\n";
                continue;
            }

            ConfidenceCloud confidenceCloud;
            confidenceCloud.globalConfidence = result.globalConfidence;
            confidenceCloud.left = left;
            confidenceCloud.right = right;
            confidenceCloud.cloud = PlyUtils::buildPointCloud(
                result.denseDisparity, result.Q, result.P1r, result.P2r,
                result.camToWorld, result.rectColor, result.minDisp,
                result.globalConfidence, config.triangulation,
                result.disparityConfidence, ConfidenceWeightConfig(),
                &confidenceCloud.confidence);
            IcpUtils::transformCloudToWorld(
                confidenceCloud.cloud, result.R1, poseLeft);

            const size_t n = confidenceCloud.cloud.pts.size();
            if (n < std::max({sourceSamples, coarseSamples, size_t(1000)}) ||
                confidenceCloud.confidence.depthConfidence.size() != n)
            {
                std::cerr << "Skipping pair with only " << n
                          << " points or incomplete confidence data.\n";
                continue;
            }

            // Per-point cue quality is measured against the independent scan.
            const std::vector<size_t> cueIndices =
                sampleIndices(n, cueSamples, cueRng);
            std::vector<Eigen::Vector3f> cuePoints;
            cuePoints.reserve(cueIndices.size());
            for (size_t index : cueIndices)
                cuePoints.push_back(confidenceCloud.cloud.pts[index]);
            const std::vector<double> cueErrors =
                nearestDistances(cuePoints, gtIndex, 1.0);
            const std::string pairName =
                std::to_string(left) + "_" + std::to_string(right);
            for (size_t j = 0; j < cueIndices.size(); ++j)
            {
                const size_t index = cueIndices[j];
                const double depth = confidenceCloud.confidence.depthConfidence[index];
                const double edge = confidenceCloud.confidence.edgeConfidence[index];
                const double stereo = confidenceCloud.confidence.stereoConfidence[index];
                cueCsv << pairName << ',' << left << ',' << right << ','
                       << result.globalConfidence << ',' << depth << ',' << edge << ','
                       << stereo << ','
                       << result.globalConfidence * depth * edge * stereo << ','
                       << cueErrors[j] << ','
                       << (insideBox(cuePoints[j], gtLower, gtUpper) ? 1 : 0) << '\n';
            }

            std::cout << "Pair contains " << n << " points; dumped "
                      << cueIndices.size() << " cue samples.\n";
            clouds.push_back(std::move(confidenceCloud));
        }

        if (static_cast<int>(clouds.size()) < numberOfCases + 1)
            throw std::runtime_error("Not enough valid consecutive clouds for the requested cases.");
        cueCsv.close();

        // Use one common normalization for every registration case, exactly as
        // the fusion executable does.  scaleMm converts normalized translations
        // and NN distances back to report units.
        const auto normalization = PlyUtils::normalise(clouds.front().cloud);
        const Eigen::Vector3f mean = normalization.first;
        const float scaleMm = normalization.second;
        for (size_t cloudIndex = 1; cloudIndex < clouds.size(); ++cloudIndex)
            for (Eigen::Vector3f &point : clouds[cloudIndex].cloud.pts)
                point = (point - mean) / scaleMm;

        const std::vector<Recipe> recipes = {
            {"uniform", false, false, false, false},
            {"global_only", true, false, false, false},
            {"depth_only", false, true, false, false},
            {"edge_only", false, false, true, false},
            {"stereo_only", false, false, false, true},
            {"full", true, true, true, true},
            {"no_global", false, true, true, true},
            {"no_depth", true, false, true, true},
            {"no_edge", true, true, false, true},
            {"no_stereo", true, true, true, false}};

        std::mt19937 trialRng(randomSeed);
        size_t completed = 0;
        const size_t total = static_cast<size_t>(numberOfCases) * numberOfTrials *
                             perturbations.size() * recipes.size();

        for (int caseIndex = 0; caseIndex < numberOfCases; ++caseIndex)
        {
            const ConfidenceCloud &targetFull = clouds[caseIndex];
            const ConfidenceCloud &sourceFull = clouds[caseIndex + 1];
            PointCloud target = PlyUtils::subsample(
                targetFull.cloud, targetSamples, trialRng);
            cv::Mat targetMatrix = pointsToMat(target.pts);
            cv::flann::Index targetIndex(
                targetMatrix, cv::flann::KDTreeIndexParams(4));

            const std::string targetPair =
                std::to_string(targetFull.left) + "_" + std::to_string(targetFull.right);
            const std::string sourcePair =
                std::to_string(sourceFull.left) + "_" + std::to_string(sourceFull.right);

            for (int trial = 0; trial < numberOfTrials; ++trial)
            {
                for (const Perturbation &magnitude : perturbations)
                {
                    const Eigen::Matrix4f perturbation = IcpUtils::randomRigid(
                        magnitude.rotationDeg,
                        magnitude.translationMm / scaleMm, trialRng);

                    ConfidenceCloud coarseSource =
                        gather(sourceFull, coarseSamples, trialRng);
                    ConfidenceCloud fineSource =
                        gather(sourceFull, sourceSamples, trialRng);
                    IcpUtils::applyRigid(coarseSource.cloud, perturbation);
                    IcpUtils::applyRigid(fineSource.cloud, perturbation);

                    // One shared, unweighted coarse result for every recipe.
                    CeresICPOptimizer coarseIcp;
                    coarseIcp.setMode(icpMode);
                    coarseIcp.useWeights(false);
                    coarseIcp.setMatchingMaxDistance(0.45f);
                    coarseIcp.setNbOfIterations(25);
                    coarseIcp.setVerbose(false);
                    const Eigen::Matrix4f coarseTransform =
                        coarseIcp.estimatePose(coarseSource.cloud, target);
                    IcpUtils::applyRigid(fineSource.cloud, coarseTransform);

                    double coarseRotationError = 0.0;
                    double coarseTranslationError = 0.0;
                    poseError(coarseTransform, perturbation, scaleMm,
                              coarseRotationError, coarseTranslationError);

                    for (const Recipe &recipe : recipes)
                    {
                        ConfidenceCloud weightedSource = fineSource;
                        const double effectiveSampleSize =
                            setRecipeWeights(weightedSource, recipe);

                        CeresICPOptimizer fineIcp;
                        fineIcp.setMode(icpMode);
                        fineIcp.useWeights(true);
                        fineIcp.setMatchingMaxDistance(0.12f);
                        fineIcp.setNbOfIterations(20);
                        fineIcp.setVerbose(false);

                        const auto start = std::chrono::steady_clock::now();
                        const Eigen::Matrix4f fineTransform =
                            fineIcp.estimatePose(weightedSource.cloud, target);
                        const auto end = std::chrono::steady_clock::now();
                        IcpUtils::applyRigid(weightedSource.cloud, fineTransform);

                        const Eigen::Matrix4f totalTransform =
                            fineTransform * coarseTransform;
                        double rotationError = 0.0;
                        double translationError = 0.0;
                        poseError(totalTransform, perturbation, scaleMm,
                                  rotationError, translationError);

                        const DistanceStats alignment = summarizeDistances(
                            nearestDistances(weightedSource.cloud.pts,
                                             targetIndex, scaleMm));

                        std::vector<Eigen::Vector3f> worldPoints;
                        worldPoints.reserve(weightedSource.cloud.pts.size());
                        for (const Eigen::Vector3f &point : weightedSource.cloud.pts)
                            worldPoints.push_back(point * scaleMm + mean);
                        const DistanceStats gt = summarizeDistances(
                            nearestDistances(worldPoints, gtIndex, 1.0));

                        csv << caseIndex << ',' << targetPair << ',' << sourcePair << ','
                            << reportedIcpMode << ',' << trial << ',' << magnitude.rotationDeg << ','
                            << magnitude.translationMm << ',' << recipe.name << ','
                            << rotationError << ',' << translationError << ','
                            << coarseRotationError << ',' << coarseTranslationError << ','
                            << alignment.median << ',' << alignment.p90 << ','
                            << gt.median << ',' << gt.p90 << ','
                            << fineIcp.lastMatchCount() << ',' << effectiveSampleSize << ','
                            << std::chrono::duration<double>(end - start).count() << '\n';

                        ++completed;
                    }

                    std::cout << "[" << completed << '/' << total << "] case "
                              << caseIndex + 1 << '/' << numberOfCases << ", trial "
                              << trial + 1 << '/' << numberOfTrials << ", perturbation "
                              << magnitude.rotationDeg << " deg / "
                              << magnitude.translationMm << " mm\n";
                }
            }
        }

        csv.close();
        std::cout << "\nWrote " << completed << " paired ICP runs to "
                  << outputDirectory << "/icp_weight_ablation.csv\n"
                  << "Next: python3 ../scripts/analyze_icp_ablation.py --in "
                  << outputDirectory << "\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ConfidenceIcpAblation: " << error.what() << '\n';
        return 1;
    }
}
