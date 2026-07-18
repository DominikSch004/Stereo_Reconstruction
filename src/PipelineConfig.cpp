#include "PipelineConfig.hpp"
#include <opencv2/core/persistence.hpp>
#include <iostream>
#include <stdexcept>

namespace
{

    std::string readKey(const cv::FileStorage &fs, const std::string &key, const std::string &def)
    {
        cv::FileNode node = fs[key];
        if (node.empty())
        {
            std::cout << "[Config] Key '" << key << "' not set, using default '" << def << "'\n";
            return def;
        }
        return static_cast<std::string>(node);
    }

    int readIntKey(const cv::FileStorage &fs, const std::string &key, int def)
    {
        cv::FileNode node = fs[key];
        if (node.empty())
        {
            std::cout << "[Config] Key '" << key << "' not set, using default '" << def << "'\n";
            return def;
        }
        return static_cast<int>(node);
    }

    [[noreturn]] void invalidValue(const std::string &key, const std::string &value, const std::string &allowed)
    {
        throw std::runtime_error("[Config] Invalid value '" + value + "' for key '" + key + "'. Allowed: " + allowed);
    }

} // namespace

PipelineConfig PipelineConfig::load(const std::string &path)
{
    cv::FileStorage fs;
    try
    {
        if (!fs.open(path, cv::FileStorage::READ))
            throw std::runtime_error("[Config] Cannot open config file: " + path);
    }
    catch (const cv::Exception &e)
    {
        throw std::runtime_error("[Config] Failed to parse '" + path + "': " + e.msg);
    }

    PipelineConfig cfg;

    // Dataset fields
    cv::FileNode node = fs["dataset_id"];
    cfg.datasetId = node.empty() ? 1 : static_cast<int>(node);

    node = fs["image_left_id"];
    cfg.imageLeftId = node.empty() ? 1 : static_cast<int>(node);

    node = fs["image_right_id"];
    cfg.imageRightId = node.empty() ? 2 : static_cast<int>(node);

    node = fs["illumination_id"];
    cfg.illuminationId = node.empty() ? 3 : static_cast<int>(node);

    node = fs["ratio_threshold"];
    cfg.ratioThreshold = node.empty() ? 0.75f : static_cast<float>(node);

    // FLANN ratio
    if (cfg.ratioThreshold < 0.01f || cfg.ratioThreshold > 1.0f)
    {
        std::cout << "[Config] WARNING: ratio_threshold out of range. Defaulting to 0.75\n";
        cfg.ratioThreshold = 0.75f;
    }

    node = fs["processing_scale"];
    cfg.processingScale = node.empty() ? 0.5f : static_cast<float>(node);

    // 1.0 = full resolution, no upper limit beyond that makes sense here
    if (cfg.processingScale <= 0.0f || cfg.processingScale > 1.0f)
    {
        std::cout << "[Config] WARNING: processing_scale out of range (0, 1.0]. Defaulting to 0.5\n";
        cfg.processingScale = 0.5f;
    }

    std::string v = readKey(fs, "feature_detector", "sift");
    if (v == "sift")
        cfg.featureDetector = FeatureDetector::SIFT;
    else if (v == "orb")
        cfg.featureDetector = FeatureDetector::ORB;
    else
        invalidValue("feature_detector", v, "sift | orb");

    v = readKey(fs, "fundamental_matrix", "opencv");
    if (v == "opencv")
        cfg.fundamental = FundamentalMethod::OpenCVRANSAC;
    else if (v == "custom")
        cfg.fundamental = FundamentalMethod::CustomRANSAC;
    else if (v == "custom_magsac")
        cfg.fundamental = FundamentalMethod::CustomMAGSAC;
    else if (v == "custom_prosac")
        cfg.fundamental = FundamentalMethod::CustomPROSAC;
    else
        invalidValue("fundamental_matrix", v, "opencv | custom | custom_magsac | custom_prosac");

    v = readKey(fs, "rectification", "opencv");
    if (v == "opencv")
        cfg.rectification = RectificationMethod::CalibratedOpenCV;
    else if (v == "custom")
        cfg.rectification = RectificationMethod::CalibratedCustom;
    else
        invalidValue("rectification", v, "opencv | custom");

    v = readKey(fs, "disparity", "opencv");
    if (v == "opencv")
        cfg.disparity = DisparityMethod::OpenCVSGBM;
    else if (v == "custom")
        cfg.disparity = DisparityMethod::Custom;
    else
        invalidValue("disparity", v, "opencv | custom");

    v = readKey(fs, "triangulation", "opencv");
    if (v == "opencv")
        cfg.triangulation = TriangulationMethod::OpenCV;
    else if (v == "custom")
        cfg.triangulation = TriangulationMethod::Manual;
    else
        invalidValue("triangulation", v, "opencv | custom");

    v = readKey(fs, "icp_mode", "point_to_plane");
    if (v == "point_to_point")
        cfg.icpMode = ICPMode::PointToPoint;
    else if (v == "point_to_plane")
        cfg.icpMode = ICPMode::PointToPlane;
    else
        invalidValue("icp_mode", v, "point_to_point | point_to_plane");

    cfg.rngSeed = readIntKey(fs, "rng_seed", 42);
    if (cfg.rngSeed == -1)
    {
        std::random_device rd;
        cfg.rng = std::mt19937(rd());
    }
    else
    {
        cfg.rng = std::mt19937(cfg.rngSeed);
    }

    v = readKey(fs, "pose_refinement", "false");
    if (v == "true")
        cfg.refinePose = true;
    else if (v == "false")
        cfg.refinePose = false;
    else
        invalidValue("pose_refinement", v, "true | false");

    return cfg;
}

void PipelineConfig::print() const
{
    auto name = [](bool isOpenCV)
    { 
        return isOpenCV ? "opencv" : "custom";
    };

    std::cout << "\n Image pair config: \n"
              << " selected dataset:     scan" << datasetId << "\n"
              << "  image_pair:         (" << imageLeftId << ", " << imageRightId << ")\n"
              << "  illumination:       " << illuminationId << "\n";

    std::cout << "\n[Config] Pipeline step backends:\n"
              << "  feature_detector:   " << (featureDetector == FeatureDetector::SIFT ? "sift" : "orb") << "\n"
              << "  fundamental_matrix: "
              << (fundamental == FundamentalMethod::OpenCVRANSAC   ? "opencv"
                  : fundamental == FundamentalMethod::CustomRANSAC ? "custom"
                  : fundamental == FundamentalMethod::CustomMAGSAC ? "custom_magsac"
                                                                   : "custom_prosac") << "\n"
              << "  rectification:      " << name(rectification == RectificationMethod::CalibratedOpenCV) << "\n"
              << "  disparity:          " << name(disparity == DisparityMethod::OpenCVSGBM) << "\n"
              << "  triangulation:      " << name(triangulation == TriangulationMethod::OpenCV) << "\n"
              << "  icp_mode:           " << (icpMode == ICPMode::PointToPoint ? "point_to_point" : "point_to_plane") << "\n";

    std::cout << "\n[Config] Tunable parameters:\n"
              << "  pose_refinement:    " << (refinePose ? "true" : "false") << "\n"
              << "  processing_scale:   " << processingScale << "\n"
              << "  ratio_threshold:    " << ratioThreshold << "\n";
}
