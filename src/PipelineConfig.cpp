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

template <typename T>
T readScalar(const cv::FileStorage &fs, const std::string &key, T def)
{
    cv::FileNode node = fs[key];
    if (node.empty())
        return def;
    T value{};
    node >> value;
    return value;
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

    std::string v = readKey(fs, "fundamental_matrix", "opencv");
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

    cfg.stereoConfidenceFilter = readScalar<int>(fs, "stereo_confidence_filter", 1) != 0;
    cfg.stereoLRMaxDiff = readScalar<float>(fs, "stereo_lr_max_diff", 1.5f);
    cfg.stereoPhotometricScale = readScalar<float>(fs, "stereo_photometric_scale", 25.0f);
    cfg.icpRobust = readScalar<int>(fs, "icp_robust", 1) != 0;
    cfg.icpReciprocal = readScalar<int>(fs, "icp_reciprocal", 1) != 0;
    cfg.icpTrimFraction = readScalar<float>(fs, "icp_trim_fraction", 0.80f);
    cfg.fusionVoxelSize = readScalar<float>(fs, "fusion_voxel_size", 0.01f);
    cfg.fusionOutlierFactor = readScalar<float>(fs, "fusion_outlier_factor", 1.5f);

    if (cfg.stereoLRMaxDiff <= 0.0f || cfg.stereoPhotometricScale <= 0.0f)
        throw std::runtime_error("[Config] Stereo confidence scales must be positive.");
    if (cfg.icpTrimFraction <= 0.0f || cfg.icpTrimFraction > 1.0f)
        throw std::runtime_error("[Config] icp_trim_fraction must be in (0,1].");
    if (cfg.fusionVoxelSize <= 0.0f || cfg.fusionOutlierFactor <= 0.0f)
        throw std::runtime_error("[Config] Fusion voxel parameters must be positive.");

    return cfg;
}

void PipelineConfig::print() const
{
    auto name = [](bool isOpenCV) { return isOpenCV ? "opencv" : "custom"; };

    std::cout << "[Config] Pipeline step backends:\n"
              << "  fundamental_matrix: "
              << (fundamental == FundamentalMethod::OpenCVRANSAC ? "opencv"
                  : fundamental == FundamentalMethod::CustomRANSAC ? "custom"
                  : fundamental == FundamentalMethod::CustomMAGSAC ? "custom_magsac"
                                                                   : "custom_prosac")
              << "\n"
              << "  rectification:      " << name(rectification == RectificationMethod::CalibratedOpenCV) << "\n"
              << "  disparity:          " << name(disparity == DisparityMethod::OpenCVSGBM) << "\n"
              << "  triangulation:      " << name(triangulation == TriangulationMethod::OpenCV) << "\n"
              << "  icp_mode:           " << (icpMode == ICPMode::PointToPoint ? "point_to_point" : "point_to_plane")
              << "\n  stereo_confidence: " << (stereoConfidenceFilter ? "on" : "off")
              << " (LR=" << stereoLRMaxDiff << " px, photo_scale=" << stereoPhotometricScale << ")"
              << "\n  icp_robust:         " << (icpRobust ? "on" : "off")
              << ", reciprocal=" << (icpReciprocal ? "on" : "off")
              << ", trim=" << icpTrimFraction
              << "\n  fusion_voxel_size:  " << fusionVoxelSize
              << ", outlier_factor=" << fusionOutlierFactor
              << "\n";
}
