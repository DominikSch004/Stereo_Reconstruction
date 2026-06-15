#pragma once
#include <vector>
#include <random>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "PairPipeline.hpp"

struct Cloud {
    std::vector<Eigen::Vector3f> pts;
    std::vector<cv::Vec3b>       colors;
};


class CloudUtils
{
public:
    static Cloud build(
        const PipelineResult& res,
        const cv::Mat& disparity);

    static void savePLY(
        const std::string& path,
        const Cloud& cloud);

    static Cloud subsample(
        const Cloud& cloud,
        size_t n,
        std::mt19937& rng);

    static std::pair<Eigen::Vector3f,float>
    normalise(Cloud& cloud);

    static void denormalise(
        Cloud& cloud,
        const Eigen::Vector3f& mean,
        float scale);
};