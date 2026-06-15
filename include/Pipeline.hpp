#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "DTULoader.hpp"

struct PipelineResult
{
    cv::Mat rectLeft, rectRight, rectColor;
    cv::Mat Q;            // 4x4 disparity-to-depth matrix
    cv::Mat camToWorld;   // 3x4 [R|t]: camera coords -> world frame
    int minDisp = 0;      // Dynamic search range start
    int numDisp = 16;     // Dynamic search range width (multiple of 16)
    std::vector<cv::Point2f> inPtsL, inPtsR;
    cv::Size imgSize;
};

cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2);
void triangulate_points(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D);
void savePLY(const std::string& path, const std::vector<cv::Vec3f>& pts, const std::vector<cv::Vec3b>& colors);
cv::Mat loadDTUProjection(const std::string& imgPath);

bool runPipeline(const std::string& pathLeft, const std::string& pathRight, PipelineResult& res);
void buildAndSavePLY(const cv::Mat& dispFloat, const PipelineResult& res, int numDisp, const std::string& plyPath);