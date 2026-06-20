#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "DTULoader.hpp"

// Selects how a disparity map is turned into 3D points (see Triangulation.hpp).
enum class TriangulationMethod {
    OpenCV,   // cv::reprojectImageTo3D using the disparity-to-depth matrix Q
    Manual    // per-pixel linear (DLT) triangulation from the rectified projections
};

struct PipelineResult
{
    cv::Mat rectLeft, rectRight, rectColor;
    cv::Mat Q;            // 4x4 disparity-to-depth matrix
    cv::Mat P1r, P2r;     // 3x4 rectified projection matrices (left, right)
    cv::Mat camToWorld;   // 3x4 [R|t]: camera coords -> world frame
    int minDisp = 0;      // Dynamic search range start
    int numDisp = 16;     // Dynamic search range width (multiple of 16)
    std::vector<cv::Point2f> inPtsL, inPtsR;
    cv::Size imgSize;
};

cv::Mat loadDTUProjection(const std::string& imgPath);

bool runPipeline(const std::string& pathLeft, const std::string& pathRight, PipelineResult& res);