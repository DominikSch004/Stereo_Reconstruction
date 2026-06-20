#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "Pipeline.hpp"   // PipelineResult, TriangulationMethod

// Writes a colored point cloud to an ASCII PLY file.
void savePLY(const std::string& path,
             const std::vector<cv::Vec3f>& pts,
             const std::vector<cv::Vec3b>& colors);

// Reprojects a disparity map to 3D (via the chosen triangulation backend),
// transforms the points into the world frame, filters them, and saves a PLY.
void buildAndSavePLY(const cv::Mat& dispFloat,
                     const PipelineResult& res,
                     int numDisp,
                     const std::string& plyPath,
                     TriangulationMethod method = TriangulationMethod::OpenCV);
