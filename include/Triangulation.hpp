#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "Pipeline.hpp"   // PipelineResult, TriangulationMethod

// Linear (DLT) triangulation of a single correspondence u1<->u2 given the two
// 3x4 projection matrices p1, p2. Returns the 3D point in p1's frame.
cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2,
                      const cv::Vec2d &u1, const cv::Vec2d &u2);

// Triangulates a list of correspondences, appending results to pts3D.
void triangulate_points(const cv::Mat &p1, const cv::Mat &p2,
                        const std::vector<cv::Vec2d> &pts1,
                        const std::vector<cv::Vec2d> &pts2,
                        std::vector<cv::Vec3d> &pts3D);

// Dense depth: converts a rectified float disparity map into a CV_32FC3 image of
// 3D points (in the rectified left-camera frame), choosing the backend via method.
// Drop-in replacement for the output of cv::reprojectImageTo3D.
cv::Mat reprojectDisparityTo3D(const cv::Mat &disp32f,
                               const PipelineResult &res,
                               TriangulationMethod method);
