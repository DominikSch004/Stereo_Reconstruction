#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "SiftFlannMatcher.hpp"

// Serialize matched point pairs to binary file.
// Format: uint32_t count, then (float x_left, float y_left, float x_right, float y_right) × count
void serializeMatchPoints(const std::string& filename,
                          const std::vector<cv::Point2f>& ptsLeft,
                          const std::vector<cv::Point2f>& ptsRight);

// Deserialize matched point pairs from binary file.
// Returns true on success; false if file doesn't exist or is invalid.
bool deserializeMatchPoints(const std::string& filename,
                            std::vector<cv::Point2f>& ptsLeft,
                            std::vector<cv::Point2f>& ptsRight);
