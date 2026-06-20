#include "PlyUtils.hpp"
#include "Triangulation.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

void savePLY(const std::string& path, const std::vector<cv::Vec3f>& pts, const std::vector<cv::Vec3b>& colors)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Failed to open " << path << " for writing\n";
        return;
    }
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    for (size_t i = 0; i < pts.size(); ++i)
        f << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " "
          << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << "\n";
    std::cout << "Saved " << pts.size() << " points to " << path << "\n";
}

void buildAndSavePLY(const cv::Mat& dispFloat, const PipelineResult& res, int numDisp, const std::string& plyPath,
                     TriangulationMethod method)
{
    cv::Mat disp32f;
    if (dispFloat.type() == CV_32F)
        disp32f = dispFloat;
    else if (dispFloat.type() == CV_16S)
        dispFloat.convertTo(disp32f, CV_32F, 1.0 / 16.0);
    else
        dispFloat.convertTo(disp32f, CV_32F);

    cv::Mat points3D = reprojectDisparityTo3D(disp32f, res, method);
    if (points3D.empty()) return;

    const cv::Matx33d R = res.camToWorld.colRange(0, 3);
    const cv::Vec3d   t(res.camToWorld.at<double>(0, 3),
                        res.camToWorld.at<double>(1, 3),
                        res.camToWorld.at<double>(2, 3));

    std::vector<cv::Vec3f> pts;
    std::vector<cv::Vec3b> colors;
    const float zMax = 9000.0f;

    for (int y = 0; y < points3D.rows; ++y) {
        for (int x = 0; x < points3D.cols; ++x) {
            if (disp32f.at<float>(y, x) <= (float)res.minDisp) continue;

            cv::Vec3f p = points3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (p[2] <= 0.f || p[2] > zMax) continue;

            cv::Vec3d w = R * cv::Vec3d(p[0], p[1], p[2]) + t;
            pts.push_back(cv::Vec3f(w[0], w[1], w[2]));
            colors.push_back(res.rectColor.at<cv::Vec3b>(y, x));
        }
    }

    std::cout << "Point cloud generation completed. Staging " << pts.size() << " elements.\n";
    savePLY(plyPath, pts, colors);
}
