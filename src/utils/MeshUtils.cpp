#include "MeshUtils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/core/eigen.hpp>

bool MeshUtils::isTriangleValid(const Mesh& mesh, int i0, int i1, int i2, float edgeThreshold)
{
    if (!mesh.validVertex[i0] || !mesh.validVertex[i1] || !mesh.validVertex[i2])
        return false;

    const Eigen::Vector3f& v0 = mesh.vertices[i0];
    const Eigen::Vector3f& v1 = mesh.vertices[i1];
    const Eigen::Vector3f& v2 = mesh.vertices[i2];

    if ((v0 - v1).norm() >= edgeThreshold) return false;
    if ((v0 - v2).norm() >= edgeThreshold) return false;
    if ((v1 - v2).norm() >= edgeThreshold) return false;

    return true;
}

Mesh MeshUtils::buildMesh(
    const cv::Mat& disparity,
    const cv::Mat& Q,
    const cv::Mat& P1r,
    const cv::Mat& P2r,
    const cv::Mat& camToWorld,
    const cv::Mat& rectColor,
    int minDisp,
    TriangulationMethod method,
    float edgeThreshold)
{
    Mesh mesh;

    cv::Mat disp32f;
    if (disparity.type() == CV_32F)
        disp32f = disparity;
    else
        disparity.convertTo(disp32f, CV_32F);

    cv::Mat pts3D = Triangulation::reprojectDisparityTo3D(disp32f, Q, P1r, P2r, minDisp, method);
    if (pts3D.empty()) return mesh;

    const int width  = pts3D.cols;
    const int height = pts3D.rows;
    mesh.width  = width;
    mesh.height = height;

    const size_t nPix = static_cast<size_t>(width) * static_cast<size_t>(height);
    mesh.vertices.assign(nPix, Eigen::Vector3f::Zero());
    mesh.colors.assign(nPix, cv::Vec3b(0, 0, 0));
    mesh.validVertex.assign(nPix, false);

    const float zMax = 9000.0f;

    const cv::Matx33d R = camToWorld.colRange(0, 3);
    const cv::Vec3d t = camToWorld.col(3);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    const Eigen::Matrix3f R_f = R_eigen.cast<float>();
    const Eigen::Vector3f t_f((float)t[0], (float)t[1], (float)t[2]);

    // 1. dense per-pixel back-projection into world frame
    // Every grid slot gets a vertex ((0,0,0) if invalid) so idx/idx+1/idx+width
    // arithmetic in pass 2 stays aligned with the dense width*height grid
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;

            if (disp32f.at<float>(y, x) <= (float)minDisp) continue;

            cv::Vec3f p = pts3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (p[2] <= 0.0f || p[2] > zMax) continue;

            const Eigen::Vector3f pCam(p[0], p[1], p[2]);
            mesh.vertices[idx] = R_f * pCam + t_f;
            mesh.colors[idx] = rectColor.at<cv::Vec3b>(y, x);
            mesh.validVertex[idx] = true;
        }
    }

    // 2. grid-connectivity triangulation, cell by cell
    // Each cell has 4 corner vertices: idx, idx+1, idx+width, idx+width+1.
    // Split into 2 triangles, each kept only if all 3 vertices are valid
    // and all 3 edges are shorter than edgeThreshold.
    mesh.faces.reserve(nPix * 2);
    for (int row = 0; row < height - 1; ++row) {
        for (int col = 0; col < width - 1; ++col) {
            int idx = row * width + col;

            int iTopLeft     = idx;
            int iTopRight    = idx + 1;
            int iBottomLeft  = idx + width;
            int iBottomRight = idx + width + 1;

            // Triangle 1: top-left, bottom-left, top-right
            if (isTriangleValid(mesh, iTopLeft, iBottomLeft, iTopRight, edgeThreshold))
                mesh.faces.emplace_back(iTopLeft, iBottomLeft, iTopRight);

            // Triangle 2: top-right, bottom-left, bottom-right
            if (isTriangleValid(mesh, iTopRight, iBottomLeft, iBottomRight, edgeThreshold))
                mesh.faces.emplace_back(iTopRight, iBottomLeft, iBottomRight);
        }
    }

    std::cout << "Mesh build complete: " << mesh.vertices.size() << " vertices ("
              << std::count(mesh.validVertex.begin(), mesh.validVertex.end(), true)
              << " valid), " << mesh.faces.size() << " faces.\n";

    return mesh;
}

void MeshUtils::saveMeshPLY(const std::string& path, const Mesh& mesh)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Failed to open target file path: " << path << "\n";
        return;
    }

    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << mesh.vertices.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "element face " << mesh.faces.size() << "\n"
      << "property list uchar int vertex_indices\n"
      << "end_header\n";

    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        if (mesh.validVertex[i]) {
            f << mesh.vertices[i].x() << " " << mesh.vertices[i].y() << " " << mesh.vertices[i].z() << " "
              << (int)mesh.colors[i][2] << " " << (int)mesh.colors[i][1] << " " << (int)mesh.colors[i][0] << "\n";
        } else {
            f << "0 0 0 0 0 0\n";
        }
    }

    for (const auto& face : mesh.faces) {
        f << "3 " << face.x() << " " << face.y() << " " << face.z() << "\n";
    }

    std::cout << "Saved mesh: " << mesh.vertices.size() << " vertices, "
              << mesh.faces.size() << " faces to path location: " << path << "\n";
}
