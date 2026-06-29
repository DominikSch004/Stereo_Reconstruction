#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "Triangulation.hpp"

/**
 * @struct Mesh
 * @brief Container holding a dense per-pixel vertex grid plus a sparse triangle face list
 *
 * Mesh keeps one vertex per disparity-map pixel (invalid ones as dummy (0,0,0))
 * Faces are only emitted where all participating vertices are valid and all
 * triangle edges pass the discontinuity threshold
 */
struct Mesh {
    std::vector<Eigen::Vector3f> vertices;     // dense grid, size == width * height
    std::vector<cv::Vec3b>       colors;       // parallel to vertices
    std::vector<bool>            validVertex;  // parallel to vertices; false => dummy (0,0,0)
    std::vector<Eigen::Vector3i> faces;        // indices into vertices/colors
    int width  = 0;
    int height = 0;
};

/**
 * @class MeshUtils
 * @brief Builds a simple grid-connectivity triangle mesh from a disparity map,
 * and writes it out as a PLY with both vertex and face elements.
 *
 * Currently a simple implementating mirroring "exercise 1" approach:
 * each disparity-map pixel is back-projected to a 3D vertex,
 * and each 2x2 block of neighboring pixels is split into two triangles.
 * A triangle is only kept if all three of its vertices are valid and
 * all three of its edges are shorter than edgeThreshold
 * (avoids bridging across depth discontinuities / occlusion boundaries)
 */
class MeshUtils
{
public:
    /**
     * @brief Builds a dense-grid mesh from a disparity map.
     * @param disparity Input disparity map
     * @param Q 4x4 disparity-to-depth reprojection matrix
     * @param P1r 3x4 rectified left projection matrix.
     * @param P2r 3x4 rectified right projection matrix.
     * @param camToWorld 3x4 [R|t] camera-to-world rigid transform
     * @param rectColor Rectified left image, used to sample per-vertex color
     * @param minDisp Disparity threshold below which a pixel is treated as invalid
     * @param method Triangulation backend method
     * @param edgeThreshold Maximum allowed triangle edge length before a triangle is rejected
     *  as bridging a depth discontinuity
     * @return Mesh with a dense vertex grid and a filtered face list.
     */
    static Mesh buildMesh(
        const cv::Mat& disparity,
        const cv::Mat& Q,
        const cv::Mat& P1r,
        const cv::Mat& P2r,
        const cv::Mat& camToWorld,
        const cv::Mat& rectColor,
        int minDisp,
        TriangulationMethod method = TriangulationMethod::OpenCV,
        float edgeThreshold = 10.0f
    );

    /**
     * @brief Writes a Mesh to a PLY file with vertex and face elements.
     * Invalid vertices are written as (0,0,0) points so that face indices
     * keep matching the dense width*height vertex grid.
     */
    static void saveMeshPLY(
        const std::string& path,
        const Mesh& mesh
    );

private:
    /**
     * @brief Rejects a triangle if any vertex is invalid or any edge exceeds edgeThreshold.
     */
    static bool isTriangleValid(
        const Mesh& mesh,
        int i0, int i1, int i2,
        float edgeThreshold
    );
};
