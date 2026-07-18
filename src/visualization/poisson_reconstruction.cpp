#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>

#include "PlyUtils.hpp"
#include "MeshUtils.hpp"
#include "PoissonReconstruction.hpp"

// Self-test cloud: a noisy oriented sphere (points + outward normals, all valid).
// Exercises the "provided normals" path used by IcpFusion and gives a known radius
// to check the reconstructed mesh against.
static PointCloud makeNoisySphere(int n, float radius, const Eigen::Vector3f& center)
{
    PointCloud cloud;
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, radius * 0.02f);  // 2% radial noise
    for (int i = 0; i < n; ++i)
    {
        Eigen::Vector3f dir(gauss(rng), gauss(rng), gauss(rng));
        if (dir.norm() < 1e-6f) { dir = Eigen::Vector3f::UnitZ(); }
        dir.normalize();
        cloud.pts.push_back(center + (radius + noise(rng)) * dir);
        cloud.normals.push_back(dir);          // outward radial normal
        cloud.validNormal.push_back(true);
        cloud.weights.push_back(1.0f);
        cloud.colors.push_back(cv::Vec3b(180, 180, 180));
    }
    return cloud;
}

/**
 * Poisson surface reconstruction demo / verification.
 *
 * Usage: PoissonReconstruction [cloud.ply] [resolution]
 *
 * Loads a (noisy) point cloud, solves for its indicator function, and:
 *   1. writes poisson_indicator.ply -- the input points recoloured by the indicator
 *      value sampled at each point (blue = below iso / "outside", red = above iso /
 *      "inside"). A clean two-tone split that hugs the surface means the solve worked;
 *      this is verifiable WITHOUT marching cubes.
 *   2. calls extractMesh and writes poisson_mesh.ply (empty until marching cubes lands).
 */
int main(int argc, char** argv)
{
    const std::string cloudPath = (argc > 1) ? argv[1] : "pointcloud.ply";
    const int resolution        = (argc > 2) ? std::atoi(argv[2]) : 64;

    std::cout << "=== Poisson Surface Reconstruction ===\n";

    PointCloud cloud;
    if (cloudPath == "--sphere")
    {
        std::cout << "Input: synthetic noisy sphere (radius 50, 40k points), resolution: "
                  << resolution << "\n";
        cloud = makeNoisySphere(40000, 50.0f, Eigen::Vector3f(10.0f, -5.0f, 100.0f));
    }
    else
    {
        std::cout << "Cloud: " << cloudPath << ", resolution: " << resolution << "\n";
        cloud = PlyUtils::loadPLY(cloudPath);
    }
    if (cloud.pts.empty())
    {
        std::cerr << "ERROR: no points available from " << cloudPath << "\n";
        return -1;
    }

    PoissonReconstruction::Config cfg;
    cfg.resolution = resolution;
    PoissonReconstruction poisson(cfg);

    IndicatorField field = poisson.computeIndicator(cloud);
    if (field.values.empty())
    {
        std::cerr << "ERROR: indicator solve produced an empty field.\n";
        return -1;
    }

    // --- Diagnostic: recolour the input points by the indicator value at each point ---
    // Samples sit near the iso-surface, so scale the colour map to the *spread* of chi
    // at the points (2 sigma) rather than the global range -- otherwise everything is
    // white. Residual colour then shows which side of the fitted surface each point fell.
    std::vector<float> chiAt(cloud.pts.size());
    double mean = 0.0;
    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        chiAt[i] = PoissonReconstruction::sampleAt(field, cloud.pts[i]);
        mean += chiAt[i];
    }
    mean /= std::max<size_t>(1, cloud.pts.size());
    double var = 0.0;
    for (float v : chiAt) var += (v - mean) * (v - mean);
    float half = 2.0f * std::sqrt(static_cast<float>(var / std::max<size_t>(1, cloud.pts.size())));
    if (half < 1e-12f) half = 1.0f;

    PointCloud tinted = cloud;
    tinted.colors.resize(cloud.pts.size());
    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        float t = std::clamp((chiAt[i] - field.isoValue) / half, -1.0f, 1.0f);  // -1..+1 around iso
        // diverging blue (below iso) -> white (on surface) -> red (above iso); stored BGR
        cv::Vec3b col;
        if (t >= 0.0f) { col = cv::Vec3b((uchar)(255 * (1 - t)), (uchar)(255 * (1 - t)), 255); }
        else           { col = cv::Vec3b(255, (uchar)(255 * (1 + t)), (uchar)(255 * (1 + t))); }
        tinted.colors[i] = col;
    }
    PlyUtils::savePLY("poisson_indicator.ply", tinted);

    // --- Mesh extraction (marching cubes -- caller-provided) ---
    Mesh mesh = poisson.extractMesh(field);
    if (!mesh.faces.empty())
        MeshUtils::saveMeshPLY("poisson_mesh.ply", mesh);
    else
        std::cout << "No mesh written (indicator field produced no triangles). "
                     "Inspect poisson_indicator.ply to verify the indicator solve.\n";

    return 0;
}
