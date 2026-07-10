#pragma once

#include <vector>
#include <Eigen/Dense>
#include "PlyUtils.hpp"    // PointCloud
#include "MeshUtils.hpp"   // Mesh

/**
 * @struct IndicatorField
 * @brief The Poisson indicator function chi sampled on a regular voxel grid.
 *
 * Poisson reconstruction does not solve for the surface directly; it solves for
 * the indicator function chi of the solid the surface bounds. chi is (up to an
 * additive constant and sign) large on one side of the surface and small on the
 * other, transitioning across the surface. The surface itself is the level set
 * chi == isoValue.
 *
 * Values live on GRID NODES, not cell centres. A marching-cubes cell is therefore
 * the cube spanned by the 8 neighbouring nodes
 *   (x,y,z), (x+1,y,z), ... (x+1,y+1,z+1),
 * and the surface crosses any cell edge whose two endpoint values straddle
 * isoValue. Storage is x-fastest, then y, then z.
 */
struct IndicatorField
{
    std::vector<float> values;              ///< dims.x()*dims.y()*dims.z() node values (x fastest)
    Eigen::Vector3i    dims   = {0, 0, 0};  ///< node counts per axis
    Eigen::Vector3f    origin = {0, 0, 0};  ///< world position of node (0,0,0)
    float              voxelSize = 1.0f;    ///< node spacing in world units (uniform)
    float              isoValue  = 0.0f;    ///< level set that IS the surface (mean chi at samples)

    /// Flat index of node (x,y,z) into @ref values (x fastest, then y, then z).
    inline size_t index(int x, int y, int z) const
    {
        return (static_cast<size_t>(z) * dims.y() + y) * dims.x() + x;
    }

    inline bool inBounds(int x, int y, int z) const
    {
        return x >= 0 && y >= 0 && z >= 0 &&
               x < dims.x() && y < dims.y() && z < dims.z();
    }

    /// Indicator value at node (x,y,z). No bounds checking.
    inline float at(int x, int y, int z) const { return values[index(x, y, z)]; }

    /// World-space position of node (x,y,z).
    inline Eigen::Vector3f nodePos(int x, int y, int z) const
    {
        return origin + voxelSize * Eigen::Vector3f(
                   static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }
};

/**
 * @class PoissonReconstruction
 * @brief Screened Poisson surface reconstruction (Kazhdan et al. 2006 / 2013) on a
 *        uniform voxel grid.
 *
 * Given a noisy, oriented point cloud, the oriented samples are read as samples of
 * the gradient of the solid's indicator function. Splatting the (confidence-weighted)
 * normals into a vector field V and requiring grad(chi) = V in a least-squares sense
 * yields the Poisson equation
 *
 *      laplacian(chi) = div(V).
 *
 * We discretize chi on a regular grid, assemble the sparse system with a
 * finite-difference gradient operator plus a screening term that pins chi to the
 * data at the sample points (screened Poisson, and the term that also removes the
 * pure-Neumann null space), and solve it with a conjugate-gradient solver.
 *
 * This is the uniform-grid variant of the method. The original paper uses an
 * adaptive octree with B-spline bases and a multigrid solve, which concentrates
 * resolution near the surface; a uniform grid is simpler and self-contained (only
 * Eigen) at the cost of memory scaling with resolution^3.
 *
 * @note Poisson consumes ORIENTED normals -- they are the data. If the input cloud
 *       carries no normals, they are estimated (PCA over a local neighbourhood) and
 *       consistently oriented as a fallback; provided normals are strongly preferred,
 *       since wrong orientation is the dominant failure mode of the method.
 */
class PoissonReconstruction
{
public:
    /**
     * @struct Config
     * @brief Tunable parameters. Defaults target the DTU-scale (millimetre) clouds
     *        produced by this pipeline.
     */
    struct Config
    {
        int   resolution      = 64;     ///< voxel cells along the longest bounding-box axis (the resolution knob)
        float boundingScale   = 1.1f;   ///< expand the sample bounding box by this factor so the watertight surface closes inside the grid
        int   padding         = 2;      ///< extra node layers around the box (room for the Neumann boundary and closing surface)
        float screeningWeight = 4.0f;   ///< alpha: sample-fidelity vs smoothness (screened Poisson); also conditions the solve. 0 => unscreened + a tiny pin
        int   solverMaxIters  = 5000;   ///< conjugate-gradient iteration cap
        float solverTol       = 1e-6f;  ///< conjugate-gradient relative residual tolerance
        int   normalNeighbors = 18;     ///< k for the PCA normal-estimation fallback (only used when the cloud has no normals)
        bool  verbose         = true;   ///< print grid / solver / iso-value diagnostics
    };

    PoissonReconstruction();
    explicit PoissonReconstruction(Config config);

    /**
     * @brief Solves the (screened) Poisson system for the indicator function of the
     *        solid sampled by @p cloud.
     *
     * Per-point confidence weights (@ref PointCloud::weights), if present, modulate
     * each sample's contribution to both the vector field and the screening term, so
     * low-confidence stereo points pull the surface less. Returns an empty field
     * (dims == 0) if the cloud has too few usable oriented samples.
     */
    IndicatorField computeIndicator(const PointCloud& cloud) const;

    /**
     * @brief Extracts the isosurface mesh { chi == field.isoValue } via marching cubes
     *        (Lorensen & Cline 1987; standard 256-entry edge/triangle lookup tables).
     *
     * Each grid cell (the cube spanned by 8 neighbouring nodes) is classified by the
     * sign of chi - isoValue at its corners, the active edges are read from the LUT,
     * and a vertex is placed on each crossed edge by linear interpolation
     * (@ref interpolateEdge). Vertices are welded across cells (keyed by grid edge) so
     * the result is a proper indexed mesh, not a triangle soup. Output vertices are all
     * marked valid; width/height stay 0 (the mesh is unstructured).
     */
    Mesh extractMesh(const IndicatorField& field) const;

    /**
     * @brief Trilinearly samples the indicator field at an arbitrary world point.
     *        Points outside the grid are clamped to the boundary. Useful for the
     *        iso-value computation and for density-based trimming of the output mesh.
     */
    static float sampleAt(const IndicatorField& field, const Eigen::Vector3f& worldPos);

    /**
     * @brief Linear interpolation of the surface crossing along the edge (a -> b),
     *        placing the vertex where the field equals @p isoValue:
     *            p = a + (isoValue - fa) / (fb - fa) * (b - a).
     *        Falls back to the midpoint when the two endpoint values are equal.
     */
    static Eigen::Vector3f interpolateEdge(
        const Eigen::Vector3f& a, float fa,
        const Eigen::Vector3f& b, float fb,
        float isoValue);

    const Config& config() const { return m_config; }

private:
    Config m_config;
};
