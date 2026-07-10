#include "PoissonReconstruction.hpp"

#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

#include <iostream>
#include <cmath>
#include <queue>
#include <array>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

namespace
{

/// One usable input sample: metric position, unit oriented normal, confidence weight.
struct Sample
{
    Eigen::Vector3f p;
    Eigen::Vector3f n;
    float           w;
};

/**
 * @brief Estimates an (unoriented) surface normal per point via PCA over its k nearest
 *        neighbours, and returns the k-NN adjacency so orientation can reuse it.
 *
 * The normal is the eigenvector of the smallest eigenvalue of the local covariance
 * matrix (the local least-squares tangent-plane normal, Hoppe et al. 1992). Sign is
 * arbitrary at this stage; @ref orientNormals fixes it.
 */
std::vector<Eigen::Vector3f> estimateNormalsPCA(
    const std::vector<Eigen::Vector3f>& pts,
    int k,
    std::vector<std::vector<int>>& neighborsOut)
{
    const int n = static_cast<int>(pts.size());
    std::vector<Eigen::Vector3f> normals(n, Eigen::Vector3f::UnitZ());
    neighborsOut.assign(n, {});
    if (n < 3) return normals;

    k = std::min(k, n - 1);

    // FLANN KD-tree over the points (same backend as NearestNeighborSearch).
    cv::Mat data(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        data.at<float>(i, 0) = pts[i].x();
        data.at<float>(i, 1) = pts[i].y();
        data.at<float>(i, 2) = pts[i].z();
    }
    cv::flann::Index index(data, cv::flann::KDTreeIndexParams(4));

    std::vector<int>   idx(k + 1);
    std::vector<float> dist(k + 1);
    std::vector<float> query(3);

    for (int i = 0; i < n; ++i)
    {
        query[0] = pts[i].x();
        query[1] = pts[i].y();
        query[2] = pts[i].z();
        index.knnSearch(query, idx, dist, k + 1, cv::flann::SearchParams(32));

        // Accumulate covariance of the neighbourhood (excluding self-matches).
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        std::vector<int>& nbrs = neighborsOut[i];
        nbrs.reserve(k);
        for (int j = 0; j < static_cast<int>(idx.size()); ++j)
        {
            int m = idx[j];
            if (m < 0 || m == i) continue;
            nbrs.push_back(m);
            mean += pts[m];
        }
        if (nbrs.size() < 3) continue;
        mean /= static_cast<float>(nbrs.size());

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (int m : nbrs)
        {
            Eigen::Vector3f d = pts[m] - mean;
            cov += d * d.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
        if (es.info() == Eigen::Success)
            normals[i] = es.eigenvectors().col(0).normalized();  // smallest eigenvalue first
    }

    return normals;
}

/**
 * @brief Consistently orients PCA normals by propagating sign agreement along a
 *        minimum-spanning-tree of the k-NN graph, edge cost 1 - |n_u . n_v|
 *        (Hoppe et al. 1992). Propagating across the flattest transitions first
 *        avoids flipping the sign across sharp folds. Absolute orientation of each
 *        connected component is arbitrary (seed forced to +Z); marching cubes
 *        extracts the same surface either way -- only LOCAL consistency matters.
 */
void orientNormals(
    const std::vector<Eigen::Vector3f>& pts,
    std::vector<Eigen::Vector3f>& normals,
    const std::vector<std::vector<int>>& neighbors)
{
    const int n = static_cast<int>(pts.size());
    std::vector<char> visited(n, 0);

    struct Edge
    {
        float cost;
        int   from;  // already-oriented node
        int   to;
        bool operator>(const Edge& o) const { return cost > o.cost; }
    };
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> pq;

    auto pushEdges = [&](int u)
    {
        for (int v : neighbors[u])
            if (!visited[v])
                pq.push({1.0f - std::abs(normals[u].dot(normals[v])), u, v});
    };

    for (int start = 0; start < n; ++start)
    {
        if (visited[start]) continue;

        // New connected component: force the seed to point roughly +Z.
        if (normals[start].z() < 0.0f) normals[start] = -normals[start];
        visited[start] = 1;
        pushEdges(start);

        while (!pq.empty())
        {
            Edge e = pq.top();
            pq.pop();
            if (visited[e.to]) continue;

            if (normals[e.from].dot(normals[e.to]) < 0.0f)
                normals[e.to] = -normals[e.to];
            visited[e.to] = 1;
            pushEdges(e.to);
        }
    }
}

/**
 * @brief Collects usable samples: metric points with a unit oriented normal and a
 *        (mean-normalized) confidence weight. Uses the cloud's own normals when
 *        present; otherwise estimates and orients them.
 */
std::vector<Sample> prepareSamples(const PointCloud& cloud, const PoissonReconstruction::Config& cfg)
{
    std::vector<Sample> samples;
    const size_t n = cloud.pts.size();
    if (n == 0) return samples;

    // Do the cloud's own normals cover enough points to trust them?
    size_t validNormals = 0;
    for (size_t i = 0; i < cloud.normals.size(); ++i)
    {
        bool flagged = (i < cloud.validNormal.size()) ? cloud.validNormal[i] : true;
        if (flagged && cloud.normals[i].squaredNorm() > 1e-12f) ++validNormals;
    }
    const bool haveNormals = (cloud.normals.size() == n) && (validNormals >= n / 2);

    std::vector<Eigen::Vector3f> normals;
    if (haveNormals)
    {
        if (cfg.verbose)
            std::cout << "[Poisson] Using " << validNormals << " provided oriented normals.\n";
        normals = cloud.normals;
    }
    else
    {
        if (cfg.verbose)
            std::cout << "[Poisson] Cloud lacks oriented normals -- estimating via PCA (k="
                      << cfg.normalNeighbors << ") + MST orientation.\n";
        std::vector<std::vector<int>> adjacency;
        normals = estimateNormalsPCA(cloud.pts, cfg.normalNeighbors, adjacency);
        orientNormals(cloud.pts, normals, adjacency);
    }

    samples.reserve(n);
    double weightSum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        // Drop points whose normal is unusable (zero / non-finite).
        if (i >= normals.size()) break;
        if (haveNormals && i < cloud.validNormal.size() && !cloud.validNormal[i]) continue;
        const Eigen::Vector3f& nrm = normals[i];
        if (!nrm.allFinite() || nrm.squaredNorm() < 1e-12f) continue;

        float w = (i < cloud.weights.size()) ? cloud.weights[i] : 1.0f;
        if (!std::isfinite(w) || w <= 0.0f) w = 1.0f;

        samples.push_back({cloud.pts[i], nrm.normalized(), w});
        weightSum += w;
    }

    // Normalize weights to mean 1 so the screening term is scale-independent of the
    // absolute confidence magnitudes.
    if (!samples.empty() && weightSum > 0.0)
    {
        float meanW = static_cast<float>(weightSum / samples.size());
        if (meanW > 1e-12f)
            for (Sample& s : samples) s.w /= meanW;
    }

    return samples;
}

} // namespace

PoissonReconstruction::PoissonReconstruction()
    : m_config(Config{})
{
}

PoissonReconstruction::PoissonReconstruction(Config config)
    : m_config(config)
{
}

IndicatorField PoissonReconstruction::computeIndicator(const PointCloud& cloud) const
{
    IndicatorField field;

    std::vector<Sample> samples = prepareSamples(cloud, m_config);
    if (samples.size() < 10)
    {
        std::cerr << "[Poisson] ERROR: only " << samples.size()
                  << " usable oriented samples -- cannot reconstruct.\n";
        return field;
    }

    // ---- 1. Grid from the (expanded, padded) sample bounding box ----
    Eigen::Vector3f mn = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    for (const Sample& s : samples)
    {
        mn = mn.cwiseMin(s.p);
        mx = mx.cwiseMax(s.p);
    }

    const Eigen::Vector3f center = 0.5f * (mn + mx);
    const Eigen::Vector3f extent = (mx - mn).cwiseMax(1e-6f) * m_config.boundingScale;
    const float h = extent.maxCoeff() / static_cast<float>(std::max(1, m_config.resolution));
    if (!(h > 0.0f))
    {
        std::cerr << "[Poisson] ERROR: degenerate bounding box.\n";
        return field;
    }

    const int pad = std::max(0, m_config.padding);
    Eigen::Vector3i dims;
    for (int d = 0; d < 3; ++d)
    {
        int cells = static_cast<int>(std::ceil(extent[d] / h));
        dims[d] = cells + 1 + 2 * pad;   // node count along axis d
    }
    const Eigen::Vector3f origin = (center - 0.5f * extent) - Eigen::Vector3f::Constant(pad * h);

    field.dims      = dims;
    field.origin    = origin;
    field.voxelSize = h;

    const int nx = dims.x(), ny = dims.y(), nz = dims.z();
    const size_t nNodes = static_cast<size_t>(nx) * ny * nz;

    auto nodeIndex = [nx, ny](int x, int y, int z) -> size_t
    {
        return (static_cast<size_t>(z) * ny + y) * nx + x;
    };

    if (m_config.verbose)
        std::cout << "[Poisson] Grid " << nx << "x" << ny << "x" << nz << " = " << nNodes
                  << " nodes, voxel size " << h << " (world units).\n";

    // ---- 2. Splat (confidence-weighted) normals into a node-sampled vector field V ----
    // Trilinear splat: each sample deposits w * n into its 8 surrounding grid nodes.
    // Also record, per sample, the 8 node indices + trilinear weights (reused for the
    // screening term and the iso-value integral).
    std::vector<double> Vx(nNodes, 0.0), Vy(nNodes, 0.0), Vz(nNodes, 0.0);

    struct Stencil { std::array<size_t, 8> node; std::array<double, 8> wgt; };
    std::vector<Stencil> stencils(samples.size());

    const float invH = 1.0f / h;
    for (size_t si = 0; si < samples.size(); ++si)
    {
        const Sample& s = samples[si];
        Eigen::Vector3f gc = (s.p - origin) * invH;   // grid coordinates
        int i0 = std::clamp(static_cast<int>(std::floor(gc.x())), 0, nx - 2);
        int j0 = std::clamp(static_cast<int>(std::floor(gc.y())), 0, ny - 2);
        int k0 = std::clamp(static_cast<int>(std::floor(gc.z())), 0, nz - 2);
        float fx = std::clamp(gc.x() - i0, 0.0f, 1.0f);
        float fy = std::clamp(gc.y() - j0, 0.0f, 1.0f);
        float fz = std::clamp(gc.z() - k0, 0.0f, 1.0f);

        Stencil& st = stencils[si];
        int c = 0;
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx, ++c)
                {
                    double tw = (dx ? fx : 1.0f - fx) *
                                (dy ? fy : 1.0f - fy) *
                                (dz ? fz : 1.0f - fz);
                    size_t nd = nodeIndex(i0 + dx, j0 + dy, k0 + dz);
                    st.node[c] = nd;
                    st.wgt[c]  = tw;

                    double contrib = tw * s.w;
                    Vx[nd] += contrib * s.n.x();
                    Vy[nd] += contrib * s.n.y();
                    Vz[nd] += contrib * s.n.z();
                }
    }

    // ---- 3. Assemble the sparse screened-Poisson system (L + alpha*S) chi = b ----
    // L = G^T G  (finite-difference Laplacian, Neumann BC by construction),
    // b = G^T v  (the divergence source term), assembled edge by edge so signs and
    // symmetry are automatic. S = lumped screening mass at the sample stencils, which
    // both enforces data fidelity and removes the pure-Neumann constant null space.
    const double invH2 = static_cast<double>(invH) * invH;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nNodes * 7 + samples.size() * 8);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nNodes);

    auto addEdge = [&](size_t a, size_t b, double va, double vb)
    {
        // stiffness contribution of edge (a,b)
        triplets.emplace_back(a, a,  invH2);
        triplets.emplace_back(b, b,  invH2);
        triplets.emplace_back(a, b, -invH2);
        triplets.emplace_back(b, a, -invH2);
        // divergence RHS: edge value = mean of the two co-located node components
        double ve = 0.5 * (va + vb);
        rhs[a] -= static_cast<double>(invH) * ve;
        rhs[b] += static_cast<double>(invH) * ve;
    };

    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
            {
                size_t a = nodeIndex(x, y, z);
                if (x + 1 < nx) { size_t b = nodeIndex(x + 1, y, z); addEdge(a, b, Vx[a], Vx[b]); }
                if (y + 1 < ny) { size_t b = nodeIndex(x, y + 1, z); addEdge(a, b, Vy[a], Vy[b]); }
                if (z + 1 < nz) { size_t b = nodeIndex(x, y, z + 1); addEdge(a, b, Vz[a], Vz[b]); }
            }

    // Lumped screening: pull chi toward the data at every sample stencil node. Scaled by
    // 1/h^2 so screeningWeight is a dimensionless fidelity/smoothness ratio. A tiny floor
    // keeps the system SPD even when screening is disabled (screeningWeight == 0).
    double screen = static_cast<double>(std::max(m_config.screeningWeight, 0.0f)) * invH2;
    if (screen <= 0.0) screen = 1e-8 * invH2;
    for (const Stencil& st : stencils)
        for (int c = 0; c < 8; ++c)
            triplets.emplace_back(st.node[c], st.node[c], screen * st.wgt[c]);

    Eigen::SparseMatrix<double> A(nNodes, nNodes);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // ---- 4. Solve with conjugate gradient (SPD, sparse) ----
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    cg.setMaxIterations(m_config.solverMaxIters);
    cg.setTolerance(m_config.solverTol);
    cg.compute(A);
    if (cg.info() != Eigen::Success)
    {
        std::cerr << "[Poisson] ERROR: solver failed to factor the system.\n";
        return IndicatorField{};
    }
    Eigen::VectorXd chi = cg.solve(rhs);

    if (m_config.verbose)
        std::cout << "[Poisson] CG finished: " << cg.iterations() << " iters, residual "
                  << cg.error() << ".\n";

    // ---- 5. Iso-value = confidence-weighted mean of chi at the samples ----
    // (chi is defined only up to the constant the screening pinned, so the surface is
    // its level set at this data-derived value, not zero.)
    double isoNum = 0.0, isoDen = 0.0;
    for (size_t si = 0; si < samples.size(); ++si)
    {
        double val = 0.0;
        for (int c = 0; c < 8; ++c) val += stencils[si].wgt[c] * chi[stencils[si].node[c]];
        isoNum += samples[si].w * val;
        isoDen += samples[si].w;
    }
    field.isoValue = (isoDen > 0.0) ? static_cast<float>(isoNum / isoDen) : 0.0f;

    field.values.resize(nNodes);
    for (size_t i = 0; i < nNodes; ++i) field.values[i] = static_cast<float>(chi[i]);

    if (m_config.verbose)
    {
        double lo = chi.minCoeff(), hi = chi.maxCoeff();
        std::cout << "[Poisson] chi range [" << lo << ", " << hi
                  << "], iso-value " << field.isoValue << ".\n";

        // Correctness check: the whole method asks for grad(chi) ~ n at the samples.
        // Report the mean cosine between the reconstructed gradient and the input
        // normal, and the fraction that agree in sign. Near +/-1 and ~100% => the
        // indicator tracks the oriented normals (which sign is cosmetic).
        double cosSum = 0.0;
        size_t aligned = 0, counted = 0;
        for (const Sample& s : samples)
        {
            Eigen::Vector3f g(
                sampleAt(field, s.p + Eigen::Vector3f(h, 0, 0)) - sampleAt(field, s.p - Eigen::Vector3f(h, 0, 0)),
                sampleAt(field, s.p + Eigen::Vector3f(0, h, 0)) - sampleAt(field, s.p - Eigen::Vector3f(0, h, 0)),
                sampleAt(field, s.p + Eigen::Vector3f(0, 0, h)) - sampleAt(field, s.p - Eigen::Vector3f(0, 0, h)));
            float gn = g.norm();
            if (gn < 1e-12f) continue;
            double c = g.dot(s.n) / gn;
            cosSum += c;
            if (c > 0.0) ++aligned;
            ++counted;
        }
        if (counted > 0)
            std::cout << "[Poisson] gradient/normal alignment: mean cos = "
                      << (cosSum / counted) << ", sign-consistent = "
                      << (100.0 * aligned / counted) << "%.\n";
    }

    return field;
}

float PoissonReconstruction::sampleAt(const IndicatorField& field, const Eigen::Vector3f& worldPos)
{
    if (field.values.empty()) return 0.0f;

    Eigen::Vector3f gc = (worldPos - field.origin) / field.voxelSize;
    int nx = field.dims.x(), ny = field.dims.y(), nz = field.dims.z();
    int i0 = std::clamp(static_cast<int>(std::floor(gc.x())), 0, nx - 1);
    int j0 = std::clamp(static_cast<int>(std::floor(gc.y())), 0, ny - 1);
    int k0 = std::clamp(static_cast<int>(std::floor(gc.z())), 0, nz - 1);
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny - 1);
    int k1 = std::min(k0 + 1, nz - 1);
    float fx = std::clamp(gc.x() - i0, 0.0f, 1.0f);
    float fy = std::clamp(gc.y() - j0, 0.0f, 1.0f);
    float fz = std::clamp(gc.z() - k0, 0.0f, 1.0f);

    auto V = [&](int x, int y, int z) { return field.at(x, y, z); };
    float c00 = V(i0, j0, k0) * (1 - fx) + V(i1, j0, k0) * fx;
    float c10 = V(i0, j1, k0) * (1 - fx) + V(i1, j1, k0) * fx;
    float c01 = V(i0, j0, k1) * (1 - fx) + V(i1, j0, k1) * fx;
    float c11 = V(i0, j1, k1) * (1 - fx) + V(i1, j1, k1) * fx;
    float c0  = c00 * (1 - fy) + c10 * fy;
    float c1  = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
}

Eigen::Vector3f PoissonReconstruction::interpolateEdge(
    const Eigen::Vector3f& a, float fa,
    const Eigen::Vector3f& b, float fb,
    float isoValue)
{
    float denom = fb - fa;
    if (std::abs(denom) < 1e-12f) return 0.5f * (a + b);
    float t = (isoValue - fa) / denom;
    return a + t * (b - a);
}

namespace
{

// Standard marching-cubes lookup tables (Lorensen & Cline 1987; Paul Bourke's layout).
// edgeTable[i] has bit e set iff cell edge e is crossed for corner sign pattern i.
// triTable[i] lists the crossed edges to connect into triangles (terminated by -1).
const int edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

const int triTable[256][16] = {
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
    { 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
    { 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
    { 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
    { 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
    { 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
    { 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
    { 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
    { 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
    { 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
    { 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
    { 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
    { 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
    { 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
    { 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
    { 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
    { 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
    { 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
    { 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
    { 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
    { 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
    { 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
    { 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
    { 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
    { 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
    { 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
    { 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
    { 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
    { 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
    { 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
    { 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
    { 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
    { 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
    { 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
    { 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
    { 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
    { 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
    { 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
    { 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
    { 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
    { 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
    { 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
    { 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
    { 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
    { 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
    { 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
    { 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
    { 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
    { 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
    { 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
    { 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
    { 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
    { 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
    { 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
    { 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
    { 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
    { 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
    { 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
    { 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
    { 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
    { 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
    { 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
    { 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
    { 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
    { 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
    { 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
    { 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
    { 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
    { 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
    { 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
    { 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
    { 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
    { 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
    { 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
    { 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
    { 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
    { 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
    { 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
    { 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
    { 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
    { 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
    { 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
    { 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
    { 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
    { 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
    { 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
    { 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
    { 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
    { 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
    { 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
    { 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
    { 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
    { 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
    { 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
    { 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
    { 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
    { 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
    { 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
    { 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
    { 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
    { 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
    { 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
    { 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
    { 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
    { 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
    { 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
    { 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
    { 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
    { 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
    { 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
    { 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

// Corner offsets (dx,dy,dz) for the 8 cube corners, and the two corners each of the
// 12 cell edges connects -- matching the LUT's numbering. This mirrors the corner /
// edge convention of the reference ProcessVolumeCell/Polygonise exactly.
const int CORNER_DX[8] = {1, 0, 0, 1, 1, 0, 0, 1};
const int CORNER_DY[8] = {0, 0, 1, 1, 0, 0, 1, 1};
const int CORNER_DZ[8] = {0, 0, 0, 0, 1, 1, 1, 1};
const int EDGE_CORNERS[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

} // namespace

Mesh PoissonReconstruction::extractMesh(const IndicatorField& field) const
{
    Mesh mesh;
    const int nx = field.dims.x(), ny = field.dims.y(), nz = field.dims.z();
    if (field.values.empty() || nx < 2 || ny < 2 || nz < 2)
    {
        std::cerr << "[Poisson] extractMesh: empty or too-small indicator field.\n";
        return mesh;
    }

    const float iso = field.isoValue;
    const uint64_t nNodes = static_cast<uint64_t>(nx) * ny * nz;

    auto nodeKey = [nx, ny](int x, int y, int z) -> uint64_t
    {
        return (static_cast<uint64_t>(z) * ny + y) * nx + x;
    };

    // A surface vertex lives on a grid edge shared by up to four cells; weld duplicates
    // by keying on the (unordered) pair of node indices that edge connects.
    std::unordered_map<uint64_t, int> edgeVertex;
    edgeVertex.reserve(nNodes / 4 + 16);

    for (int z = 0; z < nz - 1; ++z)
        for (int y = 0; y < ny - 1; ++y)
            for (int x = 0; x < nx - 1; ++x)
            {
                int   cx[8], cy[8], cz[8];
                float val[8];
                int   cubeindex = 0;
                for (int c = 0; c < 8; ++c)
                {
                    cx[c] = x + CORNER_DX[c];
                    cy[c] = y + CORNER_DY[c];
                    cz[c] = z + CORNER_DZ[c];
                    val[c] = field.at(cx[c], cy[c], cz[c]);
                    if (val[c] < iso) cubeindex |= (1 << c);
                }

                const int edges = edgeTable[cubeindex];
                if (edges == 0) continue;  // cell entirely inside or outside

                int vertIdx[12];
                for (int e = 0; e < 12; ++e)
                {
                    if (!(edges & (1 << e))) continue;
                    const int a = EDGE_CORNERS[e][0];
                    const int b = EDGE_CORNERS[e][1];
                    uint64_t ka = nodeKey(cx[a], cy[a], cz[a]);
                    uint64_t kb = nodeKey(cx[b], cy[b], cz[b]);
                    uint64_t key = (ka < kb) ? ka * nNodes + kb : kb * nNodes + ka;

                    auto it = edgeVertex.find(key);
                    if (it != edgeVertex.end())
                    {
                        vertIdx[e] = it->second;
                    }
                    else
                    {
                        Eigen::Vector3f v = interpolateEdge(
                            field.nodePos(cx[a], cy[a], cz[a]), val[a],
                            field.nodePos(cx[b], cy[b], cz[b]), val[b], iso);
                        int idx = static_cast<int>(mesh.vertices.size());
                        mesh.vertices.push_back(v);
                        edgeVertex.emplace(key, idx);
                        vertIdx[e] = idx;
                    }
                }

                for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
                    mesh.faces.emplace_back(vertIdx[triTable[cubeindex][i]],
                                            vertIdx[triTable[cubeindex][i + 1]],
                                            vertIdx[triTable[cubeindex][i + 2]]);
            }

    // Finalize the attribute arrays the Mesh / PLY writer expects.
    mesh.colors.assign(mesh.vertices.size(), cv::Vec3b(200, 200, 200));
    mesh.validVertex.assign(mesh.vertices.size(), true);
    mesh.width  = 0;
    mesh.height = 0;

    if (m_config.verbose)
        std::cout << "[Poisson] Marching cubes: " << mesh.vertices.size() << " vertices, "
                  << mesh.faces.size() << " triangles (iso-value " << iso << ").\n";

    return mesh;
}
