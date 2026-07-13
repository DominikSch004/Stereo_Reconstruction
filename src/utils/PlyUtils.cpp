#include "PlyUtils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

bool PlyUtils::buildAndSavePLY(
    const std::string &path,
    const cv::Mat &disparity,
    const cv::Mat &Q,
    const cv::Mat &P1r,
    const cv::Mat &P2r,
    const cv::Mat &camToWorld,
    const cv::Mat &rectColor,
    int minDisp,
    float globalConfidence,
    TriangulationMethod method,
    const cv::Mat &disparityConfidence)
{
    std::cout << "Orchestrating point cloud export to: " << path << "\n";
    
    PointCloud cloud = buildPointCloud(disparity, Q, P1r, P2r, camToWorld, rectColor,
                                       minDisp, globalConfidence, method, disparityConfidence);
    
    if (cloud.pts.empty()) {
        std::cerr << "WARNING: Point cloud generated no points. Aborting file write sequence.\n";
        return false;
    }

    savePLY(path, cloud);
    return true;
}

PointCloud PlyUtils::buildPointCloud(
    const cv::Mat &disparity,
    const cv::Mat &Q,
    const cv::Mat &P1r,
    const cv::Mat &P2r,
    const cv::Mat &camToWorld,
    const cv::Mat &rectColor,
    int minDisp,
    float globalConfidence,
    TriangulationMethod method,
    const cv::Mat &disparityConfidence,
    const ConfidenceWeightConfig &weightCfg,
    PointConfidenceBreakdown *breakdown)
{
    cv::Mat disp32f;
    if (disparity.type() == CV_32F)
        disp32f = disparity;
    else
        disparity.convertTo(disp32f, CV_32F);

    cv::Mat pts3D = Triangulation::reprojectDisparityTo3D(disp32f, Q, P1r, P2r, minDisp, method);
    if (pts3D.empty())
        return PointCloud();
    if (breakdown) breakdown->clear();

    cv::Mat gradX, gradY;
    cv::Sobel(disp32f, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(disp32f, gradY, CV_32F, 0, 1, 3);

    cv::Mat gradMag;
    cv::magnitude(gradX, gradY, gradMag);

    PointCloud cloud;

    const cv::Matx33d R = camToWorld.colRange(0, 3);
    const cv::Vec3d t = camToWorld.col(3);

    const float zMax = 9000.0f;

    // extract f * B from P2r(0, 3) = -f*B
    double fB = 1.0;
    if (!P2r.empty() && P2r.rows >= 1 && P2r.cols >= 4)
        fB = std::abs(P2r.at<double>(0, 3));
    else
        // Fallback safety
        fB = 1000.0; 
    
    // Baseline sub-pixel precision. Pixel-specific reliability is supplied separately
    // by the left/right and photometric consistency map.
    const float sigma_d = 0.5f;

    // Normalize inverse depth variance by the median valid measurement variance.
    // This preserves the statistically useful relative precision while avoiding a
    // unit-dependent expression such as 1/(1 + variance_mm2).
    std::vector<float> variances;
    variances.reserve(static_cast<size_t>(pts3D.total() / 2));
    for (int y = 0; y < pts3D.rows; ++y)
        for (int x = 0; x < pts3D.cols; ++x)
        {
            if (disp32f.at<float>(y, x) <= static_cast<float>(minDisp)) continue;
            const cv::Vec3f p = pts3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[2]) || p[2] <= 0.0f || p[2] > zMax) continue;
            const float zSq = p[2] * p[2];
            variances.push_back((zSq * zSq) / static_cast<float>(fB * fB) *
                                (sigma_d * sigma_d));
        }
    float medianVariance = 1.0f;
    if (!variances.empty())
    {
        auto mid = variances.begin() + variances.size() / 2;
        std::nth_element(variances.begin(), mid, variances.end());
        medianVariance = std::max(*mid, 1e-12f);
    }

    const bool havePixelConfidence = !disparityConfidence.empty() &&
                                     disparityConfidence.type() == CV_32F &&
                                     disparityConfidence.size() == disp32f.size();
    for (int y = 0; y < pts3D.rows; ++y)
    {
        for (int x = 0; x < pts3D.cols; ++x)
        {
            if (disp32f.at<float>(y, x) <= (float)minDisp)
                continue;

            cv::Vec3f p = pts3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                continue;
            if (p[2] <= 0.0f || p[2] > zMax)
                continue;

            // depth variance = Z^4 / (f * B)^2 * sigma
            float zSq = p[2] * p[2];
            float variance = (zSq * zSq) / static_cast<float>(fB * fB) * (sigma_d * sigma_d);
            
            // Precision relative to a median-depth point, clamped to prevent a few
            // close points from dominating the normal equations.
            float depthConfidence = std::min(100.0f, medianVariance /
                                                       std::max(variance, medianVariance * 0.01f));

            float edgeGradient = gradMag.at<float>(y, x);

            // Exponential decay: if the disparity gradient is low, edgeWeight is ~1.0.
            // If the disparity jumps sharply (e.g., > 3 pixels edge gradient), edgeWeight drops toward 0.
            // The denominator (5.0f) controls the sensitivity to edges.
            float edgeWeight = std::exp(-edgeGradient / 5.0f);

            const float stereoConfidence = havePixelConfidence
                                         ? std::clamp(disparityConfidence.at<float>(y, x), 0.0f, 1.0f)
                                         : 1.0f;
            // Compose the weight from the enabled factors only; a disabled factor
            // contributes a neutral 1.0. The diagnostic core showed the full product
            // buries the informative c_depth under the inverted c_edge/c_stereo, so
            // the ablation needs to switch factors in and out. The breakdown below
            // still records the RAW per-factor values regardless of the toggles.
            const float finalWeight =
                (weightCfg.useGlobal ? globalConfidence : 1.0f) *
                (weightCfg.useDepth  ? depthConfidence  : 1.0f) *
                (weightCfg.useEdge   ? edgeWeight       : 1.0f) *
                (weightCfg.useStereo ? stereoConfidence : 1.0f);

            Eigen::Vector3f normalCam = Eigen::Vector3f::Zero();
            bool normalOk = false;
            if (y > 0 && y < pts3D.rows - 1 && x > 0 && x < pts3D.cols - 1) {
                cv::Vec3f pL = pts3D.at<cv::Vec3f>(y, x - 1);
                cv::Vec3f pR = pts3D.at<cv::Vec3f>(y, x + 1);
                cv::Vec3f pU = pts3D.at<cv::Vec3f>(y - 1, x);
                cv::Vec3f pD = pts3D.at<cv::Vec3f>(y + 1, x);

                const float d0 = disp32f.at<float>(y, x);
                const float dL = disp32f.at<float>(y, x - 1);
                const float dR = disp32f.at<float>(y, x + 1);
                const float dU = disp32f.at<float>(y - 1, x);
                const float dD = disp32f.at<float>(y + 1, x);
                const bool disparitiesValid = dL > minDisp && dR > minDisp &&
                                              dU > minDisp && dD > minDisp;
                const bool sameSurface = std::max({std::abs(dL - d0), std::abs(dR - d0),
                                                   std::abs(dU - d0), std::abs(dD - d0)}) <= 3.0f;
                bool neighborsFinite = disparitiesValid && sameSurface &&
                    std::isfinite(pL[0]) && std::isfinite(pL[1]) && std::isfinite(pL[2]) &&
                    std::isfinite(pR[0]) && std::isfinite(pR[1]) && std::isfinite(pR[2]) &&
                    std::isfinite(pU[0]) && std::isfinite(pU[1]) && std::isfinite(pU[2]) &&
                    std::isfinite(pD[0]) && std::isfinite(pD[1]) && std::isfinite(pD[2]);

                if (neighborsFinite) {
                    Eigen::Vector3f dx(pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]);
                    Eigen::Vector3f dy(pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]);
                    if (dx.norm() > 1e-5f && dy.norm() > 1e-5f) {
                        Eigen::Vector3f n = dx.cross(dy);
                        if (n.norm() > 1e-8f) {
                            normalCam = n.normalized();
                            // Convention: normal should point back toward the camera (negative Z in cam frame)
                            if (normalCam.z() > 0.0f) normalCam = -normalCam;
                            normalOk = true;
                        }
                    }
                }
            }

            // Project coordinate elements into the global tracking frame
            cv::Vec3d pointInCam = cv::Vec3d(p[0], p[1], p[2]);
            cv::Vec3d w = R * pointInCam + t;

            cloud.pts.push_back(Eigen::Vector3f((float)w[0], (float)w[1], (float)w[2]));
            cloud.colors.push_back(rectColor.at<cv::Vec3b>(y, x));
            cloud.weights.push_back(finalWeight);

            // Per-factor breakdown, pushed in lockstep with the point so the arrays
            // stay index-aligned for the confidence ablation. camDepth is the raw
            // camera-frame Z (mm), captured before the world projection below.
            if (breakdown)
            {
                breakdown->camDepth.push_back(p[2]);
                breakdown->depthConf.push_back(depthConfidence);
                breakdown->edgeConf.push_back(edgeWeight);
                breakdown->stereoConf.push_back(stereoConfidence);
            }

            if (normalOk) {
                Eigen::Matrix3d R_eigen;
                cv::cv2eigen(R, R_eigen);
                cloud.normals.push_back((R_eigen.cast<float>() * normalCam).normalized());
                cloud.validNormal.push_back(true);
            } else {
                cloud.normals.push_back(Eigen::Vector3f::Zero());
                cloud.validNormal.push_back(false);
            }
        }
    }

    std::cout << "Dense extraction sequence finalized. Points compiled: " << cloud.pts.size() << "\n";
    return cloud;
}

PointCloud PlyUtils::voxelDownsample(const PointCloud &cloud, float voxelSize)
{
    if (voxelSize <= 0.0f || cloud.pts.empty()) return cloud;

    struct Key { int x, y, z; bool operator==(const Key &o) const { return x == o.x && y == o.y && z == o.z; } };
    struct Hash { size_t operator()(const Key &k) const {
        size_t h = std::hash<int>{}(k.x);
        h ^= std::hash<int>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }};
    struct Accum {
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Vector3d n = Eigen::Vector3d::Zero();
        Eigen::Vector3d color = Eigen::Vector3d::Zero();
        double w = 0.0;
        size_t count = 0;
        bool anyNormal = false;
    };

    std::unordered_map<Key, Accum, Hash> voxels;
    voxels.reserve(cloud.pts.size());
    const float inv = 1.0f / voxelSize;
    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        const auto &p = cloud.pts[i];
        Key key{static_cast<int>(std::floor(p.x() * inv)),
                static_cast<int>(std::floor(p.y() * inv)),
                static_cast<int>(std::floor(p.z() * inv))};
        float wf = (i < cloud.weights.size() && std::isfinite(cloud.weights[i]))
                 ? std::max(cloud.weights[i], 1e-6f) : 1.0f;
        Accum &a = voxels[key];
        a.p += wf * p.cast<double>();
        a.w += wf;
        ++a.count;
        if (i < cloud.colors.size())
            a.color += wf * Eigen::Vector3d(cloud.colors[i][0], cloud.colors[i][1], cloud.colors[i][2]);
        if (i < cloud.normals.size() &&
            (i >= cloud.validNormal.size() || cloud.validNormal[i]) &&
            cloud.normals[i].squaredNorm() > 1e-12f)
        {
            Eigen::Vector3f n = cloud.normals[i];
            if (a.anyNormal && a.n.dot(n.cast<double>()) < 0.0) n = -n;
            a.n += wf * n.cast<double>();
            a.anyNormal = true;
        }
    }

    PointCloud out;
    out.pts.reserve(voxels.size()); out.colors.reserve(voxels.size());
    out.weights.reserve(voxels.size()); out.normals.reserve(voxels.size());
    out.validNormal.reserve(voxels.size());
    for (const auto &[key, a] : voxels)
    {
        (void)key;
        const double w = std::max(a.w, 1e-12);
        out.pts.push_back((a.p / w).cast<float>());
        Eigen::Vector3d c = a.color / w;
        out.colors.emplace_back(cv::saturate_cast<uchar>(c.x()),
                                cv::saturate_cast<uchar>(c.y()),
                                cv::saturate_cast<uchar>(c.z()));
        out.weights.push_back(static_cast<float>(a.w / std::max<size_t>(a.count, 1)));
        if (a.anyNormal && a.n.squaredNorm() > 1e-20)
        {
            out.normals.push_back(a.n.normalized().cast<float>());
            out.validNormal.push_back(true);
        }
        else
        {
            out.normals.push_back(Eigen::Vector3f::Zero());
            out.validNormal.push_back(false);
        }
    }
    return out;
}

void PlyUtils::savePLY(const std::string &path, const PointCloud &cloud)
{
    std::ofstream f(path);
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open target file path: " << path << "\n";
        return;
    }
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << cloud.pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";

    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        f << cloud.pts[i].x() << " " << cloud.pts[i].y() << " " << cloud.pts[i].z() << " "
          << (int)cloud.colors[i][2] << " " << (int)cloud.colors[i][1] << " " << (int)cloud.colors[i][0] << "\n";
    }
    std::cout << "Saved " << cloud.pts.size() << " elements to path location: " << path << "\n";
}

PointCloud PlyUtils::loadPLY(const std::string &path)
{
    PointCloud cloud;
    std::cout << "[PlyUtils] Loading PLY: " << path << "\n";

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Failed to open point cloud file: " << path << "\n";
        return cloud;
    }

    std::string line;
    int vertexCount = 0;
    bool isBinary = false;

    while (std::getline(file, line))
    {
        line.erase(line.find_last_not_of(" \n\r\t") + 1);

        if (line == "end_header")
            break;
        if (line.find("format binary") != std::string::npos)
            isBinary = true;
        if (line.find("element vertex") != std::string::npos)
        {
            sscanf(line.c_str(), "element vertex %d", &vertexCount);
        }
    }

    if (vertexCount == 0)
    {
        std::cerr << "ERROR: No vertices found in PLY header.\n";
        return cloud;
    }

    cloud.pts.reserve(vertexCount);

    if (isBinary)
    {
        for (int i = 0; i < vertexCount; ++i)
        {
            float xyz[3];
            file.read(reinterpret_cast<char *>(&xyz), 3 * sizeof(float));
            cloud.pts.push_back(Eigen::Vector3f(xyz[0], xyz[1], xyz[2]));
            // skip normals and rgb values
            file.seekg(15, std::ios::cur);
        }
    }
    else
    {
        for (int i = 0; i < vertexCount; ++i)
        {
            float x, y, z;
            file >> x >> y >> z;
            cloud.pts.push_back(Eigen::Vector3f(x, y, z));
            file.ignore(256, '\n');
        }
    }

    file.close();
    std::cout << "[PlyUtils] Successfully loaded " << cloud.pts.size() << " points.\n";

    return cloud;
}

PointCloud PlyUtils::subsample(const PointCloud &cloud, size_t n, std::mt19937 &rng)
{
    if (cloud.pts.size() <= n)
        return cloud;
    std::vector<size_t> idx(cloud.pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(n);

    PointCloud out;
    out.pts.reserve(n);
    out.colors.reserve(n);
    out.weights.reserve(n);
    out.normals.reserve(n);
    out.validNormal.reserve(n);
    for (size_t i : idx) { 
        out.pts.push_back(cloud.pts[i]); 
        out.colors.push_back(cloud.colors[i]); 
        out.weights.push_back(cloud.weights[i]);
        if (i < cloud.normals.size())
            out.normals.push_back(cloud.normals[i]);
        if (i < cloud.validNormal.size())
            out.validNormal.push_back(cloud.validNormal[i]);
    }
    return out;
}

std::pair<Eigen::Vector3f, float> PlyUtils::normalise(PointCloud &cloud)
{
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const auto &p : cloud.pts)
        mean += p;
    mean /= float(cloud.pts.size());

    float scale = 0.f;
    for (const auto &p : cloud.pts)
        scale += (p - mean).squaredNorm();
    scale = std::sqrt(scale / float(cloud.pts.size()));
    if (scale < 1e-6f)
        scale = 1.f;

    for (auto &p : cloud.pts)
        p = (p - mean) / scale;
    return {mean, scale};
}

void PlyUtils::denormalise(PointCloud &cloud, const Eigen::Vector3f &mean, float scale)
{
    for (auto &p : cloud.pts)
        p = p * scale + mean;
}
