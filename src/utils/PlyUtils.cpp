#include "PlyUtils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

bool PlyUtils::buildAndSavePLY(
    const std::string &path,
    const cv::Mat &disparity,
    const cv::Mat &Q,
    const cv::Mat &P1r,
    const cv::Mat &P2r,
    const cv::Mat &camToWorld,
    const cv::Mat &rectColor,
    int minDisp,
    TriangulationMethod method)
{
    std::cout << "Orchestrating point cloud export to: " << path << "\n";

    PointCloud cloud = buildPointCloud(disparity, Q, P1r, P2r, camToWorld, rectColor, minDisp, method);

    if (cloud.pts.empty())
    {
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
    TriangulationMethod method)
{
    cv::Mat disp32f;
    if (disparity.type() == CV_32F)
        disp32f = disparity;
    else
        disparity.convertTo(disp32f, CV_32F);

    cv::Mat pts3D = Triangulation::reprojectDisparityTo3D(disp32f, Q, P1r, P2r, method);
    if (pts3D.empty())
        return PointCloud();

    PointCloud cloud;

    const cv::Matx33d R = camToWorld.colRange(0, 3);
    const cv::Vec3d t = camToWorld.col(3);

    const float zMax = 9000.0f;

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

            // Project coordinate elements into the global tracking frame
            cv::Vec3d pointInCam = cv::Vec3d(p[0], p[1], p[2]);
            cv::Vec3d w = R * pointInCam + t;

            cloud.pts.push_back(Eigen::Vector3f((float)w[0], (float)w[1], (float)w[2]));
            cloud.colors.push_back(rectColor.at<cv::Vec3b>(y, x));
        }
    }

    std::cout << "Dense extraction sequence finalized. Points compiled: " << cloud.pts.size() << "\n";
    return cloud;
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
    for (size_t i : idx)
    {
        out.pts.push_back(cloud.pts[i]);
        out.colors.push_back(cloud.colors[i]);
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
