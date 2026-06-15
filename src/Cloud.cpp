#include "Cloud.hpp"
#include <opencv2/calib3d.hpp>
#include <fstream>

Cloud CloudUtils::build(const PipelineResult& res, const cv::Mat& disp)
{
    cv::Mat Q;
    cv::Mat pts3D;
    cv::reprojectImageTo3D(disp, pts3D, Q, true);
    
    Cloud cloud;
    const float maxZ = 1e4f;
    for (int y = 0; y < pts3D.rows; ++y)
        for (int x = 0; x < pts3D.cols; ++x)
        {
            if (disp.at<float>(y, x) <= 0.f) continue;
            cv::Vec3f p = pts3D.at<cv::Vec3f>(y, x);
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
            if (std::abs(p[2]) > maxZ) continue;
            cloud.pts.push_back({ p[0], p[1], p[2] });
            cloud.colors.push_back(res.rectColor.at<cv::Vec3b>(y, x));
        }
    return cloud;
}

// -----------------------------------------------------------------------
// Normalise to zero-mean, unit std — makes ICP scale-agnostic
// Returns {mean, scale} so we can undo normalisation after fusion
// -----------------------------------------------------------------------
std::pair<Eigen::Vector3f, float> CloudUtils::normalise(Cloud& cloud)
{
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const auto& p : cloud.pts) mean += p;
    mean /= float(cloud.pts.size());

    float scale = 0.f;
    for (const auto& p : cloud.pts) scale += (p - mean).squaredNorm();
    scale = std::sqrt(scale / float(cloud.pts.size()));
    if (scale < 1e-6f) scale = 1.f;

    for (auto& p : cloud.pts) p = (p - mean) / scale;
    return { mean, scale };
}

void CloudUtils::denormalise(Cloud& cloud, const Eigen::Vector3f& mean, float scale)
{
    for (auto& p : cloud.pts) p = p * scale + mean;
}


Cloud CloudUtils::subsample(const Cloud& cloud, size_t n, std::mt19937& rng)
{
    if (cloud.pts.size() <= n) return cloud;
    std::vector<size_t> idx(cloud.pts.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(n);

    Cloud out;
    for (size_t i : idx) { out.pts.push_back(cloud.pts[i]); out.colors.push_back(cloud.colors[i]); }
    return out;
}

void CloudUtils::savePLY(const std::string& path, const Cloud& cloud)
{
    std::ofstream f(path);
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << cloud.pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    for (size_t i = 0; i < cloud.pts.size(); ++i)
        f << cloud.pts[i].x() << " " << cloud.pts[i].y() << " " << cloud.pts[i].z() << " "
          << (int)cloud.colors[i][2] << " " << (int)cloud.colors[i][1] << " " << (int)cloud.colors[i][0] << "\n";
    std::cout << "Saved " << cloud.pts.size() << " points to " << path << "\n";
}