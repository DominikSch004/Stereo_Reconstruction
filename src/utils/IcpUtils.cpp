#include "IcpUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>

namespace IcpUtils
{

void applyRigid(PointCloud &cloud, const Eigen::Matrix4f &transform)
{
    const Eigen::Matrix3f R = transform.block<3, 3>(0, 0);
    const Eigen::Vector3f t = transform.block<3, 1>(0, 3);
    for (auto &p : cloud.pts)
        p = R * p + t;
    for (auto &n : cloud.normals)
        n = R * n; // rigid transform -> rotation only, unit length preserved
}

Eigen::Matrix4f randomRigid(double angleDeg, double transMag, std::mt19937 &rng)
{
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::Vector3f axis(g(rng), g(rng), g(rng));
    axis.normalize();
    Eigen::Vector3f dir(g(rng), g(rng), g(rng));
    dir.normalize();

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) =
        Eigen::AngleAxisf((float)(angleDeg * M_PI / 180.0), axis).toRotationMatrix();
    T.block<3, 1>(0, 3) = dir * (float)transMag;
    return T;
}

Eigen::Matrix4f rectToWorldTransform(const cv::Mat &R1, const CameraPose &poseLeft)
{
    Eigen::Matrix3d R1e;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R1e(r, c) = R1.at<double>(r, c);

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = (poseLeft.R.transpose() * R1e.transpose()).cast<float>();
    T.block<3, 1>(0, 3) = poseLeft.t.cast<float>();
    return T;
}

void transformCloudToWorld(PointCloud &cloud, const cv::Mat &R1, const CameraPose &poseLeft)
{
    applyRigid(cloud, rectToWorldTransform(R1, poseLeft));
}

bool orderPair(DTULoader &loader, int a, int b,
               int &leftView, int &rightView,
               CameraPose &poseLeft, CameraPose &poseRight)
{
    CameraPose pa = loader.loadCameraPose(a);
    CameraPose pb = loader.loadCameraPose(b);
    Eigen::Vector3d tcam = pa.R * (pb.t - pa.t); // baseline a->b expressed in cam a's frame
    if (std::abs(tcam.x()) < std::abs(tcam.y()))
        return false; // vertical baseline
    if (tcam.x() >= 0.0)
    {
        leftView = a; rightView = b; poseLeft = pa; poseRight = pb;
    }
    else
    {
        leftView = b; rightView = a; poseLeft = pb; poseRight = pa; // reversed L/R
    }
    return true;
}

size_t cullByConfidence(PointCloud &cloud, float keepFrac)
{
    const size_t n = cloud.pts.size();
    if (keepFrac <= 0.0f || n == 0 || cloud.weights.size() != n)
        return 0;

    float wMax = 0.0f;
    for (float w : cloud.weights)
        wMax = std::max(wMax, w);
    if (wMax <= 0.0f)
        return 0;
    const float thresh = keepFrac * wMax;

    const bool hasColors = cloud.colors.size() == n;
    const bool hasNormals = cloud.normals.size() == n;
    const bool hasValid = cloud.validNormal.size() == n;

    PointCloud kept;
    kept.pts.reserve(n);
    kept.weights.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        if (cloud.weights[i] < thresh)
            continue;
        kept.pts.push_back(cloud.pts[i]);
        kept.weights.push_back(cloud.weights[i]);
        if (hasColors) kept.colors.push_back(cloud.colors[i]);
        if (hasNormals) kept.normals.push_back(cloud.normals[i]);
        if (hasValid) kept.validNormal.push_back(cloud.validNormal[i]);
    }
    const size_t removed = n - kept.pts.size();
    cloud = std::move(kept);
    return removed;
}

void saveIndividualCloud(const PointCloud &cloud, int leftView, int rightView,
                         const Eigen::Vector3f &mean, float scale,
                         const std::string &suffix, const Eigen::Matrix4f &anchor)
{
    PointCloud out = cloud; // copy so denormalising doesn't disturb the fusion pipeline
    PlyUtils::denormalise(out, mean, scale);
    applyRigid(out, anchor);

    char fname[80];
    std::snprintf(fname, sizeof(fname), "pointcloud_pair_%02d_%02d%s.ply",
                  leftView, rightView, suffix.c_str());
    PlyUtils::savePLY(fname, out);
    std::cout << "  Saved individual cloud -> " << fname << " (" << out.pts.size()
              << " points)\n";
}

} // namespace IcpUtils
