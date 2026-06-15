#include "ICP.hpp"
#include <opencv2/flann.hpp>

// -----------------------------------------------------------------------
// ICP: align source INTO target frame, returns 4x4 rigid transform
// -----------------------------------------------------------------------
Eigen::Matrix4f ICP::align(Cloud& source, const Cloud& target,
                            int maxIter, float distThresh)
{
    // Build FLANN KD-tree on target
    int n = int(target.pts.size());
    cv::Mat targetMat(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
            targetMat.at<float>(i, j) = target.pts[i](j);

    cv::flann::Index kdtree(targetMat, cv::flann::KDTreeIndexParams(4));

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    for (int iter = 0; iter < maxIter; ++iter)
    {
        int m = int(source.pts.size());
        cv::Mat queryMat(m, 3, CV_32F);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < 3; ++j)
                queryMat.at<float>(i, j) = source.pts[i](j);

        cv::Mat indices(m, 1, CV_32S);
        cv::Mat dists(m, 1, CV_32F);
        kdtree.knnSearch(queryMat, indices, dists, 1);

        // Collect inlier pairs
        std::vector<Eigen::Vector3f> src, tgt;
        for (int i = 0; i < m; ++i)
        {
            if (dists.at<float>(i, 0) > distThresh * distThresh) continue;
            src.push_back(source.pts[i]);
            tgt.push_back(target.pts[indices.at<int>(i, 0)]);
        }
        if ((int)src.size() < 6) break;

        // Compute centroids
        Eigen::Vector3f cS = Eigen::Vector3f::Zero(), cT = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < src.size(); ++i) { cS += src[i]; cT += tgt[i]; }
        cS /= float(src.size()); cT /= float(src.size());

        // Cross-covariance H = sum( (src - cS) * (tgt - cT)^T )
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < src.size(); ++i)
            H += (src[i] - cS) * (tgt[i] - cT).transpose();

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0)
        {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::Vector3f t = cT - R * cS;

        // Apply to source
        for (auto& p : source.pts) p = R * p + t;

        // Accumulate into T
        Eigen::Matrix4f dT = Eigen::Matrix4f::Identity();
        dT.block<3,3>(0,0) = R;
        dT.block<3,1>(0,3) = t;
        T = dT * T;
    }

    return T;
}