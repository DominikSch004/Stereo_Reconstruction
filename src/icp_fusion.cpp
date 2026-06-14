#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/flann.hpp>
#include "DTULoader.hpp"
#include "PairPipeline.hpp"

// -----------------------------------------------------------------------
// NCC disparity
// -----------------------------------------------------------------------
static cv::Mat computeNCC(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    const float n = float(blockSize * blockSize);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    cv::Mat sumL, sumLL;
    cv::filter2D(leftF,            sumL,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
    cv::filter2D(leftF.mul(leftF), sumLL, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    cv::Mat bestNCC(left.rows, left.cols, CV_32F, -std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat sumR, sumRR, sumLR;
        cv::filter2D(rightShifted,                   sumR,  CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(rightShifted.mul(rightShifted),  sumRR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::filter2D(leftF.mul(rightShifted),         sumLR, CV_32F, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat num = n * sumLR - sumL.mul(sumR);
        cv::Mat den; cv::sqrt((n * sumLL - sumL.mul(sumL)).mul(n * sumRR - sumR.mul(sumR)), den);

        cv::Mat ncc = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        cv::divide(num, den + 1e-6f, ncc);

        cv::Mat better = ncc > bestNCC;
        ncc.copyTo(bestNCC, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols, h))          = 0;
    disp(cv::Rect(0, left.rows - h, left.cols, h))          = 0;
    disp(cv::Rect(0,              0, h, left.rows))          = 0;
    disp(cv::Rect(left.cols - h,  0, h, left.rows))          = 0;
    return disp;
}


static cv::Mat computeSAD(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSAD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat absDiff;
        cv::absdiff(leftF, rightShifted, absDiff);

        cv::Mat sad;
        cv::filter2D(absDiff, sad, CV_32F, kernel,
                     cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = sad < bestSAD;
        sad.copyTo(bestSAD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols,  h))          = 0;
    disp(cv::Rect(0, left.rows - h, left.cols,  h))          = 0;
    disp(cv::Rect(0,              0,          h, left.rows))  = 0;
    disp(cv::Rect(left.cols - h,  0,          h, left.rows))  = 0;
    return disp;
}


static cv::Mat computeSSD(const cv::Mat& left, const cv::Mat& right,
                          int maxDisp, int blockSize)
{
    cv::Mat leftF, rightF;
    left.convertTo(leftF,  CV_32F);
    right.convertTo(rightF, CV_32F);

    cv::Mat bestSSD(left.rows, left.cols, CV_32F, std::numeric_limits<float>::max());
    cv::Mat disp(left.rows, left.cols, CV_32F, 0.0f);
    cv::Mat kernel = cv::Mat::ones(blockSize, blockSize, CV_32F);

    for (int d = 0; d < maxDisp; ++d)
    {
        cv::Mat rightShifted = cv::Mat::zeros(left.rows, left.cols, CV_32F);
        if (d < left.cols)
            rightF(cv::Rect(0, 0, left.cols - d, left.rows))
                .copyTo(rightShifted(cv::Rect(d, 0, left.cols - d, left.rows)));

        cv::Mat diff = leftF - rightShifted;
        cv::Mat ssd;
        cv::filter2D(diff.mul(diff), ssd, CV_32F, kernel,
                     cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

        cv::Mat better = ssd < bestSSD;
        ssd.copyTo(bestSSD, better);
        cv::Mat(left.rows, left.cols, CV_32F, float(d)).copyTo(disp, better);
    }

    int h = blockSize / 2;
    disp(cv::Rect(0,              0, left.cols,  h))           = 0;
    disp(cv::Rect(0, left.rows - h, left.cols,  h))           = 0;
    disp(cv::Rect(0,              0,          h, left.rows))   = 0;
    disp(cv::Rect(left.cols - h,  0,          h, left.rows))   = 0;
    return disp;
}



// -----------------------------------------------------------------------
// Point cloud
// -----------------------------------------------------------------------
struct Cloud
{
    std::vector<Eigen::Vector3f> pts;
    std::vector<cv::Vec3b>       colors;
};

static Cloud buildCloud(const PipelineResult& res, const cv::Mat& disp, int numDisp)
{
    cv::Mat Q = buildQ(res);
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
static std::pair<Eigen::Vector3f, float> normaliseCloud(Cloud& cloud)
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

static void denormaliseCloud(Cloud& cloud, const Eigen::Vector3f& mean, float scale)
{
    for (auto& p : cloud.pts) p = p * scale + mean;
}

// -----------------------------------------------------------------------
// Random subsample
// -----------------------------------------------------------------------
static Cloud subsample(const Cloud& cloud, size_t n, std::mt19937& rng)
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

// -----------------------------------------------------------------------
// ICP: align source INTO target frame, returns 4x4 rigid transform
// -----------------------------------------------------------------------
static Eigen::Matrix4f icp(Cloud& source, const Cloud& target,
                            int maxIter = 30, float distThresh = 0.1f)
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

// -----------------------------------------------------------------------
// Save PLY
// -----------------------------------------------------------------------
static void savePLY(const std::string& path, const Cloud& cloud)
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

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main()
{
    const std::string datasetPath = "../data/dtu";
    const int scanId    = 1;
    const int numPairs  = 5;
    const int maxDisp   = 128;
    const int blockSz   = 11;
    const size_t icpSamples = 4000;

    DTULoader loader(datasetPath);
    std::mt19937 rng(42);

    std::vector<Cloud> clouds;
    clouds.reserve(numPairs);

    for (int i = 1; i <= numPairs; ++i)
    {
        std::cout << "\n=== Pair (" << i << ", " << i+1 << ") ===\n";
        StereoPair pair = loader.loadPair(scanId, i, i + 1);

        PipelineResult res;
        if (!runPipeline(pair, res))
        {
            std::cerr << "Pipeline failed for pair " << i << "\n";
            continue;
        }


        // HERE CHANGE THIS TO YOUR DESIRED DISPARITY METHOD (NCC, SAD, SSD)
        std::cout << "SAD disparity...\n";
        cv::Mat disp = computeSAD(res.rectLeft, res.rectRight, maxDisp, blockSz);

        //std::cout << "NCC disparity...\n";
        //cv::Mat disp = computeNCC(res.rectLeft, res.rectRight, maxDisp, blockSz);

        //std::cout << "SSD disparity...\n";
        //cv::Mat disp = computeSSD(res.rectLeft, res.rectRight, maxDisp, blockSz); 
        

        Cloud cloud = buildCloud(res, disp, maxDisp);
        std::cout << "Cloud " << i << ": " << cloud.pts.size() << " points\n";
        clouds.push_back(std::move(cloud));
    }

    if (clouds.empty()) { std::cerr << "No clouds generated\n"; return -1; }

    // Normalise all clouds to the same scale for ICP
    std::vector<std::pair<Eigen::Vector3f, float>> norms(clouds.size());
    for (size_t i = 0; i < clouds.size(); ++i)
        norms[i] = normaliseCloud(clouds[i]);

    // ICP: align each cloud to the accumulated fused cloud
    Cloud fused = clouds[0];

    for (size_t i = 1; i < clouds.size(); ++i)
    {
        std::cout << "\nICP aligning cloud " << i+1 << " to fused cloud...\n";
        Cloud srcSub = subsample(clouds[i], icpSamples, rng);
        Cloud tgtSub = subsample(fused,     icpSamples, rng);

        icp(srcSub, tgtSub);

        // Apply the same transform found on subsampled source to full cloud
        // by re-running ICP on full source using the subsampled result as warm start
        icp(clouds[i], tgtSub, 10);

        // Append to fused
        for (size_t j = 0; j < clouds[i].pts.size(); ++j)
        {
            fused.pts.push_back(clouds[i].pts[j]);
            fused.colors.push_back(clouds[i].colors[j]);
        }

        std::cout << "Fused cloud size: " << fused.pts.size() << " points\n";
    }

    savePLY("pointcloud_fused.ply", fused);
    return 0;
}
