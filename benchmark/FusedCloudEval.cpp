// Alignment metrics for saved point clouds, in mm.
//
// Usage: FusedCloudEval cloud1.ply [cloud2.ply ...]
//
// Two views of quality:
//   1. Accuracy vs the DTU scan1 structured-light GT (stl001_total.ply),
//      restricted to points inside the GT bounding box (+10 mm margin) so
//      background/table points the GT does not cover are excluded.
//   2. Mutual consistency: for every input cloud after the first, the NN
//      distance to the FIRST cloud over the overlap region (NN < 20 mm).
//      This isolates the ICP placement error from stereo depth error --
//      well-registered clouds have a small consistency median regardless of
//      their absolute accuracy.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/flann.hpp>
#include "DTULoader.hpp"
#include "PlyUtils.hpp"

namespace {

struct Stats { double mean, median, rms, p90; double insideFrac; };

Stats distStats(std::vector<double> &d)
{
    Stats s{0, 0, 0, 0, 0};
    if (d.empty()) return s;
    std::sort(d.begin(), d.end());
    double sum = 0, sumSq = 0;
    for (double v : d) { sum += v; sumSq += v * v; }
    s.mean = sum / d.size();
    s.median = d[d.size() / 2];
    s.rms = std::sqrt(sumSq / d.size());
    s.p90 = d[(size_t)(0.9 * (d.size() - 1))];
    return s;
}

cv::Mat toMat(const std::vector<Eigen::Vector3f> &pts)
{
    cv::Mat m((int)pts.size(), 3, CV_32F);
    for (int i = 0; i < (int)pts.size(); ++i)
    {
        m.at<float>(i, 0) = pts[i].x();
        m.at<float>(i, 1) = pts[i].y();
        m.at<float>(i, 2) = pts[i].z();
    }
    return m;
}

// NN distances (mm) of query points against an index; only queries inside
// [lo,hi] are considered, and insideFrac reports how many qualified.
std::vector<double> nnDistances(const std::vector<Eigen::Vector3f> &pts,
                                cv::flann::Index &index,
                                const Eigen::Vector3f &lo, const Eigen::Vector3f &hi,
                                double &insideFrac)
{
    std::vector<Eigen::Vector3f> kept;
    kept.reserve(pts.size());
    for (const auto &p : pts)
        if ((p.array() >= lo.array()).all() && (p.array() <= hi.array()).all())
            kept.push_back(p);
    insideFrac = pts.empty() ? 0.0 : (double)kept.size() / (double)pts.size();
    if (kept.empty()) return {};

    cv::Mat query = toMat(kept);
    cv::Mat indices((int)kept.size(), 1, CV_32S), dists((int)kept.size(), 1, CV_32F);
    index.knnSearch(query, indices, dists, 1, cv::flann::SearchParams(64));

    std::vector<double> d(kept.size());
    for (int i = 0; i < (int)kept.size(); ++i)
        d[i] = std::sqrt((double)dists.at<float>(i, 0)); // FLANN returns squared L2
    return d;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " cloud1.ply [cloud2.ply ...]\n";
        return 1;
    }

    DTULoader loader("../data/dtu/");
    std::vector<cv::Point3f> gt = loader.loadPointCloud(1);
    if (gt.empty())
    {
        std::cerr << "Failed to load DTU ground-truth scan.\n";
        return 1;
    }

    Eigen::Vector3f lo(1e9f, 1e9f, 1e9f), hi(-1e9f, -1e9f, -1e9f);
    cv::Mat gtMat((int)gt.size(), 3, CV_32F);
    for (int i = 0; i < (int)gt.size(); ++i)
    {
        gtMat.at<float>(i, 0) = gt[i].x;
        gtMat.at<float>(i, 1) = gt[i].y;
        gtMat.at<float>(i, 2) = gt[i].z;
        lo = lo.cwiseMin(Eigen::Vector3f(gt[i].x, gt[i].y, gt[i].z));
        hi = hi.cwiseMax(Eigen::Vector3f(gt[i].x, gt[i].y, gt[i].z));
    }
    const Eigen::Vector3f margin(10.0f, 10.0f, 10.0f);
    lo -= margin;
    hi += margin;

    std::cout << "Building KD-tree on " << gt.size() << " GT points...\n";
    cv::flann::Index gtIndex(gtMat, cv::flann::KDTreeIndexParams(4));

    std::vector<PointCloud> clouds;
    std::vector<std::string> names;
    for (int a = 1; a < argc; ++a)
    {
        PointCloud c = PlyUtils::loadPLY(argv[a]);
        if (c.pts.empty())
        {
            std::cerr << argv[a] << ": no points, skipping.\n";
            continue;
        }
        clouds.push_back(std::move(c));
        names.push_back(argv[a]);
    }
    if (clouds.empty()) return 1;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- Accuracy vs DTU GT (inside GT bbox +10mm), mm ---\n"
              << std::left << std::setw(36) << "cloud"
              << std::right << std::setw(10) << "in-box%"
              << std::setw(9) << "mean" << std::setw(9) << "median"
              << std::setw(9) << "rms" << std::setw(9) << "p90" << "\n";
    for (size_t i = 0; i < clouds.size(); ++i)
    {
        double insideFrac = 0.0;
        auto d = nnDistances(clouds[i].pts, gtIndex, lo, hi, insideFrac);
        Stats s = distStats(d);
        std::cout << std::left << std::setw(36) << names[i]
                  << std::right << std::setw(10) << (100.0 * insideFrac)
                  << std::setw(9) << s.mean << std::setw(9) << s.median
                  << std::setw(9) << s.rms << std::setw(9) << s.p90 << "\n";
    }

    if (clouds.size() > 1)
    {
        // Consistency of every later cloud against the first input, over the
        // overlap region only (NN < 20 mm keeps it insensitive to coverage
        // differences between viewpoints).
        const double overlapCap = 20.0;
        cv::Mat refMat = toMat(clouds[0].pts);
        cv::flann::Index refIndex(refMat, cv::flann::KDTreeIndexParams(4));
        const Eigen::Vector3f noLo(-1e9f, -1e9f, -1e9f), noHi(1e9f, 1e9f, 1e9f);

        std::cout << "\n--- Consistency vs " << names[0]
                  << " (overlap = NN < " << overlapCap << "mm), mm ---\n"
                  << std::left << std::setw(36) << "cloud"
                  << std::right << std::setw(10) << "overlap%"
                  << std::setw(9) << "mean" << std::setw(9) << "median"
                  << std::setw(9) << "rms" << std::setw(9) << "p90" << "\n";
        for (size_t i = 1; i < clouds.size(); ++i)
        {
            double unused = 0.0;
            auto d = nnDistances(clouds[i].pts, refIndex, noLo, noHi, unused);
            std::vector<double> overlap;
            overlap.reserve(d.size());
            for (double v : d)
                if (v < overlapCap)
                    overlap.push_back(v);
            const double overlapFrac = d.empty() ? 0.0 : (double)overlap.size() / (double)d.size();
            Stats s = distStats(overlap);
            std::cout << std::left << std::setw(36) << names[i]
                      << std::right << std::setw(10) << (100.0 * overlapFrac)
                      << std::setw(9) << s.mean << std::setw(9) << s.median
                      << std::setw(9) << s.rms << std::setw(9) << s.p90 << "\n";
        }
    }
    return 0;
}
