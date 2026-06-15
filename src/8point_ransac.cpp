#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "DTULoader.hpp"
#include "MatchSerializer.hpp"
#include "ImgUtils.hpp"
#include "FundamentalMatrix.hpp"

int main(int argc, char** argv)
{
    std::string leftPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png";

    std::string rightPath = "../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png";

    DTULoader loader("");
    StereoPair pair = loader.loadPair(leftPath, rightPath);

    if (!pair.imageLeft.data || !pair.imageRight.data)
    {
        std::cerr << "ERROR: Failed to load images\n";
        return -1;
    }

    std::vector<cv::Point2f> ptsL, ptsR;

    if (!deserializeMatchPoints("matches.bin", ptsL, ptsR))
    {
        std::cerr << "ERROR: Failed to load matches\n";
        return -1;
    }

    if (ptsL.size() < 8)
    {
        std::cerr << "Not enough correspondences\n";
        return -1;
    }

    std::vector<bool> inliers;

    Eigen::Matrix3d F =
        FundamentalMatrix::ransac(
            ptsL, ptsR, inliers);

    int nIn = std::count(inliers.begin(), inliers.end(), true);

    std::cout << "Inliers: "
              << nIn << " / " << ptsL.size() << "\n";

    std::cout << "F:\n" << F << "\n";

    return 0;
}