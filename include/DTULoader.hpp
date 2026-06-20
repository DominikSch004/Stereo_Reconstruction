#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <FreeImageHelper.h>
#include <opencv2/calib3d.hpp>

/* Assumes path is saved as %datasetDir%/Rectified/scan%id%/%image%, loads an image pair to memory from
three ids, scan_id, rightImageId and viewIdRight + an optional illumination */

struct StereoPair
{
    FreeImageB imageLeft;
    FreeImageB imageRight;
};

class DTULoader
{
public:
    DTULoader(const std::string &datasetDir) : m_baseDir(datasetDir) {}

    StereoPair loadPair(int scanId, int viewIdLeft, int viewIdRight, int illumination = 3)
    {
        StereoPair pair;

        std::string pathLeft = buildImagePath(scanId, viewIdLeft, illumination);
        std::string pathRight = buildImagePath(scanId, viewIdRight, illumination);

        std::cout << "Loading left image: " << pathLeft << "\n";
        pair.imageLeft = loadToFreeImage(pathLeft);

        std::cout << "Loading right image: " << pathRight << "\n";
        pair.imageRight = loadToFreeImage(pathRight);

        return pair;
    }

    StereoPair loadPair(const std::string &pathLeft, const std::string &pathRight)
    {
        StereoPair pair;

        std::cout << "Loading left image: " << pathLeft << "\n";
        pair.imageLeft = loadToFreeImage(pathLeft);

        std::cout << "Loading right image: " << pathRight << "\n";
        pair.imageRight = loadToFreeImage(pathRight);

        return pair;
    }

    cv::Mat loadDTUProjection(const std::string& imgPath) 
    {
        size_t rpos = imgPath.find("Rectified");
        size_t fpos = imgPath.find("rect_");
        if (rpos == std::string::npos || fpos == std::string::npos) {
            std::cerr << "ERROR: cannot parse view id from " << imgPath << "\n";
            return cv::Mat();
        }
        std::string base = imgPath.substr(0, rpos);
        std::string id   = imgPath.substr(fpos + 5, 3);
        std::string calPath = base + "Calibration/cal18/pos_" + id + ".txt";

        std::ifstream f(calPath);
        if (!f.is_open()) {
            std::cerr << "ERROR: cannot open calibration file " << calPath << "\n";
            return cv::Mat();
        }
        cv::Mat P(3, 4, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                if (!(f >> P.at<double>(i, j))) {
                    std::cerr << "ERROR: malformed calibration file " << calPath << "\n";
                    return cv::Mat();
                }
        return P;
    }

    cv::Mat getIntrinsicFromProjection(const cv::Mat& P)
    {
        cv::Mat K, R, t;
        cv::decomposeProjectionMatrix(P, K, R, t);
        cv::Mat K64;
        K.convertTo(K64, CV_64F);
        return K64;
    }

private:
    std::string m_baseDir;

    std::string buildImagePath(int scanId, int viewId, int illumination)
    {
        char viewStr[10];
        snprintf(viewStr, sizeof(viewStr), "%03d", viewId);
        return m_baseDir + "/SampleSet/MVS Data/Rectified/scan" + std::to_string(scanId) +
               "/rect_" + std::string(viewStr) + "_" + std::to_string(illumination) + "_r5000.png";
    }

    FreeImageB loadToFreeImage(const std::string &filepath)
    {
        FreeImageB fi;
        if (!fi.LoadImageFromFile(filepath))
        {
            std::cerr << "ERROR: Failed to load image at " << filepath << "\n";
        }
        return fi;
    }
};