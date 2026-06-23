#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <FreeImageHelper.h>

struct StereoPair
{
    FreeImageB imageLeft;
    FreeImageB imageRight;
};

struct CameraPose
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix3d K;
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

    CameraPose loadCameraPose(const std::string &imgPath)
    {
        CameraPose pose;

        size_t rpos = imgPath.find("Rectified");
        size_t fpos = imgPath.find("rect_");
        if (rpos == std::string::npos || fpos == std::string::npos)
        {
            std::cerr << "ERROR: cannot parse view id from " << imgPath << "\n";
            return pose;
        }
        std::string base = imgPath.substr(0, rpos);
        std::string id = imgPath.substr(fpos + 5, 3);
        std::string calPath = base + "Calibration/cal18/pos_" + id + ".txt";

        std::ifstream file(calPath);
        if (!file.is_open())
        {
            std::cerr << "ERROR: cannot open calibration file " << calPath << "\n";
            return pose;
        }

        // read 3x4 Reprojection Matrix
        cv::Mat P(3, 4, CV_64F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; ++j)
            {
                file >> P.at<double>(i, j);
            }
        }
        file.close();

        // decomposition and Eigen conversion
        cv::Mat K_cv, R_cv, t_homogeneous;
        cv::decomposeProjectionMatrix(P, K_cv, R_cv, t_homogeneous);

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                pose.K(i, j) = K_cv.at<double>(i, j);
                pose.R(i, j) = R_cv.at<double>(i, j);
            }
            // divide by W to convert homogeneous back to normal 3D coordinates
            pose.t(i) = t_homogeneous.at<double>(i, 0) / t_homogeneous.at<double>(3, 0);
        }
        return pose;
    }

    // only load k matrix
    cv::Mat loadIntrinsicCV(const std::string &imgPath)
    {
        CameraPose pose = loadCameraPose(imgPath);
        cv::Mat K_cv(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                K_cv.at<double>(i, j) = pose.K(i, j);
            }
        }
        return K_cv;
    }

    static inline void getRelativePose(const CameraPose &pose1, const CameraPose &pose2,
                                       Eigen::Matrix3d &R_rel, Eigen::Vector3d &t_rel)
    {
        R_rel = pose2.R * pose1.R.transpose();
        t_rel = pose2.R * (pose1.t - pose2.t);
        if (t_rel.norm() > 0)
        {
            t_rel.normalize();
        }
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