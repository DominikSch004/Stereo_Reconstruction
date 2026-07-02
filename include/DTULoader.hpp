#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include "PlyUtils.hpp"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

// #include <FreeImageHelper.h> if we cannot use openCV data structres for custom approach we can go back to this.

struct StereoPair
{
    cv::Mat imageLeft;
    cv::Mat imageRight;
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

    StereoPair loadPair(int viewIdLeft, int viewIdRight, int datasetID = 1, int illumination = 3)
    {
        std::string pathLeft = buildImagePath(datasetID, viewIdLeft, illumination);
        std::string pathRight = buildImagePath(datasetID, viewIdRight, illumination);

        return loadPair(pathLeft, pathRight);
    }

    StereoPair loadPair(const std::string &pathLeft, const std::string &pathRight)
    {
        StereoPair pair;
        std::cout << "Loading left image: " << pathLeft << "\n";
        pair.imageLeft = cv::imread(pathLeft, cv::IMREAD_COLOR);
        if (pair.imageLeft.empty())
        {
            std::cerr << "ERROR: Failed to load left image! Check if the path exists: " << pathLeft << "\n";
        }

        std::cout << "Loading right image: " << pathRight << "\n";
        pair.imageRight = cv::imread(pathRight, cv::IMREAD_COLOR);
        if (pair.imageRight.empty())
        {
            std::cerr << "ERROR: Failed to load right image! Check if the path exists: " << pathRight << "\n";
        }

        return pair;
    }

    CameraPose loadCameraPose(int imageId, int datasetId = 1)
    {
        char idStr[4];
        snprintf(idStr, sizeof(idStr), "%03d", imageId);
        std::string calPath = m_baseDir + "SampleSet/MVS Data/Calibration/cal18/pos_" + std::string(idStr) + ".txt";

        return loadPoseFromTxt(calPath);
    }

    CameraPose loadCameraPose(const std::string &imgPath)
    {
        size_t rpos = imgPath.find("Rectified");
        size_t fpos = imgPath.find("rect_");
        if (rpos == std::string::npos || fpos == std::string::npos)
        {
            std::cerr << "ERROR: Cannot parse view id from path: " << imgPath << "\n";
            return CameraPose(); // Return empty pose
        }

        std::string base = imgPath.substr(0, rpos);
        std::string id = imgPath.substr(fpos + 5, 3);
        std::string calPath = base + "SampleSet/MVS Data/Calibration/cal18/pos_" + id + ".txt";

        return loadPoseFromTxt(calPath);
    }

    std::vector<cv::Point3f> loadPointCloud(int datasetId = 1)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "SampleSet/MVS Data/Points/stl/stl%03d_total.ply", datasetId);
        std::string plyPath = m_baseDir + std::string(buf);
        return loadPointCloudFromPath(plyPath);
    }

    std::vector<cv::Point3f> loadPointCloud(const std::string &plyPath)
    {
        return loadPointCloudFromPath(plyPath);
    }

    cv::Mat loadIntrinsicCV(int imageId, int datasetId = 1)
    {
        CameraPose pose = loadCameraPose(imageId, datasetId);
        return poseToInstrinsics(pose);
    }

    cv::Mat loadIntrinsicCV(const std::string &imgPath)
    {
        CameraPose pose = loadCameraPose(imgPath);
        return poseToInstrinsics(pose);
    }

    // Compute relative rotation and translation between two poses
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

    std::string buildImagePath(int datasetID, int viewId, int illumination)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "SampleSet/MVS Data/Rectified/scan%d/rect_%03d_%d_r5000.png", datasetID, viewId, illumination);
        return m_baseDir + std::string(buf);
    }

    CameraPose loadPoseFromTxt(const std::string &calPath)
    {
        CameraPose pose;
        pose.R = Eigen::Matrix3d::Identity();
        pose.K = Eigen::Matrix3d::Identity();
        pose.t = Eigen::Vector3d::Zero();

        std::ifstream file(calPath);
        if (!file.is_open())
        {
            std::cerr << "ERROR: Cannot open calibration file! Path does not exist: " << calPath << "\n";
            return pose;
        }

        // read 3x4 Reprojection Matrix
        cv::Mat P(3, 4, CV_64F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (!(file >> P.at<double>(i, j)))
                {
                    std::cerr << "ERROR: Failed to read matrix data from " << calPath << " (file might be corrupted or empty)\n";
                    return pose;
                }
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

    std::vector<cv::Point3f> loadPointCloudFromPath(const std::string &plyPath)
    {
        PointCloud eigenCloud = PlyUtils::loadPLY(plyPath);

        std::vector<cv::Point3f> cvCloud;
        cvCloud.reserve(eigenCloud.pts.size());

        for (const auto &pt : eigenCloud.pts)
        {
            cvCloud.push_back(cv::Point3f(pt.x(), pt.y(), pt.z()));
        }

        return cvCloud;
    }

    cv::Mat poseToInstrinsics(const CameraPose &pose)
    {
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
};