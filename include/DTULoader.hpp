#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include "PlyUtils.hpp"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

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

namespace fs = std::filesystem;

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

    CameraPose loadCameraPose(int imageId)
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

    cv::Mat loadFundamental(int leftImageId, int rightImageId)
    {
        char idStrLeft[4];
        char idStrRight[4];
        snprintf(idStrLeft, sizeof(idStrLeft), "%03d", leftImageId);
        snprintf(idStrRight, sizeof(idStrRight), "%03d", rightImageId);
        std::string calPathLeft = m_baseDir + "SampleSet/MVS Data/Calibration/cal18/pos_" + std::string(idStrLeft) + ".txt";
        std::string calPathRight = m_baseDir + "SampleSet/MVS Data/Calibration/cal18/pos_" + std::string(idStrRight) + ".txt";
        cv::Mat Pl = loadProjectionFromPath(calPathLeft);
        cv::Mat Pr = loadProjectionFromPath(calPathRight);
        return getFundamentalFromProjection(Pl, Pr);
    }

    cv::Mat loadFundamental(const std::string &leftProjectionMatrixPath, const std::string &rightProjectionMatrixPath)
    {
        cv::Mat Pl = loadProjectionFromPath(leftProjectionMatrixPath);
        cv::Mat Pr = loadProjectionFromPath(rightProjectionMatrixPath);
        return getFundamentalFromProjection(Pl, Pr);
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

    cv::Mat loadIntrinsicCV(int imageId)
    {
        CameraPose pose = loadCameraPose(imageId);
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

    std::string findRectified()
    {
        std::string rectifiedPath = "";

        for (const auto &entry : fs::recursive_directory_iterator(m_baseDir))
        {
            if (entry.is_directory() && entry.path().filename() == "Rectified")
            {
                rectifiedPath = entry.path().string();
                break;
            }
        }
        if (rectifiedPath.empty())
        {
            std::cerr << "ERROR: Could not find 'Rectified' folder inside " << m_baseDir << "\n";
            return "";
        }

        return rectifiedPath;
    }

    std::string buildImagePath(int datasetID, int viewId, int illumination)
    {
        std::string rectifiedFolder = findRectified();

        if (rectifiedFolder.empty())
        {
            return "";
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "scan%d/rect_%03d_%d_r5000.png", datasetID, viewId, illumination);

        fs::path finalPath = fs::path(rectifiedFolder) / buf;

        return finalPath.string();
    }

    CameraPose loadPoseFromTxt(const std::string &calPath)
    {
        CameraPose pose;
        pose.R = Eigen::Matrix3d::Identity();
        pose.K = Eigen::Matrix3d::Identity();
        pose.t = Eigen::Vector3d::Zero();

        cv::Mat P = loadProjectionFromPath(calPath);

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

    cv::Mat loadProjectionFromPath(const std::string &calPath)
    {
        // read 3x4 Reprojection Matrix
        cv::Mat P(3, 4, CV_64F);
        std::ifstream file(calPath);
        if (!file.is_open())
        {
            std::cerr << "ERROR: Cannot open calibration file! Path does not exist: " << calPath << "\n";
            return P;
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (!(file >> P.at<double>(i, j)))
                {
                    std::cerr << "ERROR: Failed to read matrix data from " << calPath << " (file might be corrupted or empty)\n";
                    return P;
                }
            }
        }
        file.close();
        return P;
    }

    cv::Mat getFundamentalFromProjection(const cv::Mat &P1, const cv::Mat &P2)
    {

        // SVD decomposes P1 into U, W, and V^T. The nullspace is the last row of V^T.
        cv::Mat w, u, vt;
        cv::SVD::compute(P1, w, u, vt, cv::SVD::FULL_UV);
        cv::Mat C1 = vt.row(3).t();

        // compute the epipole in the second view
        cv::Mat e2 = P2 * C1;

        // construct the skew-symmetric matrix for the epipole [e2]_x
        double x = e2.at<double>(0, 0);
        double y = e2.at<double>(1, 0);
        double z = e2.at<double>(2, 0);

        cv::Mat e2_skew = (cv::Mat_<double>(3, 3) << 0.0, -z, y,
                           z, 0.0, -x,
                           -y, x, 0.0);

        // compute the Moore-Penrose pseudo-inverse of P1
        cv::Mat P1_pinv;
        cv::invert(P1, P1_pinv, cv::DECOMP_SVD);

        // construct the Fundamental Matrix
        cv::Mat F = e2_skew * P2 * P1_pinv;

        // normalize F (standardizing by the bottom-right element F_33)
        if (std::abs(F.at<double>(2, 2)) > 1e-8)
        {
            F /= F.at<double>(2, 2);
        }
        else
        {
            cv::normalize(F, F);
        }

        return F;
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