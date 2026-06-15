#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

/* Parse ground truth projection matrix from DTU dataset by camera id,
to get a relative camera pose between two cameras */

struct CameraPose
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix3d K;
};

class ProjectionMatrixParser
{
public:
    ProjectionMatrixParser(const std::string &calibDir) : m_baseDir(calibDir) {}

    inline CameraPose loadPose(int cameraId)
    {
        CameraPose pose;

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/SampleSet/MVS Data/Calibration/cal18/pos_%03d.txt", m_baseDir.c_str(), cameraId);

        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: could not open file" << filename << "\n";
            return pose;
        }

        // read 3x4 reprojection matrix
        cv::Mat P(3, 4, CV_64F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; ++j)
            {
                file >> P.at<double>(i, j);
            }
        }
        file.close();
        cv::Mat K_cv, R_cv, t_homogeneous;
        cv::decomposeProjectionMatrix(P, K_cv, R_cv, t_homogeneous);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                pose.K(i, j) = K_cv.at<double>(i, j);
                pose.R(i, j) = R_cv.at<double>(i, j);
            }
            // divide by W to convert it back to normal 3D coordinates
            pose.t(i) = t_homogeneous.at<double>(i, 0) / t_homogeneous.at<double>(3, 0);
        }
        return pose;
    }

    static inline void getRelativePose(const CameraPose &pose1,
                                       const CameraPose &pose2, Eigen::Matrix3d &R_rel,
                                       Eigen::Vector3d &t_rel)
    {
        R_rel = pose2.R * pose1.R.transpose();
        t_rel = pose2.t - (R_rel * pose1.t);
        if (t_rel.norm() > 0)
        {
            t_rel.normalize();
        }
    }

private:
    std::string m_baseDir;
};