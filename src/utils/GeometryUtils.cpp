#include "GeometryUtils.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace GeometryUtils
{

    bool extractPoseFromFundamental(const Eigen::Matrix3d &F_eigen,
                                    const std::vector<cv::Point2f> &ptsL,
                                    const std::vector<cv::Point2f> &ptsR,
                                    const std::vector<bool> &inlierMask,
                                    const cv::Mat &K,
                                    Eigen::Matrix3d &R_est,
                                    Eigen::Vector3d &t_est)
    {
        // filter points to only include the robust inliers
        std::vector<cv::Point2f> inL, inR;
        for (size_t i = 0; i < inlierMask.size(); ++i)
        {
            if (inlierMask[i])
            {
                inL.push_back(ptsL[i]);
                inR.push_back(ptsR[i]);
            }
        }

        if (inL.size() < 5)
            return false;

        // Convert Eigen F to OpenCV F
        cv::Mat F_cv;
        cv::eigen2cv(F_eigen, F_cv);

        // E = K^T * F * K
        cv::Mat E = K.t() * F_cv * K;

        cv::Mat R_cv, t_cv;
        cv::recoverPose(E, inL, inR, K, R_cv, t_cv);

        cv::cv2eigen(R_cv, R_est);
        cv::cv2eigen(t_cv, t_est);

        return true;
    }

}