#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>

int main() {
    // Load stereo images
    cv::Mat left  = cv::imread("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_001_3_r5000.png",  cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread("../data/dtu/SampleSet/MVS Data/Rectified/scan1/rect_002_3_r5000.png", cv::IMREAD_GRAYSCALE);

    // Compute disparity map
    auto stereo = cv::StereoSGBM::create(
        0,    // minDisparity
        64,   // numDisparities (must be divisible by 16)
        11     // blockSize
    );

    cv::Mat disparity;
    stereo->compute(left, right, disparity);

    // Normalize for visualization
    cv::Mat disp8;
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity", disp8);
    cv::waitKey(0);

    return 0;   
}