#include <iostream>
#include <opencv2/core.hpp>
#include "DTULoader.hpp"
#include <opencv2/imgproc.hpp>

int main()
{
    std::string datasetPath = "../data/dtu";

    DTULoader loader(datasetPath);
    StereoPair pair = loader.loadPair(1, 1, 2);

    if (pair.imageLeft.data == nullptr || pair.imageRight.data == nullptr)
    {
        std::cerr << "\nERROR: Failed to load images\n";
        return -1;
    }

    cv::Mat cvLeft(pair.imageLeft.h, pair.imageLeft.w, CV_8UC4, pair.imageLeft.data);
    cv::Mat bgrLeft;
    cv::cvtColor(cvLeft, bgrLeft, cv::COLOR_RGBA2BGR);

    return 0;
}