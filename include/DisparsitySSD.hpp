// DisparitySSD.hpp

class DisparitySSD
{
public:
    static cv::Mat compute(
        const cv::Mat& left,
        const cv::Mat& right,
        int maxDisp,
        int blockSize);
};