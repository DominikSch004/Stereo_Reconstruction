#include "MatchSerializer.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

void serializeMatchPoints(const std::string& filename,
                          const std::vector<cv::Point2f>& ptsLeft,
                          const std::vector<cv::Point2f>& ptsRight)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: Failed to open " << filename << " for writing\n";
        return;
    }

    uint32_t count = ptsLeft.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (size_t i = 0; i < count; ++i) {
        float x_left = ptsLeft[i].x, y_left = ptsLeft[i].y;
        float x_right = ptsRight[i].x, y_right = ptsRight[i].y;
        out.write(reinterpret_cast<const char*>(&x_left), sizeof(x_left));
        out.write(reinterpret_cast<const char*>(&y_left), sizeof(y_left));
        out.write(reinterpret_cast<const char*>(&x_right), sizeof(x_right));
        out.write(reinterpret_cast<const char*>(&y_right), sizeof(y_right));
    }

    out.close();
    std::cout << "Serialized " << count << " matches to " << filename << "\n";
}

bool deserializeMatchPoints(const std::string& filename,
                            std::vector<cv::Point2f>& ptsLeft,
                            std::vector<cv::Point2f>& ptsRight)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: Failed to open " << filename << " for reading\n";
        return false;
    }

    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    ptsLeft.clear();
    ptsRight.clear();
    ptsLeft.reserve(count);
    ptsRight.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        float x_left, y_left, x_right, y_right;
        in.read(reinterpret_cast<char*>(&x_left), sizeof(x_left));
        in.read(reinterpret_cast<char*>(&y_left), sizeof(y_left));
        in.read(reinterpret_cast<char*>(&x_right), sizeof(x_right));
        in.read(reinterpret_cast<char*>(&y_right), sizeof(y_right));
        ptsLeft.push_back(cv::Point2f(x_left, y_left));
        ptsRight.push_back(cv::Point2f(x_right, y_right));
    }

    in.close();
    std::cout << "Deserialized " << count << " matches from " << filename << "\n";
    return true;
}
