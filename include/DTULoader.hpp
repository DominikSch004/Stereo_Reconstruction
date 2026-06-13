#pragma once
#include <iostream>
#include <string>
#include <FreeImageHelper.h>

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