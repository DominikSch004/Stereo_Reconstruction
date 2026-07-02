#include "DTULoader.hpp"
#include "SparseKeyPointMatcher.hpp"
#include "ImgUtils.hpp"

int main()
{
    std::cout << "Initializing Pipeline Evaluation\n";

    int datasetChoice = 0;

    // Loop until the user provides a valid choice (1 or 2)
    while (true)
    {
        std::cout << "\nSelect dataset:\n";
        std::cout << "1) DTU\n";
        std::cout << "2) Other...\n";
        std::cout << "Enter choice (1-2): ";

        std::cin >> datasetChoice;

        if (datasetChoice == 1 || datasetChoice == 2)
        {
            break;
        }

        std::cout << "Invalid choice! Please select 1 or 2.\n";
    }

    StereoPair pair;
    if (datasetChoice == 1)
    {
        std::cout << "\nLoading DTU dataset...\n";
        // 1. Load data
        DTULoader loader("../data/dtu/");

        pair = loader.loadPairVerbose();
    }
    else if (datasetChoice == 2)
    {
        DTULoader loader("");
        std::string pathLeft;
        std::string pathRight;

        std::cout << "\nEnter 1st img path: ";
        std::cin >> pathLeft;

        std::cout << "Enter 2nd img path: ";
        std::cin >> pathRight;

        pair = loader.loadPair(pathLeft, pathRight);
    }
    cv::Mat grayLeft = toGray(pair.imageLeft);
    cv::Mat grayRight = toGray(pair.imageRight);

    float userRatio;

    std::cout << "\n";

    std::cout << "Initializing Sparse Feature Matching... \n\n";

    std::cout << "Enter the ratio threshold for matching in range 0.01 - 1.0 (default = 0.75): ";
    std::cin >> userRatio;

    if (userRatio < 0.01f || userRatio > 1.0f)
    {
        std::cout << "Invalid input! Using default value (0.75).\n\n";
        userRatio = 0.75f;
    }

    SparseKeyPointMatcher matcher(userRatio);
    MatchResult result = matcher.match(grayLeft, grayRight);

    matcher.visualize(result, grayLeft, grayRight);

    return 0;
}