#include <iostream>
#include "ProjectionMatrixParser.hpp"

int main()
{
    std::string dataDir = "../data/dtu";

    ProjectionMatrixParser parser(dataDir);

    CameraPose pose1 = parser.loadPose(1);
    CameraPose pose2 = parser.loadPose(2);

    Eigen::Matrix3d R_rel;
    Eigen::Vector3d t_rel;
    ProjectionMatrixParser::getRelativePose(pose1, pose2, R_rel, t_rel);

    std::cout << "Relative Rotation:\n"
              << R_rel << "\n";
    std::cout << "Relative Translation:\n"
              << t_rel.transpose() << "\n\n";
}