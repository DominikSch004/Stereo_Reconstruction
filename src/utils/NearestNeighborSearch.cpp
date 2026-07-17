#include "NearestNeighborSearch.hpp"

void NearestNeighborSearch::buildIndex(const std::vector<Eigen::Vector3f>& targetPoints)
{
    const int n = static_cast<int>(targetPoints.size());
    m_targetMat.create(n, 3, CV_32F);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
            m_targetMat.at<float>(i, j) = targetPoints[i](j);

    m_index = std::make_unique<cv::flann::Index>(m_targetMat, cv::flann::KDTreeIndexParams(4));
}

std::vector<Match> NearestNeighborSearch::queryMatches(const std::vector<Eigen::Vector3f>& queryPoints) const
{
    const int m = static_cast<int>(queryPoints.size());
    std::vector<Match> matches(m, Match{-1, 0.0f});
    if (!m_index || m == 0)
        return matches;

    cv::Mat queryMat(m, 3, CV_32F);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < 3; ++j)
            queryMat.at<float>(i, j) = queryPoints[i](j);

    cv::Mat indices(m, 1, CV_32S);
    cv::Mat dists(m, 1, CV_32F); // FLANN L2 index returns squared distances
    // 128 leaf checks instead of the default 32: on fused targets of 10^5-10^6
    // points the default returns visibly suboptimal neighbors, which reads as
    // correspondence noise in ICP and caps the achievable alignment accuracy.
    m_index->knnSearch(queryMat, indices, dists, 1, cv::flann::SearchParams(128));

    for (int i = 0; i < m; ++i)
    {
        const float d = dists.at<float>(i, 0);
        matches[i].distance = d;
        matches[i].idx = (d > m_maxDistanceSq) ? -1 : indices.at<int>(i, 0);
    }
    return matches;
}
