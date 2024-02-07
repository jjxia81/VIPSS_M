#pragma once

#include <iostream>
#include <vector>

namespace PTSample
{
    double squaredDistance(double x, double y, double z, double px, double py, double pz);

    double squaredDistance(const std::vector<double>& pts, size_t i, size_t j);

    void FurthestSamplePointCloud(const std::vector<double>& pts, size_t sample_num, std::vector<double>& out_pts, std::vector<double>& out_auxi_pts);
}



