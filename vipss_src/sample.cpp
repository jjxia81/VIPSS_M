#include <iostream>
#include <vector>
#include <set>
#include <limits>
#include <armadillo>
#include <string>
#include "sample.h"
#include "readers.h"

namespace PTSample {

double squaredDistance(double x, double y, double z, double px, double py, double pz)
{
    double dx = x - px;
    double dy = y - py;
    double dz = z - pz;
    return dx * dx + dy * dy + dz * dz;
}

double squaredDistance(const std::vector<double>& pts, size_t i, size_t j)
{
    double dx = pts[3 * i] - pts[3 * j];
    double dy = pts[3 * i + 1] - pts[3 * j + 1];
    double dz = pts[3 * i + 2] - pts[3 * j + 2];

    return dx* dx + dy * dy + dz * dz;
}

void FurthestSamplePointCloud(const std::vector<double>& pts, size_t sample_num, std::vector<double>& out_key_pts, std::vector<double>& out_auxi_pts)
{
    size_t pt_num = pts.size() / 3;
    std::set<size_t> sample_ids;
    
    sample_num = sample_num > pt_num ?  pt_num : sample_num;
    
    double MAXDV = std::numeric_limits<double>::max();
    arma::mat distMat(sample_num, pt_num); 
    size_t can_id = 0;
    arma::vec min_dist_vec(pt_num, arma::fill::ones);
    min_dist_vec *= MAXDV;

    while(sample_ids.size() < sample_num)
    {
        sample_ids.insert(can_id);
        size_t r_id = sample_ids.size() - 1; 
        for(size_t i = 0; i < pt_num; ++i)
        {
            double dist = squaredDistance(pts, i, can_id);
            min_dist_vec[i] = min_dist_vec[i] < dist ? min_dist_vec[i] : dist;
        }
        can_id = min_dist_vec.index_max();
    }

    // std::vector<double> out_pts;
    out_key_pts.clear();
    out_auxi_pts.clear();

    for(size_t i = 0; i < pt_num; ++i)
    {
        if(sample_ids.find(i) != sample_ids.end())
        {
            out_key_pts.push_back(pts[3* i]);
            out_key_pts.push_back(pts[3* i + 1]);
            out_key_pts.push_back(pts[3* i + 2]);
        } else {
            out_auxi_pts.push_back(pts[3* i]);
            out_auxi_pts.push_back(pts[3* i + 1]);
            out_auxi_pts.push_back(pts[3* i + 2]);
        }
    }

    // std::string out_path = "sample.ply";
    // writePLYFile(out_path, out_key_pts);
}

}

