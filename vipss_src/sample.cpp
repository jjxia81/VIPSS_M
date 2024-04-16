#include <iostream>
#include <vector>
#include <set>
#include <limits>

#include <string>
#include "sample.h"
#include "readers.h"

// namespace PTSample {


void PTSampler::init(const std::vector<double>& pts)
{
    inpts = pts;
    size_t pt_num = inpts.size() / 3;
    double MAXDV = std::numeric_limits<double>::max();
    min_dist_vec = arma::vec(pt_num, arma::fill::ones);
    min_dist_vec *= MAXDV;
    sample_ids.clear();

}

double PTSampler::squaredDistance(double x, double y, double z, double px, double py, double pz)
{
    double dx = x - px;
    double dy = y - py;
    double dz = z - pz;
    return dx * dx + dy * dy + dz * dz;
}

double PTSampler::squaredDistance(const std::vector<double>& pts, size_t i, size_t j)
{
    double dx = pts[3 * i] - pts[3 * j];
    double dy = pts[3 * i + 1] - pts[3 * j + 1];
    double dz = pts[3 * i + 2] - pts[3 * j + 2];

    return dx* dx + dy * dy + dz * dz;
}

void PTSampler::SplitSamplePts( std::vector<double>& out_key_pts, std::vector<double>& out_auxi_pts)
{
    out_key_pts.clear();
    out_auxi_pts.clear();
    size_t pt_num = inpts.size() / 3;

    for(size_t i = 0; i < pt_num; ++i)
    {
        if(sample_ids.find(i) != sample_ids.end())
        {
            out_key_pts.push_back(inpts[3* i]);
            out_key_pts.push_back(inpts[3* i + 1]);
            out_key_pts.push_back(inpts[3* i + 2]);
        } else {
            out_auxi_pts.push_back(inpts[3* i]);
            out_auxi_pts.push_back(inpts[3* i + 1]);
            out_auxi_pts.push_back(inpts[3* i + 2]);
        }
    }
}


void PTSampler::FurthestSamplePointCloud(size_t sample_num)
{
    size_t pt_num = inpts.size() / 3;
    sample_num = sample_num > pt_num ?  pt_num : sample_num;
    
    size_t can_id = 0;
    if(!sample_ids.empty())
    {
        can_id = min_dist_vec.index_max();
    }
    // cout << "can id " << can_id << endl;
    while(sample_ids.size() < sample_num)
    {
        sample_ids.insert(can_id);
        // size_t r_id = sample_ids.size() - 1; 
        for(size_t i = 0; i < pt_num; ++i)
        {
            double dist = squaredDistance(inpts, i, can_id);
            // cout << "idst  id " << i << " dist " <<dist << endl;
            // cout << "min_dist_vec " << min_dist_vec[i] << endl;
            min_dist_vec[i] = min_dist_vec[i] < dist ? min_dist_vec[i] : dist;
        }
        can_id = min_dist_vec.index_max();
    }
}

// }

