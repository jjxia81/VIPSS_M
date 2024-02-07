#pragma once
#include <string>


class RBF_Energy_PARA
{
public:
    bool enable_debug = false;
    bool is_surfacing = true;
    bool use_gradient = true;
    bool use_confidence = false;
    bool use_scan_data = false;
    bool solve_with_Eigen = false;
    bool save_inputs = true;
    bool save_estimate_normal = true;
    bool save_visualization = true;
    bool save_eigen_vec = false;
    bool enable_constriants = false;
    bool optimize_normal = false;
    int max_solve_iterate_num = 1;
    int save_iter = 0;
    double normal_weight_incre = 1.0;
    int volumn_dim = 50;
    double e_lambda = 0.1;
    double vipss_beta = 1.0;
    double e_beta = 0.1;
    double normal_iter_threshold = 0.000001;
    std::string mesh_points_path;
    std::string gradients_path;
    std::string out_dir;
    void loadYamlFile(const std::string& yaml_path);
};

