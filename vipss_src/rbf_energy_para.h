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
    bool vipss_apply_sample = false;
    int max_solve_iterate_num = 1;
    int vipss_incre_pt_num = 10;
    bool vipss_incre_debug = false;
    double vipss_incre_init_pt_num = 50.0; 
    int save_iter = 0;
    int vipss_incre_max_iter = 2;
    int vipss_type = 0;
    double normal_weight_incre = 1.0;
    int volumn_dim = 50;
    double e_lambda = 0.1;
    double v_lambda = 0.1;
    double vipss_beta = 1.0;
    double e_beta = 0.1;
    bool use_input_normal = false;
    // int kernel_type = 3;
    bool only_vipss_hrbf = false;

    double normal_iter_threshold = 0.000001;
    double vipss_incre_shreshold = 0.04;
    bool normalize_input_pts = false;
    bool use_compact_kernel = false;
    bool use_eigen_sparse = false;
    double compact_radius = 0.1;


    std::string mesh_points_path;
    std::string gradients_path;
    std::string out_dir;
    void loadYamlFile(const std::string& yaml_path);
};

