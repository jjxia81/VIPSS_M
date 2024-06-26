#include "rbf_energy_para.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
using namespace std;

void RBF_Energy_PARA::loadYamlFile(const std::string& yaml_path)
{
    YAML::Node config = YAML::LoadFile(yaml_path);
    use_gradient = config["use_gradient"].as<bool>();
    cout << "use_gradient " << use_gradient << endl;
    //double surface_w = config["surface_weight"].as<double>();
    double surface_w = 1.0;
    e_lambda = config["smooth_lambda"].as<double>();
    v_lambda = config["vipss_lambda"].as<double>();
    e_beta = config["normal_beta"].as<double>();

    cout << "use_input_normal 0 0" << endl;
    e_lambda = e_lambda / surface_w;
    e_beta = e_beta / surface_w;
    volumn_dim = config["volumn_dim"].as<int>();
    enable_debug = config["debug"].as<bool>();
    out_dir = config["out_dir"].as<std::string>();
    use_scan_data = config["use_scan_data"].as<bool>();
    save_eigen_vec = config["save_eigen_vec"].as<bool>();
    save_estimate_normal = config["save_estimate_normal"].as<bool>();
    save_visualization = config["save_visualization"].as<bool>();
    save_inputs = config["save_inputs"].as<bool>();
    use_confidence = config["use_confidence"].as<bool>();
    enable_constriants = config["enable_constriants"].as<bool>();
    optimize_normal = config["optimize_normal"].as<bool>();
    max_solve_iterate_num = config["max_solve_iterate_num"].as<int>();
    normal_weight_incre = config["normal_weight_incre"].as<double>();
    normal_iter_threshold = config["normal_iter_threshold"].as<double>();
    save_iter = config["save_iter"].as<int>();
    vipss_beta = config["vipss_beta"].as<double>();

    cout << "use_input_normal 0 1" << endl;

    vipss_apply_sample = config["vipss_apply_sample"].as<bool>();
    vipss_incre_max_iter = config["vipss_incre_max_iter"].as<int>();
    vipss_incre_pt_num = config["vipss_incre_pt_num"].as<int>();
    vipss_incre_debug = config["vipss_incre_debug"].as<bool>();
    vipss_incre_init_pt_num = config["vipss_incre_init_pt_num"].as<double>();

    cout << "use_input_normal 0 2" << endl;
    vipss_incre_shreshold = config["vipss_incre_shreshold"].as<double>();
    normalize_input_pts = config["normalize_input_pts"].as<bool>();
    use_compact_kernel = config["use_compact_kernel"].as<bool>();
    vipss_use_compact_kernel = config["vipss_use_compact_kernel"].as<bool>();
    cout << "use_input_normal 0 3" << endl;
    // kernel_dist = config["kernel_dist"].as<double>();
    cout << "use_input_normal 0 4" << endl;
    vipss_type = config["vipss_type"].as<int>();
    cout << "use_input_normal 0 5" << endl;
    use_input_normal = config["use_input_normal"].as<bool>();

    only_vipss_hrbf = config["only_vipss_hrbf"].as<bool>();
    compact_radius = config["compact_radius"].as<double>();
    use_multilevel_vipss = config["use_multilevel_vipss"].as<bool>();
    use_vipss_RBF = config["use_vipss_RBF"].as<bool>();

    cout << "use_input_normal " << use_input_normal << endl;
    if (!use_scan_data)
    {
        mesh_points_path = config["points_dir"].as<std::string>();
        if (use_gradient)
        {
            gradients_path = config["graidents_dir"].as<std::string>();
        }
        // cout << "mesh_points_path " << mesh_points_path << endl;
    }
    else {
        std::string data_id = config["data_id"].as<std::string>();
        std::string data_dir = config["data_dir"].as<std::string>();
        //std::string pt_normal_folder = config["pt_normal_folder"].as<std::string>();
        //std::string pt_gradient_folder = config["pt_gradient_folder"].as<std::string>();
        std::string pt_normal_file = config["pt_normal_file"].as<std::string>();
        std::string pt_gradient_file = config["pt_gradient_file"].as<std::string>();
        mesh_points_path = data_dir + data_id  + pt_normal_file;
        if (use_gradient)
        {
            gradients_path = data_dir + data_id + pt_gradient_file;
        }
    }

}