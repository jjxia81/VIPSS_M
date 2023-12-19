#include "armadillo"

#include "vipss_src/rbfcore.h"
#include "vipss_src/rbf_energy.h"
#include <iostream>
#include <fstream>
// #include <jsoncpp/json/json.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace fs = std::filesystem;

void test()
{
    arma::mat X ;
    arma::vec beta ;

    beta.resize ( 3 ) ;

    beta (0) = 1.0 ;
    beta (1) = 3.0 ;
    beta (2) = 2.0 ;

    X.resize ( 3, 2 ) ;

    X (0,0) = 1.0 ;
    X (0,1) = 2.0 ;
    X (1,0) = 3.0 ;
    X (1,1) = 4.0 ;
    X (2,0) = 5.0 ;
    X (2,1) = 6.0 ;

    std::cout << X << std::endl ;
    // std::cout << X.each_row() + beta << std::endl ;
}

int main(int argc, char* argv[])
{
    // std::ifstream ifs;
    // ifs.open(argv[1]);

    RBF_Energy_PARA rbf_e_para;
    std::string config_path;
    if(argc <= 1)
    {
        config_path = "../config.yaml";
    } else {
        config_path = argv[1];
        cout <<" yaml file exists : " <<  argv[1] << endl;
    }
    cout <<  " config path : " <<  config_path << endl;
    if( !std::filesystem::exists(config_path))
    {
        cout << config_path <<" not existed!" << endl;
        return 0;
    }
    rbf_e_para.loadYamlFile(config_path);
    
    cout <<  " loadYamlFile succeed! " << endl;
    // YAML::Node config = YAML::LoadFile(argv[1]);
    // const std::string username = config["username"].as<std::string>();
    // const std::string password = config["password"].as<std::string>();
    // cout << username <<" " << password << endl;
    // arma::mat M = arma::ones(10, 10);
    // std::cout << " M" << M(0, 0, arma::size(2, 2)) << std::endl;
    // std::string data_id = "6007";
    // std::string data_dir = "/home/jjxia/Documents/projects/All_Elbow/contours_3/";
    // std::string pt_normal = "/contour_points/contour_ptnormal_6/";
    // std::string gradient_dir = "/contour_points/gradients_tangent_6/";
    // std::string ply_path = data_dir + data_id + pt_normal+ "contour_ptnormal_6.ply";
    // std::string grad_path =  data_dir + data_id + gradient_dir+"gradients_tangent_6.ply";
    // // std::string ply_path = "../test0.ply";
    // std::string out_dir = data_dir + data_id + "/" + data_id + "_mesh";
    // std::string out_dir = "../data/mesh";
    // std::string ply_path = "../data/test/contour_ptnormal_6.ply";
    // std::string grad_path = "../data/test/gradients_tangent_6.ply";
    RBF_Energy rfb_energy;
    rfb_energy.SetEnergyParameters(rbf_e_para);
    rfb_energy.RunTest();

    return 0;
    
}