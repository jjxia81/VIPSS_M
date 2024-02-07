#include "armadillo"

#include "vipss_src/rbfcore.h"
#include "vipss_src/rbf_energy.h"
#include <iostream>
#include <fstream>
// #include <jsoncpp/json/json.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include "vipss_src/normal_opt.h"
#include "sample.h"

namespace fs = std::filesystem;

void test()
{
    std::string pt_file = "../data/cases/wireframes/doghead/doghead_in_ptn.ply";
    std::vector<double> pts;
    std::vector<double> normals;
    readPLYFile(pt_file, pts, normals);

    std::vector<double> out_pts;
    std::vector<double> auxi_pts;
    PTSample::FurthestSamplePointCloud(pts, 48, out_pts, auxi_pts);

    std::string out_path = "sample.ply";
    writePLYFile(out_path, out_pts);
}


int main(int argc, char* argv[])
{
    // test();
    // return 0;
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
    cout << " loadYamlFile succeed! " << endl;
    if (rbf_e_para.optimize_normal)
    {
        NormalOptimizer n_opt;
        n_opt.e_para_ = rbf_e_para;
        n_opt.OptimizeNormalSigns();
    } else {
        RBF_Energy rfb_energy;
        rfb_energy.SetEnergyParameters(rbf_e_para);
        rfb_energy.SolveVipss();
        // rfb_energy.RunTest();
    }
    return 0;
}