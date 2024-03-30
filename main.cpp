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
#include "rbf_octree.h"


namespace fs = std::filesystem;

//using namespace OrthoTree;

using std::array;
using std::vector;


void test_octree()
{
    //RBF_Octree octree;
    ///*std::string pt_file = "../data/vipss_cases/planck/data/planck2000.ply";*/
    std::string pt_file = "../data/cases/wireframes/doghead/doghead_in_ptn.ply";
    ///*octree.LoadOctreePts(pt_file);
    //octree.GetPts(0);*/
    CGAL_OCTREE::test_CGAL(pt_file);



}

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
    /*test_octree();
    return 0;*/
    // test();
    // return 0;
    // std::ifstream ifs;
    // ifs.open(argv[1]);
   
    RBF_Energy_PARA rbf_e_para;
    std::string config_path;
    if(argc <= 1)
    {
        //config_path = "../config.yaml";
        config_path = "E:/projects/VIPSS_M/config.yaml";
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
    /*rbf_compact_kernel_radius = rbf_e_para.compact_radius;
    rbf_compact_kernel_scale = 1.0 / pow(rbf_compact_kernel_radius, 5);*/
    cout << " loadYamlFile succeed! " << endl;
    if (rbf_e_para.optimize_normal)
    {
        NormalOptimizer n_opt;
        n_opt.e_para_ = rbf_e_para;
        n_opt.OptimizeNormalSigns();
    } else {
        RBF_Energy rbf_energy;
        rbf_energy.SetEnergyParameters(rbf_e_para);
        switch (rbf_e_para.vipss_type) {
            case 0:
                rbf_energy.RunTest();
                break;
            case 1:
                rbf_energy.SolveVipss();
                break;
            case 2:
                rbf_energy.SolveWithVipssOptNormal();
            default:
                break;
        }
    }
    return 0;
}