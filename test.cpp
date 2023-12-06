#include "armadillo"

#include "vipss_src/rbfcore.h"
#include "vipss_src/rbf_energy.h"
#include <iostream>


int main()
{
    // arma::mat M = arma::ones(10, 10);
    // std::cout << " M" << M(0, 0, arma::size(2, 2)) << std::endl;

    std::string ply_path = "../data/test/contour_ptnormal_6.ply";
    std::string grad_path = "../data/test/gradients_tangent_6.ply";
    // std::string ply_path = "../test0.ply";
    RBF_Energy rfb_energy;
    rfb_energy.RunTest(ply_path, grad_path, true);
}