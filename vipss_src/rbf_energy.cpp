#include "rbf_energy.h"


RBF_Energy::RBF_Energy()
{

}

RBF_Energy::~RBF_Energy()
{

}

void RBF_Energy::SetRBFPARA(){

    RBF_Paras& para = rbf_para_;
    RBF_InitMethod initmethod = Lamnbda_Search;

    RBF_Kernal Kernal = XCube;
    int polyDeg = 1;
    double sigma = 0.9;
    double rangevalue = 0.001;

    para.Kernal = Kernal;para.polyDeg = polyDeg;para.sigma = sigma;para.rangevalue = rangevalue;
    para.Hermite_weight_smoothness = 0.0;
    para.Hermite_ls_weight = 0;
    para.Hermite_designcurve_weight = 00.0;
    para.Method = RBF_METHOD::Hermite_UnitNormal;

    para.InitMethod = initmethod;
    para.user_lamnbda = 0;
    para.isusesparse = false;
    // return para;
}

void RBF_Energy::InitRBFCore()
{
    // std::vector<double>Vs;
    rbf_core_ = std::make_shared<RBF_Core>();
    this->SetRBFPARA();
    rbf_para_.user_lamnbda = 0.0;

    // readXYZ(infilename,Vs);
    rbf_core_->InjectData(pts_,rbf_para_);
    rbf_core_->BuildK(rbf_para_);
}

void RBF_Energy::SetPts(const std::string& ply_path)
{
    LoadPtsAndGradients(ply_path);
}

void RBF_Energy::TextLog(const std::string& log_str)
{
    if (enable_debug_)
    {
        std::cout << log_str << std::endl;
    }
}

void RBF_Energy::InitPara()
{
    enable_debug_ = true;
    e_lambda_ = 0;
    e_beta_ = 1.0;
}

void RBF_Energy::LoadPtsAndGradients(const std::string& ply_path)
{
    pts_.clear();
    gradients_.clear();
    readPLYFile(ply_path, pts_, gradients_);
}



void RBF_Energy::BuildMatrixB()
{
    if(gradients_.empty()) return;
    // B_.set_size(pt_n_* 3, 1);
    arma::mat G_mat = arma::zeros(1, pt_n_* 3);
    for(size_t i =0; i < gradients_.size(); ++i)
    {
        G_mat(0, i) = gradients_[i];
    }
    B_ = 2 * e_beta_ * (G_mat * F_g_);
}

void RBF_Energy::SolveConditionMatSVD()
{
    arma::mat U;
    arma::colvec sigma;
    arma::mat V_T;
    arma::svd(U, sigma, V_T, C_a_);
    SVD_V_ = V_T(4, 0, arma::size(pt_n_* 4, pt_n_* 4 + 4));
    SVD_V_ = SVD_V_.t();
}

void RBF_Energy::BuildConditionMat()
{
    C_a_ = arma::zeros(4, pt_n_ * 4 + 4);
    C_a_(0, 0, arma::size(4, pt_n_ *4)) = rbf_core_->N.t();
}

void RBF_Energy::BuildSurfaceTermMat()
{
    if(rbf_core_ == nullptr)  return;
    auto M00 = rbf_core_->M(0, 0, arma::size(pt_n_, pt_n_));
    auto M01 = rbf_core_->M(0, pt_n_, arma::size(pt_n_,3 * pt_n_));
    auto N0 = rbf_core_->N(0, 0, arma::size(pt_n_, 3));
    F_s_ = arma::ones(pt_n_, pt_n_* 4 + 4);
    F_s_(0, 0, arma::size(pt_n_, pt_n_)) = M00;
    F_s_(0, pt_n_, arma::size(pt_n_,3 * pt_n_)) = M01;
    F_s_(0, 4*pt_n_, arma::size(pt_n_,3)) = N0;
    std::string log = "ConstructSurfaceTermMat succeed !";
    TextLog(log);
}

void RBF_Energy::BuildGradientTermMat()
{
    if(rbf_core_ == nullptr)  return;
    auto M10 = rbf_core_->M(pt_n_, 0, arma::size(3*pt_n_, pt_n_));
    auto M11 = rbf_core_->M(pt_n_, pt_n_, arma::size(3*pt_n_,3 * pt_n_));
    auto N1 = rbf_core_->N(pt_n_, 0, arma::size(pt_n_, 3));
    F_g_ = arma::zeros(3*pt_n_, pt_n_* 4 + 4);
    F_g_(0, 0, arma::size(3*pt_n_, pt_n_)) = M10;
    F_g_(0, pt_n_, arma::size(3*pt_n_,3 * pt_n_)) = M11;
    F_g_(0, 4*pt_n_, arma::size(pt_n_,3)) = N1;
    std::string log = "ConstructGradientTermMat succeed !";
    TextLog(log);
}

void RBF_Energy::BuildConfidenceMat()
{
    
    confidence_mat_s_ = arma::diagmat(arma::ones(pt_n_, pt_n_));
    confidence_mat_g_ = arma::diagmat(arma::ones(3 * pt_n_, 3 * pt_n_));
}

void RBF_Energy::BuildHessianMat()
{
    D_M_ = arma::zeros(pt_n_* 4 + 4, pt_n_*4 + 4);
    D_M_(0, 0, arma::size(pt_n_*4, pt_n_*4)) = rbf_core_->M;
    H_ = F_s_.t() * F_s_ + e_lambda_ * D_M_ + e_beta_ * F_g_.t()*F_g_;
}

void RBF_Energy::ReduceBAndHMatWithSVD()
{
    B_ = B_ * SVD_V_;
    H_ = SVD_V_.t() * H_ * SVD_V_; 
}

void RBF_Energy::BuildEnergyMatrix()
{
    if(pts_.empty() || gradients_.empty())
    {
        std::cout <<"ConstructEnergyMatrix failed! Empty points data or normal data! " << std::endl;
        return;
    }
    BuildGradientTermMat();
    BuildSurfaceTermMat();
    BuildConditionMat();
    BuildMatrixB();
    SolveConditionMatSVD();
    BuildConfidenceMat();
    BuildHessianMat();
    ReduceBAndHMatWithSVD();
}

void RBF_Energy::SolveReducedLinearSystem()
{
    
}

void RBF_Energy::RunTest(const std::string& ply_path)
{
    InitPara();
    SetPts(ply_path);
    SetRBFPARA();
    InitRBFCore();

}
// arma::mat& RBF_Energy::getM() const
// {
//     if(rbf_core_ != nullptr)
//     {
//         return rbf_core_->M;
//     } 
// }
// arma::mat RBF_Energy::getM00() const;
// arma::mat RBF_Energy::getM01() const;
// arma::mat RBF_Energy::getM11() const;

