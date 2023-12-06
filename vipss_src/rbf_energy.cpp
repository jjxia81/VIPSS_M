#include "rbf_energy.h"
#include <chrono>
#include "../include/Eigen/Dense"
// #include "../include/Eigen/RequiredModuleName"

typedef std::chrono::high_resolution_clock Clock;

Eigen::MatrixXd ConvertArmaMatToEigenMat(const arma::mat& in_mat)
{
    int rows = in_mat.n_rows;
    int cols = in_mat.n_cols;

    Eigen::MatrixXd e_mat(rows, cols);
    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
        {
            e_mat(i, j) = in_mat[i, j];
        }
    }
    return e_mat;
}

void SaveMatrix(const arma::mat& in_mat, const std::string& path)
{
    // rbf_core_->M.save("../mat/M.txt", arma::raw_ascii);
    in_mat.save(path, arma::raw_ascii);
    return;
}

RBF_Energy::RBF_Energy()
{

}

RBF_Energy::~RBF_Energy()
{

}

void RBF_Energy::SetRBFPara(){

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
    this->SetRBFPara();
    rbf_para_.user_lamnbda = 1.0;

    // readXYZ(infilename,Vs);
    rbf_core_->InjectData(pts_, rbf_para_);
    // rbf_core_->BuildK(rbf_para_);
    rbf_core_->Set_HermiteRBF(pts_);
    std::cout <<"Finish init rbf core !" << std::endl;
}

void RBF_Energy::SetPts(const std::string& ply_path)
{
    LoadPtsAndNormals(ply_path);
}

void RBF_Energy::DebugLog(const std::string& log_str)
{
    if (enable_debug_)
    {
        std::cout << log_str << std::endl;
    }
}

void RBF_Energy::InitPara()
{
    enable_debug_ = true;
    is_surfacing_ = true;
    use_gradient_ = false;
    solve_with_Eigen_ = false;
    e_lambda_ = 5.0;
    e_beta_ = 0.3;
    n_voxel_line_ = 50;
}

void RBF_Energy::LoadPtsGradients(const std::string& gradient_path)
{
    readPLYFile(gradient_path, gradients_, tangents_);
}

void RBF_Energy::LoadPtsAndNormals(const std::string& ply_path)
{
    pts_.clear();
    gradients_.clear();
    readPLYFile(ply_path, pts_, normals_);
    pt_n_ = pts_.size() / 3;
    if(! use_gradient_)
    {
        gradients_ = normals_;
    }
}

void RBF_Energy::ProcessGradientsAndConfidenceMat()
{
    if(! use_gradient_) return;
    grad_lens_.resize(pt_n_);
    for(size_t i = 0; i < pt_n_; ++i)
    {
        arma::vec grad = {gradients_[3*i], gradients_[3*i+1], gradients_[3*i+2]};
        arma::vec normal = {normals_[3*i], normals_[3*i + 1], normals_[3*i + 2]};
        arma::vec tangent = {tangents_[3*i], tangents_[3*i + 1], tangents_[3*i + 2]};
        tangent = arma::normalise(tangent);
        double len = arma::norm(grad);
        double tang_proj = arma::dot(grad, tangent);
        grad = grad - tang_proj * tangent;
        grad_lens_[pt_n_] = len;
        arma::vec grad_normalized = arma::normalise(grad);
        if (arma::dot(grad_normalized, normal) < 0)
        {
            grad_normalized = -1 * grad_normalized;
        }
        // if (arma::dot(grad_normalized, normal) < 0.6)
        // {
        //     grad_normalized = (grad_normalized + normal) / 2.0;
        // }
        gradients_[3*i] = grad_normalized[0];
        gradients_[3*i + 1] = grad_normalized[1];
        gradients_[3*i + 2] = grad_normalized[2];
    }
}

void RBF_Energy::BuildMatrixB()
{
    if(gradients_.empty()) return;
    // B_.set_size(pt_n_* 3, 1);
    arma::mat G_mat = arma::zeros(1, pt_n_* 3);
    for(size_t i =0; i < pt_n_; ++i)
    {
        for(size_t j =0 ; j < 3; ++j)
        {
            G_mat(0, j * pt_n_ + i) = gradients_[i * 3 + j];
        }
    }
    B_ =  e_beta_ * (G_mat * F_g_);
}

void RBF_Energy::SolveConditionMatSVD()
{
    arma::mat U;
    arma::colvec sigma;
    arma::mat V;
    arma::svd(U, sigma, V, C_a_);
    SVD_V_ = V(0, 4, arma::size(pt_n_* 4 + 4, pt_n_* 4));
    // SVD_V_ = SVD_V_.t();
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
    auto N0 = rbf_core_->N(0, 0, arma::size(pt_n_, 4));
    F_s_ = arma::ones(pt_n_, pt_n_* 4 + 4);
    F_s_(0, 0, arma::size(pt_n_, pt_n_)) = M00;
    F_s_(0, pt_n_, arma::size(pt_n_,3 * pt_n_)) = M01;
    F_s_(0, 4*pt_n_, arma::size(pt_n_,4)) = N0;
    std::string log = "ConstructSurfaceTermMat succeed !";
    DebugLog(log);
}

void RBF_Energy::BuildGradientTermMat()
{
    if(rbf_core_ == nullptr)  return;
    auto M10 = rbf_core_->M(pt_n_, 0, arma::size(3*pt_n_, pt_n_));
    auto M11 = rbf_core_->M(pt_n_, pt_n_, arma::size(3*pt_n_,3 * pt_n_));
    arma::mat N1 = rbf_core_->N(pt_n_, 0, arma::size(3*pt_n_, 4));
    F_g_ = arma::zeros(3*pt_n_, pt_n_* 4 + 4);
    F_g_(0, 0, arma::size(3*pt_n_, pt_n_)) = M10;

    F_g_(0, pt_n_, arma::size(3*pt_n_,3 * pt_n_)) = M11;
    // cout << " N1 size " << N1.size() << endl;
    F_g_(0, 4*pt_n_, arma::size(3*pt_n_,4)) = N1;
    std::string log = "ConstructGradientTermMat succeed !";
    DebugLog(log);
}

void RBF_Energy::BuildConfidenceMat()
{
    alpha_s_ = arma::diagmat(arma::ones(pt_n_, pt_n_));
    alpha_g_ = arma::diagmat(arma::ones(3 * pt_n_, 3 * pt_n_));
    if(!use_gradient_) return;

    double max_len = arma::max(grad_lens_);
    arma::vec confidence = grad_lens_ / max_len;
    
    if(use_gradient_)
    {
        for(size_t i =0; i < pt_n_; ++i)
        {
            alpha_s_[i, i] = confidence[i];
            for(size_t j = 0; j < 3; ++j)
            {
                alpha_g_[i * 3 + j, i * 3 + j] = confidence[i];
            }
        }
    }
}

void RBF_Energy::BuildHessianMat()
{
    D_M_ = arma::zeros(pt_n_* 4 + 4, pt_n_*4 + 4);
    D_M_(0, 0, arma::size(pt_n_*4, pt_n_*4)) = rbf_core_->M;
    H_ = F_s_.t() * alpha_s_ * F_s_ + e_lambda_ * D_M_ + e_beta_ * F_g_.t() * alpha_g_ * F_g_;
    // H_ = F_s_.t() * F_s_; 
}

void RBF_Energy::ReduceBAndHMatWithSVD()
{
    B_ = B_ * SVD_V_;
    H_ = SVD_V_.t() * H_ * SVD_V_; 
}

void RBF_Energy::BuildEnergyMatrixAndSolve()
{
    if(pts_.empty() || gradients_.empty())
    {
        std::cout <<"ConstructEnergyMatrix failed! Empty points data or normal data! " << std::endl;
        return;
    }
    double re_time;
    auto t1 = Clock::now();
    ProcessGradientsAndConfidenceMat();
    auto t2 = Clock::now();
    cout << "ProcessGradients time: " <<  (re_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) <<endl;
    BuildGradientTermMat();
    auto t3 = Clock::now();
    cout << "BuildGradientTermMat time: " <<  (re_time = std::chrono::nanoseconds(t3 - t2).count()/1e9) <<endl;
    BuildSurfaceTermMat();
    auto t4 = Clock::now();
    cout << "BuildSurfaceTermMat time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
    BuildConditionMat();
    auto t5 = Clock::now();
    cout << "BuildConditionMat time: " <<  (re_time = std::chrono::nanoseconds(t5 - t4).count()/1e9) <<endl;
    BuildMatrixB();
    auto t6 = Clock::now();
    cout << "BuildMatrixB time: " <<  (re_time = std::chrono::nanoseconds(t6 - t5).count()/1e9) <<endl;
    SolveConditionMatSVD();
    auto t7 = Clock::now();
    cout << "SolveConditionMatSVD time: " <<  (re_time = std::chrono::nanoseconds(t7 - t6).count()/1e9) <<endl;
    BuildConfidenceMat();
    auto t8 = Clock::now();
    cout << "BuildConfidenceMat time: " <<  (re_time = std::chrono::nanoseconds(t8 - t7).count()/1e9) <<endl;
    BuildHessianMat();
    auto t9 = Clock::now();
    cout << "BuildHessianMat time: " <<  (re_time = std::chrono::nanoseconds(t9 - t8).count()/1e9) <<endl;
    ReduceBAndHMatWithSVD();
    auto t10 = Clock::now();
    cout << "ReduceBAndHMatWithSVD time: " <<  (re_time = std::chrono::nanoseconds(t10 - t9).count()/1e9) <<endl;
    SolveReducedLinearSystem();
    auto t11 = Clock::now();
    cout << "ReduceBAndHMatWithSVD time: " <<  (re_time = std::chrono::nanoseconds(t11 - t10).count()/1e9) <<endl;
    // SaveMatrix();
}



void RBF_Energy::SolveReducedLinearSystem()
{
    double re_time;
    if(solve_with_Eigen_)
    {
        // Eigen::MatrixXd e_H = ConvertArmaMatToEigenMat(H_);
        Eigen::Map<Eigen::MatrixXd> e_H(H_.memptr(), H_.n_rows, H_.n_cols);
        arma::mat B_t = B_.t();
        // Eigen::MatrixXd e_B_t = ConvertArmaMatToEigenMat(B_t);
        Eigen::Map<Eigen::MatrixXd> e_B_t(B_t.memptr(), B_t.n_rows, B_t.n_cols);
        
        auto t1 = Clock::now();
        Eigen::MatrixXd s_x = e_H.ldlt().solve(e_B_t);
        auto t2 = Clock::now();
        cout << "Eigen linear system solve time: " <<  (re_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) <<endl;
        // e_H.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(B_t);
		// Eigen::MatrixXd x_Eigen = lu_of_AtA.solve(B_t);

        // Eigen::MatrixXd s_x = e_H.colPivHouseholderQr().solve(e_B_t);

        // template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>()

        X_reduced_ = arma::mat(s_x.data(), s_x.rows(), s_x.cols());
        // SaveMatrix(X_reduced_, "./X_reduced_eigen.txt");   
        // X_ = SVD_V_ * X_reduced_;
        // rbf_core_->a = X_(0, 0, arma::size(pt_n_*4, 1));
        // rbf_core_->b = X_(pt_n_*4, 0, arma::size(4, 1));
    }  else{
        auto t3 = Clock::now();
        X_reduced_ = arma::solve(H_, B_.t(), arma::solve_opts::fast);
        auto t4 = Clock::now();
        cout << "Arma linear system solve time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
        // SaveMatrix(X_reduced_, "./X_reduced_arma.txt"); 
        X_ = SVD_V_ * X_reduced_;
        // cout <<"X size " << X_.size() << endl;
        rbf_core_->a = X_(0, 0, arma::size(pt_n_*4, 1));
        rbf_core_->b = X_(pt_n_*4, 0, arma::size(4, 1));
    }
}

void RBF_Energy::RunTest(const std::string& ply_path, const std::string& gradient_path, bool is_gradient)
{
    double re_time;
    cout<<"start solve rbf matrix "<<endl;
    if(is_gradient) use_gradient_ = true;
    auto t1 = Clock::now();
    InitPara();
    auto t2 = Clock::now();
    cout << "InitPara time: " <<  (re_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) <<endl;
    SetPts(ply_path);
    if(use_gradient_)
    {
        LoadPtsGradients(gradient_path);
    }
    auto t3 = Clock::now();
    cout << "SetPts time: " <<  (re_time = std::chrono::nanoseconds(t3 - t2).count()/1e9) <<endl;

    SetRBFPara();
    InitRBFCore();
    auto t4 = Clock::now();
    cout << "InitRBFCore time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
    
    BuildEnergyMatrixAndSolve();
    auto t5 = Clock::now();
    cout << "BuildEnergyMatrixAndSolve time: " <<  (re_time = std::chrono::nanoseconds(t5 - t4).count()/1e9) <<endl;
    t2 = Clock::now();
    cout << "Total matrix solve time: " <<  (re_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) <<endl;

    if(is_surfacing_){
        rbf_core_->Surfacing(0,n_voxel_line_);
        std::string out_path = "../out_mesh";
        rbf_core_->Write_Surface(out_path);
    }

}


