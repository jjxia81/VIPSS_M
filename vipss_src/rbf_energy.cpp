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

void SavePointsAndNormals(const std::vector<double>& points, 
                          const std::vector<double>& normals,  
                          const std::string& out_path, double scale =2.0)
{
    std::vector<double> new_pts;
    std::vector<unsigned int> edge2vertices;
    for(size_t i = 0; i < points.size()/3; ++i)
    {
        new_pts.push_back(points[i*3]);
        new_pts.push_back(points[i*3 + 1]);
        new_pts.push_back(points[i*3 + 2]);
        new_pts.push_back(points[i*3]     + normals[i*3] * scale);
        new_pts.push_back(points[i*3 + 1] + normals[i*3 + 1] * scale);
        new_pts.push_back(points[i*3 + 2] + normals[i*3 + 2] * scale);
        edge2vertices.push_back(i*2 + 0);
        edge2vertices.push_back(i*2 + 1);
    }
    // std::string out_path = out_dir_ + "_rbf_normal_line";
    writeObjFile_line(out_path, new_pts, edge2vertices);
}


void RBF_Energy_PARA::loadYamlFile(const std::string& yaml_path)
{
    YAML::Node config = YAML::LoadFile(yaml_path);
    use_gradient = config["use_gradient"].as<bool>();
    cout << "use_gradient " << use_gradient << endl;
    double surface_w = config["surface_weight"].as<double>();
    e_lambda = config["smooth_lambda"].as<double>();
    e_beta = config["normal_beta"].as<double>();
    e_lambda = e_lambda / surface_w;
    e_beta = e_beta / surface_w;
    n_voxel_line = config["volumn_dim"].as<int>();
    enable_debug = config["debug"].as<bool>();
    out_dir = config["out_dir"].as<std::string>();
    use_scan_data = config["use_scan_data"].as<bool>();
    if(!use_scan_data)
    {
        mesh_points_path = config["points_dir"].as<std::string>();
        if(use_gradient)
        {
            gradients_path = config["graidents_dir"].as<std::string>();
        }
        // cout << "mesh_points_path " << mesh_points_path << endl;
    } else{
        std::string data_id = config["data_id"].as<std::string>();
        std::string data_dir = config["data_dir"].as<std::string>();
        std::string pt_normal_folder = config["pt_normal_folder"].as<std::string>();
        std::string pt_gradient_folder = config["pt_gradient_folder"].as<std::string>();
        std::string pt_normal_file = config["pt_normal_file"].as<std::string>();
        std::string pt_gradient_file = config["pt_gradient_file"].as<std::string>();
        mesh_points_path = data_dir + data_id + pt_normal_folder + pt_normal_file;
        if(use_gradient)
        {
            gradients_path = data_dir + data_id + pt_gradient_folder + pt_gradient_file;
        }
    }
    
}

RBF_Energy::RBF_Energy()
{
    InitPara();
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

    para.Kernal = Kernal;
    para.polyDeg = polyDeg;
    para.sigma = sigma;
    para.rangevalue = rangevalue;
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

    // rbf_core_->normals = normals_;
    // rbf_core_->newnormals = normals_;
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
    use_confidence_ = true;
    solve_with_Eigen_ = false;
    e_lambda_ = 0.0;
    e_beta_ = 0.1;
    n_voxel_line_ = 50;
}

void RBF_Energy::SetEnergyParameters(const RBF_Energy_PARA& rbf_e_para)
{
    enable_debug_ = rbf_e_para.enable_debug;
    // is_surfacing_ = rbf_e_para;
    use_gradient_ = rbf_e_para.use_gradient;
    solve_with_Eigen_ = false;
    e_lambda_ = rbf_e_para.e_lambda;
    e_beta_ = rbf_e_para.e_beta;
    n_voxel_line_ = rbf_e_para.n_voxel_line;

    pt_path_ = rbf_e_para.mesh_points_path;
    if(use_gradient_)
    {
        gd_path_ = rbf_e_para.gradients_path;
    }
    out_dir_ = rbf_e_para.out_dir;
}

void RBF_Energy::LoadPtsGradients(const std::string& gradient_path)
{
    gradients_.clear();
    tangents_.clear();
    readPLYFile(gradient_path, gradients_, tangents_);
    std::string out_path = out_dir_ + "_ori_gradient_line";
    SavePointsAndNormals(pts_, gradients_, out_path);
}

void RBF_Energy::LoadPtsAndNormals(const std::string& ply_path)
{
    pts_.clear();
    normals_.clear();
    readPLYFile(ply_path, pts_, normals_);
    std::string out_path = out_dir_ + "_ori_normal_line";
    SavePointsAndNormals(pts_, normals_, out_path);
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
    arma::mat lens(pt_n_, 1);
    for(size_t i = 0; i < pt_n_; ++i)
    {
        arma::vec grad = {gradients_[3*i], gradients_[3*i+1], gradients_[3*i+2]};
        arma::vec normal = {normals_[3*i], normals_[3*i + 1], normals_[3*i + 2]};
        arma::vec tangent = {tangents_[3*i], tangents_[3*i + 1], tangents_[3*i + 2]};
        tangent = arma::normalise(tangent);
        double len = arma::norm(grad);
        // std::cout << " --- " << grad[0] <<" " << grad[1] <<" " <<grad[2] << " " << len << std::endl;
        // std::cout <<"
        double tang_proj = arma::dot(grad, tangent);
        grad = grad - tang_proj * tangent;
        grad_lens_[i] = len;
        // std::cout << " --- " << grad_lens_[pt_n_] << std::endl;
        arma::vec grad_normalized = arma::normalise(grad);
        // std::cout << " --- g n " << grad_normalized[0] <<" " << grad_normalized[1] <<" " <<grad_normalized[2] << " " << len << std::endl; 
        if (arma::dot(grad_normalized, normal) < 0)
        {
            grad_normalized = -1 * grad_normalized;
        }
        double dot_v = arma::dot(grad_normalized, normal);
        if (dot_v < 0.7)
        {
            if (dot_v < 0.5)
            {
                grad_normalized = normal;
                // grad_normalized = (grad_normalized + normal) / 2.0;
            } else {
                grad_normalized = (grad_normalized + normal) / 2.0;
            }
        }
        gradients_[3*i] = grad_normalized[0];
        gradients_[3*i + 1] = grad_normalized[1];
        gradients_[3*i + 2] = grad_normalized[2];
    }
    grad_lens_.save("grad_lens_.txt", arma::raw_ascii);

    // SaveMatrix(grad_lens_, "grad_lens_.txt");
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
    // alpha_s_ = (arma::diagmat(arma::ones(pt_n_, pt_n_)));
    // alpha_g_ = (arma::diagmat(arma::ones(3 * pt_n_, 3 * pt_n_)));
    alpha_s_ = arma::mat(1, pt_n_).fill(1.0);
    alpha_g_ = arma::mat(1, 3 * pt_n_).fill(1.0);
    if(!use_gradient_) return;
    cout << " start to fill gradients !" << std::endl;
    double max_len = arma::max(grad_lens_);
    arma::mat confidence = grad_lens_ / max_len;
    // SaveMatrix(confidence, "confidence.txt");
    for(size_t i =0; i < pt_n_; ++i)
    {
        alpha_s_[0, i] = max(confidence[i], 0.01);
        for(size_t j = 0; j < 3; ++j)
        {
            alpha_g_[0, i * 3 + j] = max(confidence[i], 0.01);
        }
    }
}

void RBF_Energy::BuildHessianMat()
{
    D_M_ = arma::zeros(pt_n_* 4 + 4, pt_n_*4 + 4);
    D_M_(0, 0, arma::size(pt_n_*4, pt_n_*4)) = rbf_core_->M;
    arma::mat fs_t =  F_s_.t();
    arma::mat fg_t =  F_g_.t();
    if(use_confidence_)
    {
        fs_t = fs_t.each_row() % alpha_s_;
        fg_t = fg_t.each_row() % alpha_g_;
    } 
    H_ = fs_t * F_s_ + e_lambda_ * D_M_ + e_beta_ * fg_t * F_g_;
    // SaveMatrix(H_, "h_mat.txt");
    // H_ = F_s_.t() * alpha_s_ * F_s_ + e_lambda_ * D_M_ + e_beta_ * F_g_.t() * alpha_g_ * F_g_;
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
    cout << "SolveReducedLinearSystem time: " <<  (re_time = std::chrono::nanoseconds(t11 - t10).count()/1e9) <<endl;
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

void RBF_Energy::SetOutdir(const std::string& dir)
{
    out_dir_ = dir;
}

void RBF_Energy::RunTest(const std::string& ply_path, const std::string gradient_path, bool is_gradient)
{
    if(is_gradient) use_gradient_ = true;
    pt_path_ = ply_path;
    gd_path_ = gradient_path;
    RunTest();
}

void RBF_Energy::RunTest()
{
    double re_time;
    cout<<"start solve rbf matrix "<<endl;
    
    std::cout << "use_gradient_ " << use_gradient_ << std::endl;
    auto t1 = Clock::now();
    auto t2 = Clock::now();

    SetPts(pt_path_);
    std::cout << "use_gradient_ " << use_gradient_ << std::endl;
    if(use_gradient_)
    {
        LoadPtsGradients(gd_path_);
        std::cout <<"Load gradients succeed !" << std::endl;
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
        // rbf_core_->InitNormal();
        // rbf_core_->OptNormal(0);
        rbf_core_->Surfacing(0,n_voxel_line_);
        // std::string out_path = "../out_mesh";
        rbf_core_->Write_Surface(out_dir_);
        EstimateRBFNormals();
        // rbf_core_->Update_Newnormals();
        // rbf_core_->Write_Hermite_NormalPrediction(out_dir_ + "_normal", 1);
    }
}


void RBF_Energy::EstimateRBFNormals()
{
    // std::vector<arma::vec3> estimated_normals;
    std::vector<double> estimated_normals;
    double delta = 0.00001;
    for(size_t i = 0; i < pt_n_; ++i)
    {
        const R3Pt pt(pts_[i*3], pts_[i*3 + 1], pts_[i*3 + 2]);
        double dist = rbf_core_->Dist_Function(pt);
        const R3Pt pt_x(pts_[i*3] + delta, pts_[i*3 + 1], pts_[i*3 + 2]);
        double dist_x = rbf_core_->Dist_Function(pt_x);
        double dx = -1.0 * (dist_x - dist) / delta;

        const R3Pt pt_y(pts_[i*3], pts_[i*3 + 1] + delta, pts_[i*3 + 2]);
        double dist_y = rbf_core_->Dist_Function(pt_y);
        double dy = -1.0 * (dist_y - dist) / delta;

        const R3Pt pt_z(pts_[i*3], pts_[i*3 + 1], pts_[i*3 + 2] + delta);
        double dist_z = rbf_core_->Dist_Function(pt_z);
        double dz = -1.0 * (dist_z - dist) / delta;

        double n_len = sqrt(dx * dx + dy * dy + dz * dz);
        estimated_normals.push_back(dx/n_len );
        estimated_normals.push_back(dy/n_len );
        estimated_normals.push_back(dz/n_len);
    }
    // rbf_core_->newnormals = estimated_normals;
    std::string out_path = out_dir_ + "_rbf_normal_line";
    SavePointsAndNormals(pts_, estimated_normals, out_path);
}

