#include "rbf_energy.h"
#include <chrono>
#include "Eigen/Dense"
#include <cmath>
#include <limits>
#include "sample.h"
//#include "Eigen/SPQRSupport"
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
    // rbf_core_.M.save("../mat/M.txt", arma::raw_ascii);
    in_mat.save(path, arma::raw_ascii);
    return;
}

void SavePointsAndNormals(const std::vector<double>& points, 
                          const std::vector<double>& normals,  
                          const std::string& out_path, double scale =1.0)
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
    // std::string out_path = e_para_.out_dir + "_rbf_normal_line";
    writeObjFile_line(out_path, new_pts, edge2vertices);
}




RBF_Energy::RBF_Energy()
{
    // InitPara();
}

RBF_Energy::~RBF_Energy()
{
    
}

void RBF_Energy::SetRBFPara(){

    RBF_Paras& para = rbf_para_;
    RBF_InitMethod initmethod = Lamnbda_Search;

    RBF_Kernal Kernal = XCube;
    para.compact_radius = 1.0;
    switch (e_para_.kernel_type)
    {
    case 3:
        Kernal = XCube;
        break;
    case 4:
        Kernal = Compact;
        para.compact_radius = e_para_.compact_radius; 
        break;
    case 5:
        Kernal = Bump;
        break;
    default:
        break;
    }

    std::cout << "!!!kernel type : " << Kernal << std::endl;
    
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
    para.user_lamnbda = e_para_.e_beta;
    para.isusesparse = false;
    // return para;
}

void RBF_Energy::InitRBFCore()
{
    // std::vector<double>Vs;
    // rbf_core_ = std::make_shared<RBF_Core>();
    this->SetRBFPara();

    rbf_para_.user_lamnbda = e_para_.e_lambda;
    // std::cout << " rbf_para_ lambda ------------: " << rbf_para_.user_lamnbda << endl;
    // readXYZ(infilename,Vs);
     
    //rbf_core_.isuse_sparse = e_para_.use_sparse;
    rbf_core_.only_build_M = only_build_M_;
    rbf_core_.InjectData(pts_, rbf_para_);
    if (e_para_.use_sparse)
    {
        double kernel_dist = e_para_.compact_radius;
        rbf_core_.use_eigen_sparse =  e_para_.use_eigen_sparse;
        rbf_core_.Set_HermiteRBFSparse(pts_, kernel_dist);
    }
    else {
        rbf_core_.Set_HermiteRBF(pts_);
    }
    
    std::cout <<"Finish init rbf core !" << std::endl;
}

void RBF_Energy::SetPts(const std::string& ply_path)
{
    LoadPtsAndNormals(ply_path);
    // NormalizePts();
}

void RBF_Energy::SetPts(const std::vector<double>& points, std::vector<double>& normals)
{
    this->pts_ = points;
    this->normals_ = normals;
    this->gradients_ = normals;
    pt_n_ = pts_.size() / 3;
}

void RBF_Energy::DebugLog(const std::string& log_str)
{
    if (e_para_.enable_debug)
    {
        std::cout << log_str << std::endl;
    }
}



void RBF_Energy::SetEnergyParameters(const RBF_Energy_PARA& rbf_e_para)
{
    e_para_ = rbf_e_para;
}

void RBF_Energy::LoadPtsGradients(const std::string& gradient_path)
{
    gradients_.clear();
    tangents_.clear();
    readPLYFile(gradient_path, gradients_, tangents_);
    if (e_para_.save_inputs && e_para_.save_visualization)
    {
        std::string out_path = e_para_.out_dir + "_ori_gradient_line";
        SavePointsAndNormals(pts_, gradients_, out_path, 1.0);
    }
}

void RBF_Energy::LoadPtsAndNormals(const std::string& ply_path)
{
    pts_.clear();
    normals_.clear();
    readPLYFile(ply_path, pts_, normals_);
    if (e_para_.save_inputs && e_para_.save_visualization)
    {
        std::string out_path = e_para_.out_dir + "_ori_normal_line";
        SavePointsAndNormals(pts_, normals_, out_path);
    }
 
    pt_n_ = pts_.size() / 3;
    if(! e_para_.use_gradient)
    {
        gradients_ = normals_;
    }
}

void RBF_Energy::NormalizePts()
{
    double DMIN = std::numeric_limits<double>::min();
    double DMAX = std::numeric_limits<double>::max();
    double max_x = DMIN, max_y = DMIN, max_z = DMIN;
    double min_x = DMAX, min_y = DMAX, min_z = DMAX;
    for (size_t i = 0; i < pts_.size() / 3; ++i)
    {
        max_x = std::max(pts_[3 * i], max_x);
        max_y = std::max(pts_[3 * i + 1], max_y);
        max_z = std::max(pts_[3 * i + 2], max_z);

        min_x = std::min(pts_[3 * i], min_x);
        min_y = std::min(pts_[3 * i + 1], min_y);
        min_z = std::min(pts_[3 * i + 2], min_z);
    }
    double delt_x = (max_x - min_x)/2.0 ;
    double delt_y = (max_y - min_y)/2.0 ;
    double delt_z = (max_z - min_z)/2.0 ;

    double scale = std::max(std::max(delt_x, delt_y), delt_z);

    double origin_x = (max_x + min_x) / 2;
    double origin_y = (max_y + min_y) / 2;
    double origin_z = (max_z + min_z) / 2;

    for (size_t i = 0; i < pts_.size() / 3; ++i)
    {
        pts_[3 * i] = (pts_[3 * i] - origin_x) / scale;
        pts_[3 * i + 1] = (pts_[3 * i + 1] - origin_y) / scale;
        pts_[3 * i + 2] = (pts_[3 * i + 2] - origin_z) / scale; 
    }

    // std::string normalized_pts_path = e_para_.out_dir + "_normalized_pts.ply";
    // writePLYFile_VN(normalized_pts_path, pts_, gradients_);
}

void RBF_Energy::ProcessGradientsAndConfidenceMat()
{
    if(! e_para_.use_gradient) return;
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
        // if (dot_v < 0.7)
        // {
        //     if (dot_v < 0.5)
        //     {
        //         grad_normalized = normal;
        //         // grad_normalized = (grad_normalized + normal) / 2.0;
        //     } else {
        //         grad_normalized = (grad_normalized + normal) / 2.0;
        //     }
        // }
        gradients_[3*i] = grad_normalized[0];
        gradients_[3*i + 1] = grad_normalized[1];
        gradients_[3*i + 2] = grad_normalized[2];
    }
    pt_colors_.clear();
    if(e_para_.use_confidence)
    {
        double max_len = arma::max(grad_lens_);
        confidence_ = grad_lens_ / max_len;
        for(auto col_val : confidence_)
        {
            uint8_t red = (uint8_t)(255 * col_val);
            pt_colors_.push_back(red);
            pt_colors_.push_back(0);
            pt_colors_.push_back(0);
        }

        double confidence_sum = arma::accu(confidence_);
        confidence_ = grad_lens_ / confidence_sum;

    }

    // grad_lens_.save("grad_lens_.txt", arma::raw_ascii);
    // cout <<" --------------------" << e_para_.save_inputs << endl;
    if(e_para_.save_inputs)
    {
        std::string input_ptn_dir = e_para_.out_dir + "_in_ptn";
        writePLYFile_VN(input_ptn_dir, pts_, gradients_);
        std::cout << " !!!!!!!!!!!!save " << input_ptn_dir << "succeed !" << endl;
        
        if(e_para_.use_confidence)
        {
            // cout << " cal confidence succeed !" << endl;
            std::vector<double> new_gradients;
            for(size_t i = 0; i < pt_n_; ++i)
            {
                for(size_t j = 0; j < 3; ++j)
                {
                    double val = gradients_[3*i +j] * confidence_[i];
                    new_gradients.push_back(val);
                }
            }
            input_ptn_dir = input_ptn_dir + "_line";
            double scale = 0.001;
            SavePointsAndNormals(pts_, new_gradients, input_ptn_dir, scale);
        }
    }
}

void RBF_Energy::SaveFuncInputs()
{
    if(e_para_.save_inputs)
    {
        std::string input_ptn_dir = e_para_.out_dir + "_in_ptn";
        writePLYFile_VN(input_ptn_dir, pts_, gradients_);
        if(e_para_.use_gradient)
        {
            std::string input_ptn_color_dir = e_para_.out_dir + "_in_ptn_color";
            writePLYFile_VN_CO(input_ptn_color_dir, pts_, gradients_, pt_colors_);
        }
        
        std::cout << " save " << input_ptn_dir << "succeed !" << endl;
        input_ptn_dir = input_ptn_dir + "_line";
        if(e_para_.use_confidence)
        {
            // cout << " cal confidence succeed !" << endl;
            std::vector<double> new_gradients;
            for(size_t i = 0; i < pt_n_; ++i)
            {
                for(size_t j = 0; j < 3; ++j)
                {
                    double val = gradients_[3*i +j] * confidence_[i];
                    new_gradients.push_back(val);
                }
            }
            SavePointsAndNormals(pts_, new_gradients, input_ptn_dir);
        } else {
            SavePointsAndNormals(pts_, gradients_, input_ptn_dir);
        }
        //SaveMatrix(confidence_, "confidence_mat.txt");
    }
}

void RBF_Energy::BuildMatrixB()
{
    if(gradients_.empty()) return;
    // B_.set_size(pt_n_* 3, 1);
    G_mat_ = arma::zeros(1, pt_n_* 3);
    for(size_t i =0; i < pt_n_; ++i)
    {
        for(size_t j =0 ; j < 3; ++j)
        {
            G_mat_(0, j * pt_n_ + i) = gradients_[i * 3 + j];
        }
    }
    double beta =  e_para_.e_beta;
    // arma::mat g_mat_alpha = G_mat_.each_row() % alpha_g_;
    // B_ = beta * (g_mat_alpha * F_g_);
    B_ = beta * (G_mat_ * F_g_);
    

    //if (e_para_.use_confidence)
    //{
    //    //cout << "G_mat " << G_mat_.size() << endl;
    //    G_mat_ = G_mat_.each_row() % alpha_g_;
    //    
    //    B_ =  beta * (G_mat_ * F_g_);
    //} else {
    //    arma::mat g_mat_alpha = G_mat_.each_row() % alpha_g_;
    //    B_ =  beta * (g_mat_alpha * F_g_);
    //}
}

void RBF_Energy::SolveConditionMatSVD()
{
    arma::mat U;
    arma::colvec sigma;
    // arma::mat V;
    cout << "start SVD of constraint matrix ......" << endl;
    arma::svd(U, sigma, SVD_V_, C_a_);
    cout << "finished SVD of constraint matrix !" << endl;
    
    SVD_V_ = std::move(SVD_V_(0, 4, arma::size(pt_n_* 4 + 4, pt_n_* 4)));
    // SVD_V_ = SVD_V_.t();
    if(e_para_.save_eigen_vec)
    {
        std::string dir = "./eigen_vec_mat.txt";
        SaveMatrix(SVD_V_, dir);
    }
    C_a_.clear();
}

void RBF_Energy::BuildConditionMat()
{
    C_a_ = arma::zeros(4, pt_n_ * 4 + 4);
    C_a_(0, 0, arma::size(4, pt_n_ *4)) = rbf_core_.N.t();
}

void RBF_Energy::BuildSurfaceTermMat()
{
    // if(rbf_core_ == nullptr)  return;
    // auto M00 = rbf_core_.M(0, 0, arma::size(pt_n_, pt_n_));
    // auto M01 = rbf_core_.M(0, pt_n_, arma::size(pt_n_,3 * pt_n_));
    // auto N0 = rbf_core_.N(0, 0, arma::size(pt_n_, 4));
    F_s_ = arma::ones(pt_n_, pt_n_* 4 + 4);
    // F_s_(0, 0, arma::size(pt_n_, pt_n_)) = M00;
    // F_s_(0, pt_n_, arma::size(pt_n_,3 * pt_n_)) = M01;
    // F_s_(0, 4*pt_n_, arma::size(pt_n_,4)) = N0;
    F_s_(0, 0, arma::size(pt_n_, pt_n_)) = rbf_core_.M(0, 0, arma::size(pt_n_, pt_n_));
    F_s_(0, pt_n_, arma::size(pt_n_,3 * pt_n_)) = rbf_core_.M(0, pt_n_, arma::size(pt_n_,3 * pt_n_));
    F_s_(0, 4*pt_n_, arma::size(pt_n_,4)) = rbf_core_.N(0, 0, arma::size(pt_n_, 4));

    std::string log = "ConstructSurfaceTermMat succeed !";
    DebugLog(log);
}

void RBF_Energy::BuildGradientTermMat()
{
    //std::cout << " rbf core " << std::endl;
    /*if(rbf_core_ == nullptr)  return;*/
    // auto M10 = rbf_core_.M(pt_n_, 0, arma::size(3*pt_n_, pt_n_));
    // auto M11 = rbf_core_.M(pt_n_, pt_n_, arma::size(3*pt_n_,3 * pt_n_));
    // arma::mat N1 = rbf_core_.N(pt_n_, 0, arma::size(3*pt_n_, 4));
    F_g_ = arma::zeros(3*pt_n_, pt_n_* 4 + 4);
    // F_g_(0, 0, arma::size(3*pt_n_, pt_n_)) = M10;
    // F_g_(0, pt_n_, arma::size(3*pt_n_,3 * pt_n_)) = M11;
    // F_g_(0, 4*pt_n_, arma::size(3*pt_n_,4)) = N1;
    F_g_(0, 0, arma::size(3*pt_n_, pt_n_)) = rbf_core_.M(pt_n_, 0, arma::size(3*pt_n_, pt_n_));
    F_g_(0, pt_n_, arma::size(3*pt_n_,3 * pt_n_)) = rbf_core_.M(pt_n_, pt_n_, arma::size(3*pt_n_,3 * pt_n_));
    F_g_(0, 4*pt_n_, arma::size(3*pt_n_,4)) = rbf_core_.N(pt_n_, 0, arma::size(3*pt_n_, 4));
    std::string log = "ConstructGradientTermMat succeed !";
    DebugLog(log);
}


void RBF_Energy::BuildConfidenceMat()
{
    double fill_w = 1.0 / pt_n_;
    //double fill_w = 1.0;
    alpha_s_ = arma::mat(1, pt_n_).fill(fill_w);
    alpha_g_ = arma::mat(1, 3 * pt_n_).fill(fill_w);
    if(!e_para_.use_gradient) return;
    if(!e_para_.use_confidence) return;
    DebugLog(" start to fill gradients !");
    //double minimum_w = 0.000001;
    for(size_t i =0; i < pt_n_; ++i)
    {
        //alpha_s_[0, i] = max(confidence_[i], minimum_w);
        alpha_s_[0, i] = confidence_[i]; 
        for(size_t j = 0; j < 3; ++j)
        {
            //alpha_g_[0, j * pt_n_ + i] = max(confidence_[i], minimum_w);
            alpha_g_[0, j * pt_n_ + i] = (confidence_[i]);
        }
    }
}

void RBF_Energy::BuildMatrixH()
{   
    double re_time = 0;
    auto t3 = Clock::now();
    BuildSurfaceTermMat();
    auto t4 = Clock::now();
    cout << "BuildSurfaceTermMat time: " << (re_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;

    H_ = F_s_.t() * F_s_;
    auto t5 = Clock::now();
    cout << "Build H surface time: " << (re_time = std::chrono::nanoseconds(t5 - t4).count() / 1e9) << endl;

    if(this->max_iter_num_ <= 1) 
    {
        cout << " clear memory F_s " << endl;
        F_s_.clear();
    }
    
    auto t6 = Clock::now();
    BuildGradientTermMat();
    auto t7 = Clock::now();
    cout << "BuildGradientTermMat time: " << (re_time = std::chrono::nanoseconds(t7 - t6).count() / 1e9) << endl;

    H_ += e_para_.e_beta * F_g_.t() * F_g_;
    auto t8 = Clock::now();
    cout << "Build H gradient time: " << (re_time = std::chrono::nanoseconds(t8 - t7).count() / 1e9) << endl;

    BuildMatrixB();
    auto t9 = Clock::now();
    cout << "BuildMatrixB time: " << (re_time = std::chrono::nanoseconds(t9 - t8).count() / 1e9) << endl;
    
    if(this->max_iter_num_ <= 1)
    {
        cout << " clear memory F_g_ " << endl;
        F_g_.clear();
    } 

    auto t10 = Clock::now();
    H_.submat(0, 0, arma::size(pt_n_*4, pt_n_*4)) += e_para_.e_lambda * rbf_core_.M;
    auto t11 = Clock::now();
    cout << "Build H_ with duchon time: " << (re_time = std::chrono::nanoseconds(t11 - t10).count() / 1e9) << endl;

    // arma::mat fg_t =  F_g_.t();
    // fg_t = fg_t.each_row() % alpha_g_;
    
    // H_ = fs_t * F_s_ + e_para_.e_lambda * D_M_ + beta * fg_t * F_g_;
    // if(this->max_iter_num_ <= 1)
    // {
    //     D_M_.clear();
    //     F_g_.clear();
    //     F_s_.clear();
    // }
    
}

void RBF_Energy::BuildMatrixHSparse()
{
    double re_time = 0;
    auto t3 = Clock::now();
    arma::sp_mat F_s_sp(pt_n_, pt_n_ * 4 + 4);
    //arma::sp_mat F_s_sp(pt_n_, pt_n_ * 4);
    F_s_sp(0, 0, arma::size(pt_n_, pt_n_)) = rbf_core_.M_s(0, 0, arma::size(pt_n_, pt_n_));
    F_s_sp(0, pt_n_, arma::size(pt_n_, 3 * pt_n_)) = rbf_core_.M_s(0, pt_n_, arma::size(pt_n_, 3 * pt_n_));
    F_s_sp(0, 4 * pt_n_, arma::size(pt_n_, 4)) = rbf_core_.N(0, 0, arma::size(pt_n_, 4));

    cout << "S mat size " << F_s_sp.n_nonzero << endl;

    auto t4 = Clock::now();
    cout << "Build sparse SurfaceTermMat time: " << (re_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;
    H_sp_ = F_s_sp.t() * F_s_sp;
    auto t5 = Clock::now();
    cout << "Build sparse H surface time: " << (re_time = std::chrono::nanoseconds(t5 - t4).count() / 1e9) << endl;

    if (this->max_iter_num_ <= 1)
    {
        cout << " clear memory F_s " << endl;
        F_s_sp.clear();
    }

    auto t6 = Clock::now();
    arma::sp_mat F_g_sp(3 * pt_n_, pt_n_ * 4 + 4);
    //arma::sp_mat F_g_sp(3 * pt_n_, pt_n_ * 4);
    F_g_sp(0, 0, arma::size(3 * pt_n_, pt_n_)) = rbf_core_.M_s(pt_n_, 0, arma::size(3 * pt_n_, pt_n_));
    F_g_sp(0, pt_n_, arma::size(3 * pt_n_, 3 * pt_n_)) = rbf_core_.M_s(pt_n_, pt_n_, arma::size(3 * pt_n_, 3 * pt_n_));
    F_g_sp(0, 4 * pt_n_, arma::size(3 * pt_n_, 4)) = rbf_core_.N(pt_n_, 0, arma::size(3 * pt_n_, 4));
    auto t7 = Clock::now();
    cout << "Build sparse GradientTermMat time: " << (re_time = std::chrono::nanoseconds(t7 - t6).count() / 1e9) << endl;

    H_sp_ += e_para_.e_beta * F_g_sp.t() * F_g_sp;
    auto t8 = Clock::now();
    cout << "Build sparse H gradient time: " << (re_time = std::chrono::nanoseconds(t8 - t7).count() / 1e9) << endl;

    G_mat_ = arma::zeros(1, pt_n_ * 3);
    for (size_t i = 0; i < pt_n_; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            G_mat_(0, j * pt_n_ + i) = gradients_[i * 3 + j];
        }
    }

    double beta = e_para_.e_beta;
    B_ = beta * (G_mat_ * F_g_sp);

    auto t9 = Clock::now();
    cout << "BuildMatrixB time: " << (re_time = std::chrono::nanoseconds(t9 - t8).count() / 1e9) << endl;

    if (this->max_iter_num_ <= 1)
    {
        cout << " clear memory F_g_ " << endl;
        F_g_sp.clear();
    }

    auto t10 = Clock::now();
    H_sp_.submat(0, 0, arma::size(pt_n_ * 4, pt_n_ * 4)) += e_para_.e_lambda * rbf_core_.M_s;
    auto t11 = Clock::now();
    cout << "Build H_ with duchon time: " << (re_time = std::chrono::nanoseconds(t11 - t10).count() / 1e9) << endl;

}


void RBF_Energy::BuildMatrixHEigenSparse()
{
    double re_time = 0;
    auto t3 = Clock::now();

   /* SpMat F_s_sp(pt_n_, pt_n_ * 4 + 4);
    F_s_sp.block(0, 0, pt_n_, pt_n_) = rbf_core_.M_es.block(0, 0, pt_n_, pt_n_);
    F_s_sp.block(0, pt_n_, pt_n_, 3 * pt_n_) = rbf_core_.M_es.block(0, pt_n_, pt_n_, 3 * pt_n_);
    F_s_sp.block(0, 4 * pt_n_, pt_n_, 4) = rbf_core_.N_es.block(0, 0, pt_n_, 4);
    */

    auto t4 = Clock::now();
    cout << "Build sparse SurfaceTermMat time: " << (re_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;
    H_esp_ = SpMat(rbf_core_.F_s_sp.transpose()) * rbf_core_.F_s_sp;
    auto t5 = Clock::now();
    cout << "Build sparse H surface time: " << (re_time = std::chrono::nanoseconds(t5 - t4).count() / 1e9) << endl;

    if (this->max_iter_num_ <= 1)
    {
        cout << " clear memory F_s " << endl;
        rbf_core_.F_s_sp.data().clear();
    }

    auto t6 = Clock::now();
    /*SpMat F_g_sp(3 * pt_n_, pt_n_ * 4 + 4);
    F_g_sp.block(0, 0, 3 * pt_n_, pt_n_) = rbf_core_.M_es.block(pt_n_, 0, 3 * pt_n_, pt_n_);
    F_g_sp.block(0, pt_n_, 3 * pt_n_, 3 * pt_n_) = rbf_core_.M_es.block(pt_n_, pt_n_, 3 * pt_n_, 3 * pt_n_);
    F_g_sp.block(0, 4 * pt_n_, 3 * pt_n_, 4) = rbf_core_.N_es.block(pt_n_, 0, 3 * pt_n_, 4);*/
    auto t7 = Clock::now();
    cout << "Build sparse GradientTermMat time: " << (re_time = std::chrono::nanoseconds(t7 - t6).count() / 1e9) << endl;

     
    H_esp_ += e_para_.e_beta * SpMat(rbf_core_.F_g_sp.transpose()) * rbf_core_.F_g_sp;
    auto t8 = Clock::now();
    cout << "Build sparse H gradient time: " << (re_time = std::chrono::nanoseconds(t8 - t7).count() / 1e9) << endl;

    Eigen::VectorXd G_mat(pt_n_ * 3);
    
    for (size_t i = 0; i < pt_n_; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            G_mat(j * pt_n_ + i) = gradients_[i * 3 + j];
        }
    }

    double beta = e_para_.e_beta;
    B_e_ = beta * (G_mat * rbf_core_.F_g_sp);

    auto t9 = Clock::now();
    cout << "BuildMatrixB time: " << (re_time = std::chrono::nanoseconds(t9 - t8).count() / 1e9) << endl;

    if (this->max_iter_num_ <= 1)
    {
        cout << " clear memory F_g_ " << endl;
        rbf_core_.F_g_sp.data().clear();
    }

    auto t10 = Clock::now();
    if (e_para_.e_lambda > 1e-8)
    {
        H_esp_ += e_para_.e_lambda * rbf_core_.M_es;
        rbf_core_.M_es.data().clear();
    }
    auto t11 = Clock::now();
    cout << "Build H_ with duchon time: " << (re_time = std::chrono::nanoseconds(t11 - t10).count() / 1e9) << endl;

}


void RBF_Energy::ReduceBAndHMatWithSVD()
{
    if (e_para_.enable_constriants)
    {
        if (e_para_.use_sparse)
        {
            B_reduced_ = B_ * SVD_V_;
            H_sp_ = SVD_V_.t() * H_sp_;
            H_sp_ = H_sp_ * SVD_V_;
            return;
        }

        B_reduced_ = B_ * SVD_V_;
        B_.clear();
        // H_ = SVD_V_.t() * H_ * SVD_V_;
        H_ = SVD_V_.t() * H_; 
        H_ =  H_ * SVD_V_;    
    }
}

void RBF_Energy::BuildEnergyMatrix()
{
    if(pts_.empty() || gradients_.empty())
    {
        std::cout <<"ConstructEnergyMatrix failed! Empty points data or normal data! " << std::endl;
        return;
    }
    double re_time;
    //auto t1 = Clock::now();
    // ProcessGradientsAndConfidenceMat();
    //auto t2 = Clock::now();
    

    //cout << "ProcessGradients time: " <<  (re_time = std::chrono::nanoseconds(t2 - t1).count()/1e9) <<endl;
    // BuildGradientTermMat();
    //auto t3 = Clock::now();
    //cout << "BuildGradientTermMat time: " <<  (re_time = std::chrono::nanoseconds(t3 - t2).count()/1e9) <<endl;
    // BuildSurfaceTermMat();
    auto t4 = Clock::now();
    //cout << "BuildSurfaceTermMat time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
    if(e_para_.enable_constriants)
    {
         BuildConditionMat();
    }
   
    auto t5 = Clock::now();
    cout << "BuildConditionMat time: " <<  (re_time = std::chrono::nanoseconds(t5 - t4).count()/1e9) <<endl;
    
    //auto t7 = Clock::now();
    //cout << "SolveConditionMatSVD time: " <<  (re_time = std::chrono::nanoseconds(t7 - t5).count()/1e9) <<endl;
    // BuildConfidenceMat();
    //auto t8 = Clock::now();
    // cout << "BuildConfidenceMat time: " <<  (re_time = std::chrono::nanoseconds(t8 - t7).count()/1e9) <<endl;
    auto t6 = Clock::now();
    // cout << "BuildMatrixB time: " <<  (re_time = std::chrono::nanoseconds(t6 - t8).count()/1e9) <<endl;
    if (e_para_.use_sparse)
    {
        if (e_para_.use_eigen_sparse)
        {
            BuildMatrixHEigenSparse();
        }
        else {
            BuildMatrixHSparse();
        }
        
    }
    else {
        BuildMatrixH();
    }

    auto t9 = Clock::now();
    cout << "BuildMatrixH time: " <<  (re_time = std::chrono::nanoseconds(t9 - t6).count()/1e9) <<endl;
    if(max_iter_num_ <= 1)
    {
        rbf_core_.clearMemory();
    }
    
    if(e_para_.enable_constriants) SolveConditionMatSVD();
    auto t10 = Clock::now();
    cout << "ConditionMat SVD time: " << (re_time = std::chrono::nanoseconds(t10 - t9).count() / 1e9) << endl;
    /* ReduceBAndHMatWithSVD();
    auto t10 = Clock::now();
    cout << "ReduceBAndHMatWithSVD time: " <<  (re_time = std::chrono::nanoseconds(t10 - t9).count()/1e9) <<endl;
    SolveReducedLinearSystem();
    auto t11 = Clock::now();
    cout << "SolveReducedLinearSystem time: " <<  (re_time = std::chrono::nanoseconds(t11 - t10).count()/1e9) <<endl;*/
    // SaveMatrix();
}

void RBF_Energy::SolveEnergyMatrix()
{
    double re_time;
    auto t9 = Clock::now();
    ReduceBAndHMatWithSVD();
    auto t10 = Clock::now();
    cout << "--- ReduceBAndHMatWithSVD time: " << (re_time = std::chrono::nanoseconds(t10 - t9).count() / 1e9) << endl;
    
    SolveReducedLinearSystem();
    auto t11 = Clock::now();
    cout << "SolveReducedLinearSystem time: " << (re_time = std::chrono::nanoseconds(t11 - t10).count() / 1e9) << endl;
}


void RBF_Energy::SolveReducedLinearSystem()
{
    double re_time;
    std::cout << " start to solve arma linear system ....... " << std::endl;
    
    if (e_para_.enable_constriants)
    {
        if (e_para_.use_sparse)
        {
            if (e_para_.use_eigen_sparse)
            {

            }
            else {
                arma::superlu_opts opts;
                opts.allow_ugly = true;
                opts.equilibrate = true;
                arma::spsolve(X_reduced_, H_sp_, B_reduced_.t(), "lapack", opts);
                H_sp_.clear();
            }
            
            X_ = SVD_V_ * X_reduced_;
        }
        else {
            X_reduced_ = arma::solve(H_, B_reduced_.t(), arma::solve_opts::likely_sympd);
            //X_reduced_ = arma::solve(H_, B_reduced_.t(), arma::solve_opts::fast);
            H_.clear();
            X_ = SVD_V_ * X_reduced_;
        }
        
        if(max_iter_num_<=1) SVD_V_.clear();
    } else {
        if (e_para_.use_sparse)
        {
            if (e_para_.use_eigen_sparse)
            {
                //Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
               /* Eigen::SPQR<SpMat> solver;
                auto x = solver.compute(H_esp_).solve(B_e_);
                X_.set_size(pt_n_ * 4 + 4, 1);
                for (size_t i = 0; i < pt_n_ * 4 + 4; ++i)
                {
                    X_(i, 0) = x(i);
                }*/
            }
            else {
                arma::superlu_opts opts;
                opts.allow_ugly = true;
                opts.equilibrate = true;
                arma::spsolve(X_, H_sp_, B_.t(), "lapack", opts);
            }
            
        }
        else {
            X_ = arma::solve(H_, B_.t(), arma::solve_opts::fast);
        }
        
    }
     

    //SaveMatrix(X_, "X_.txt");
    rbf_core_.a = X_(0, 0, arma::size(pt_n_ * 4, 1));
    double sum_a = arma::accu(rbf_core_.a);
    // std::cout << " ----rbf sum : " << sum_a << std::endl;
    rbf_core_.b = X_(pt_n_ * 4, 0, arma::size(4, 1));
   
    //rbf_core_.b.zeros(4,1);

}

void RBF_Energy::SetOutdir(const std::string& dir)
{
    e_para_.out_dir = dir;
}

void RBF_Energy::RunTest(const std::string& ply_path, const std::string gradient_path, bool is_gradient)
{
    if(is_gradient) e_para_.use_gradient = true;
    e_para_.mesh_points_path = ply_path;
    e_para_.gradients_path = gradient_path;
    RunTest();
}

void RBF_Energy::RunTest()
{
    double re_time;
    cout<<"start solve rbf matrix "<<endl;
    // std::cout << "e_para_.use_gradient " << e_para_.use_gradient << std::endl;
    auto t1 = Clock::now();
    auto t2 = Clock::now();

    SetPts(e_para_.mesh_points_path);
    // std::cout << "e_para_.use_gradient " << e_para_.use_gradient << std::endl;
    
    if(e_para_.use_gradient)
    {
        LoadPtsGradients(e_para_.gradients_path);
        std::cout <<"Load gradients succeed !" << std::endl;
        ProcessGradientsAndConfidenceMat();
    }
    
    NormalizePts();
     
    SaveFuncInputs();
    auto t3 = Clock::now();
    cout << "set pts time: " <<  (re_time = std::chrono::nanoseconds(t3 - t2).count()/1e9) <<endl;
    
    only_build_M_ = true;
    InitRBFCore();
    auto t4 = Clock::now();
    cout << "InitRBFCore time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
    if (e_para_.max_solve_iterate_num > 1)
    {
        max_iter_num_ = e_para_.max_solve_iterate_num;
        SolveRBFIterate();
    }
    else {
        SolveRBF();
    }
}


void RBF_Energy::RunTestWithOptNormal()
{
    double re_time;
    cout<<"start solve rbf matrix "<<endl;
    // std::cout << "e_para_.use_gradient " << e_para_.use_gradient << std::endl;
    auto t1 = Clock::now();
    auto t2 = Clock::now();

    NormalizePts();
    SaveFuncInputs();
    auto t3 = Clock::now();
    cout << "set pts time: " <<  (re_time = std::chrono::nanoseconds(t3 - t2).count()/1e9) <<endl;
    
    only_build_M_ = true;
    InitRBFCore();
    
    auto t4 = Clock::now();
    cout << "InitRBFCore time: " <<  (re_time = std::chrono::nanoseconds(t4 - t3).count()/1e9) <<endl;
    if (e_para_.max_solve_iterate_num > 1)
    {
        max_iter_num_ = e_para_.max_solve_iterate_num;
        SolveRBFIterate();
    }
    else {
        SolveRBF();
    }

    auto t5 = Clock::now();
    cout << "all least solve time: " <<  (re_time = std::chrono::nanoseconds(t5 - t2).count()/1e9) <<endl;
}



void RBF_Energy::SolveRBF()
{
    double re_time;
    auto t3 = Clock::now();
    //cout << "SetPts time: " << (re_time = std::chrono::nanoseconds(t3 - t2).count() / 1e9) << endl;

    /*SetRBFPara();
    InitRBFCore();
    auto t4 = Clock::now();
    cout << "InitRBFCore time: " << (re_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;*/

    BuildEnergyMatrix();

    
    //cout << " ~~~~~~~~~~~~~~~ finish build energy matrix" << endl;
    SolveEnergyMatrix();
    auto t5 = Clock::now();
    cout << "BuildEnergyMatrixAndSolve time: " << (re_time = std::chrono::nanoseconds(t5 - t3).count() / 1e9) << endl;
    auto t2 = Clock::now();
    cout << "Total matrix solve time: " << (re_time = std::chrono::nanoseconds(t2 - t3).count() / 1e9) << endl;

    // double duchon_en = CalculateDuchonEnergy();
    // cout << "DuchonEnergy : " << duchon_en << endl;

    // double gradient_en = CalculateGradientEnergy();
    // cout << "GradientEnergy : " << gradient_en << endl;

    // double surface_en = CalculateSurfaceEnergy();
    // cout << "SurfaceEnergy : " << surface_en << endl;

  /*  double all_en = CalculateAllEnergy();
    cout << "all Energy : " << all_en << endl;*/
    VisualFuncValues();
    if (e_para_.is_surfacing) {
        // rbf_core_.InitNormal();
        // rbf_core_.OptNormal(0);
        rbf_core_.Surfacing(0, e_para_.volumn_dim);
        // std::string out_path = "../out_mesh";
        rbf_core_.Write_Surface(e_para_.out_dir);
        // rbf_core_.Update_Newnormals();
        // rbf_core_.Write_Hermite_NormalPrediction(e_para_.out_dir + "_normal", 1);
    }
    if (e_para_.save_estimate_normal)
    {
        EstimateRBFNormals();
    }
}

void RBF_Energy::SolveRBFIterate()
{
    BuildEnergyMatrix();
    // max_iter_num_ = (size_t)e_para_.max_solve_iterate_num;
    iter_threshold_ = e_para_.normal_iter_threshold;

    for (size_t i = 0; i < max_iter_num_; ++i)
    {
        double re_time;
        auto t1 = Clock::now(); 
        SolveEnergyMatrix();
        auto t2 = Clock::now();
        cout << "matrix solve time: " << (re_time = std::chrono::nanoseconds(t2 - t1).count() / 1e9) << endl;
       /* ReduceBAndHMatWithSVD();
        SolveReducedLinearSystem();*/
        //if (beta_weights_ * e_para_.e_beta > pt_n_ * 2) beta_weights_ = pt_n_ * 2;
        beta_weights_ = beta_weights_ * e_para_.normal_weight_incre;
        UpdateGradient();
        auto t3 = Clock::now();
        cout << "UpdateGradient time: " << (re_time = std::chrono::nanoseconds(t3 - t2).count() / 1e9) << endl;
        
        // avg_squred_distance = 
        cout << "aveg squared normal distance " <<normal_delt_avg_ << endl;
       
        if (normal_delt_avg_ < iter_threshold_)
        {
            cout << "aveg squared normal distance is smaller than the shreshold " << iter_threshold_ << endl;
            break;
        }
        cout << " start to update H and B matrix !" << endl;
        UpdateHAndBMat();
        auto t4 = Clock::now();
        cout << "UpdateHAndBMat time: " << (re_time = std::chrono::nanoseconds(t4 - t3).count() / 1e9) << endl;
        cout << std::to_string(i) << " iter Total matrix solve time: " << (re_time = std::chrono::nanoseconds(t4 - t1).count() / 1e9) << endl;
        if (e_para_.save_iter > 0)
        {
            if ((i + 1) % e_para_.save_iter == 0)
            {
                std::vector<double> updated_normals;
                for (size_t p_id = 0; p_id < pt_n_; ++p_id)
                {
                    updated_normals.push_back(G_mat_[p_id]);
                    updated_normals.push_back(G_mat_[pt_n_ + p_id]);
                    updated_normals.push_back(G_mat_[pt_n_ * 2 + p_id]);
                }
                std::string iter_save_dir = e_para_.out_dir + "_iter_normal_" + std::to_string(i);
                writePLYFile_VN(iter_save_dir, pts_, updated_normals);
                iter_save_dir = e_para_.out_dir + "_iter_normal_line_" + std::to_string(i);
                SavePointsAndNormals(pts_, updated_normals, iter_save_dir, 0.2);
                rbf_core_.Surfacing(0, e_para_.volumn_dim);
                iter_save_dir = e_para_.out_dir + "_iter_mesh_" + std::to_string(i);
                rbf_core_.Write_Surface(iter_save_dir);
            }
        }
    }
    if (e_para_.is_surfacing) {
        rbf_core_.Surfacing(0, e_para_.volumn_dim);
        rbf_core_.Write_Surface(e_para_.out_dir);
    }
    if (e_para_.save_estimate_normal)
    {
        EstimateRBFNormals();
    }
}

void RBF_Energy::EstimateRBFNormals()
{
    // std::vector<arma::vec3> estimated_normals;
    std::vector<double> estimated_normals;
    std::vector<double> estimated_normals_normalized;
    double delta = 0.000001;
    pt_n_ = pts_.size() / 3;
    cout << "  save pt normals num: " << pt_n_<< endl;
    for(size_t i = 0; i < pt_n_; ++i)
    {
        const R3Pt pt(pts_[i*3], pts_[i*3 + 1], pts_[i*3 + 2]);
        double dist = rbf_core_.Dist_Function(pt);
        const R3Pt pt_x(pts_[i*3] + delta, pts_[i*3 + 1], pts_[i*3 + 2]);
        double dist_x = rbf_core_.Dist_Function(pt_x);
        double dx = -1.0 * (dist_x - dist) / delta;

        const R3Pt pt_y(pts_[i*3], pts_[i*3 + 1] + delta, pts_[i*3 + 2]);
        double dist_y = rbf_core_.Dist_Function(pt_y);
        double dy = -1.0 * (dist_y - dist) / delta;

        const R3Pt pt_z(pts_[i*3], pts_[i*3 + 1], pts_[i*3 + 2] + delta);
        double dist_z = rbf_core_.Dist_Function(pt_z);
        double dz = -1.0 * (dist_z - dist) / delta;

        double n_len = sqrt(dx * dx + dy * dy + dz * dz);
        // n_len = 1.0;
        estimated_normals.push_back(dx);
        estimated_normals.push_back(dy);
        estimated_normals.push_back(dz);

        estimated_normals_normalized.push_back(dx/n_len );
        estimated_normals_normalized.push_back(dy/n_len );
        estimated_normals_normalized.push_back(dz/n_len);
        
    }
    if( max_iter_num_ <= 1)
    {
        this->gradients_ = estimated_normals_normalized;
    }
    
    // rbf_core_.newnormals = estimated_normals;
    // cout << "  save pt normals " << endl;
    if(e_para_.save_estimate_normal)
    {
        if(e_para_.save_visualization)
        {
            std::string out_path = e_para_.out_dir + "_rbf_normal_line";
            SavePointsAndNormals(pts_, estimated_normals, out_path);
        }
        
        std::string out_path = e_para_.out_dir + "_out_ptn";
        writePLYFile_VN(out_path, pts_, estimated_normals);
        std::string out_path2 = e_para_.out_dir + "_out_ptn_normalized";
        cout << "  save pt normals : " << out_path2  << endl;
        writePLYFile_VN(out_path2, pts_, estimated_normals_normalized);
    }
}

void RBF_Energy::VisualFuncValues()
{
    double step = 0.01;
    std::vector<double> pts;
    std::vector<uint8_t> pts_co;
    //rbf_core_.isHermite = true;
    //cout << "is hermite " << rbf_core_.isHermite << endl;
    for (size_t i = 0; i < 201; ++i)
    {
        double x = -1.0 + step * i;
        for (size_t j = 0; j < 201; ++j)
        {
            double y = -1.0 + step * j;
            pts.push_back(x);
            pts.push_back(y);
            pts.push_back(0);
            const R3Pt pt(x, y, 0);

            double dist = rbf_core_.Dist_Function(pt);
            double scale = 0.03;
            
            dist = dist > -scale ? dist : -scale;
            dist = dist < scale ? dist : scale;
            dist = dist / scale;
            int co_val = abs(dist) * 255;
            uint8_t c_val = uint8_t(co_val);
            if (dist >= 0)
            {
                pts_co.push_back(c_val);
                pts_co.push_back(0);
                pts_co.push_back(0);
            }
            else {
                pts_co.push_back(0);
                pts_co.push_back(c_val);
                pts_co.push_back(0);
            }
        }
    }
    std::string out_path = e_para_.out_dir + "_dist_val_co";
    writePLYFile_CO(out_path, pts, pts_co);
}



void RBF_Energy::VisualSamplePtsFuncValues()
{
    double step = 0.01;
    std::vector<double> pts;
    std::vector<uint8_t> pts_co;
    //rbf_core_.isHermite = true;
    //cout << "is hermite " << rbf_core_.isHermite << endl;
    pt_n_ = pts_.size() / 3;
    cout << "VisualSamplePtsFuncValues " << pt_n_ << endl;
    cout << "VisualSamplePtsFuncValues " << normals_.size() << endl;
    for (size_t i = 0; i < pt_n_; ++i)
    {
        double x = pts_[3 * i];
        double y = pts_[3 * i + 1];
        double z = pts_[3 * i + 2];
        const R3Pt pt(x, y, z);

        double dx = normals_[3 * i];
        double dy = normals_[3 * i + 1];
        double dz = normals_[3 * i + 2];

        double step = 0.02;
        cout << " new dist list ------------------------ " << endl;
        cout << " normal: " << dx << " " << dy << " " << dz << endl;
        for (size_t id = 0; id < 10; ++id)
        {
            double scale_v =  id * step;
            double x1 = scale_v * dx + x;
            double y1 = scale_v * dy + y;
            double z1 = scale_v * dz + z;
            const R3Pt pt(x1, y1, z1);
            pts.push_back(x1);
            pts.push_back(y1);
            pts.push_back(z1);
            double dist = rbf_core_.Dist_Function(pt);
            cout << " dist " << dist << endl;
            double scale = 0.03;
            dist = dist > -scale ? dist : -scale;
            dist = dist < scale ? dist : scale;
            dist = dist / scale;
            int co_val = abs(dist) * 255;
            uint8_t c_val = uint8_t(co_val);
            if (dist >= 0)
            {
                pts_co.push_back(c_val);
                pts_co.push_back(0);
                pts_co.push_back(0);
            }
            else {
                pts_co.push_back(0);
                pts_co.push_back(c_val);
                pts_co.push_back(0);
            }
        }
    }
    std::string out_path = e_para_.out_dir + "_sampe_pts_dist_color";
    writePLYFile_CO(out_path, pts, pts_co);
}




void RBF_Energy::CalNewNormalMat()
{
    cout << "start to get normal_mat_ " << endl;
    normal_mat_ = F_g_ * X_;
    cout << "finish get normal_mat_ " << endl;
    for (size_t i = 0; i < pt_n_; ++i)
    {
        double dx = normal_mat_[i];
        double dy = normal_mat_[pt_n_ + i];
        double dz = normal_mat_[2 * pt_n_ + i];
        double len = sqrt(dx * dx + dy * dy + dz * dz);
        normal_mat_[i] = normal_mat_[i] / len;
        normal_mat_[pt_n_ + i] = normal_mat_[pt_n_ + i] / len;
        normal_mat_[2 * pt_n_ + i] = normal_mat_[2 * pt_n_ + i] / len;
    }
}

void RBF_Energy::UpdateGradient()
{
    CalNewNormalMat();
    //cout << "finish update get gm value " << endl;
    arma::mat delt_mat = normal_mat_.t() - G_mat_;
    normal_delt_avg_ = arma::accu(delt_mat % delt_mat) / double(pt_n_);
    G_mat_ = normal_mat_.t();
}

void RBF_Energy::UpdateHAndBMat()
{
    // double beta = e_para_.e_beta * beta_weights_;
    // arma::mat g_mat_alpha = G_mat_.each_row() % alpha_g_;
    // B_ = beta * (g_mat_alpha * F_g_);
    // arma::mat fg_t =  F_g_.t();
    // fg_t = fg_t.each_row() % alpha_g_;
    // arma::mat fs_t =  F_s_.t();
    // H_ = fs_t * F_s_ + e_para_.e_lambda * D_M_ + beta * fg_t * F_g_;
    double beta = e_para_.e_beta * beta_weights_;
    B_ = beta * (G_mat_ * F_g_);
    cout << " finish update B mat "<< endl;
    H_ = F_s_.t() * F_s_;
    cout << " finish update Fs mat "<< endl;
    H_.submat(0, 0, arma::size(pt_n_*4, pt_n_*4)) += e_para_.e_lambda * rbf_core_.M;
    cout << " finish update dm mat "<< endl;
    H_ += beta * F_g_.t() * F_g_;
    cout << " finish update Fg mat "<< endl;
}

double RBF_Energy::CalculateDuchonEnergy() const
{
    // arma::mat energy = X_.t() * D_M_ * X_;
    //std::cout << " DuchonEnergy size " << energy.size() << endl;
    // return energy[0];
    return 0;
}

double RBF_Energy::CalculateGradientEnergy() const
{   
    arma::mat g_m = F_g_ * X_;
    SaveMatrix(g_m, "g_m.txt");
    //SaveMatrix(F_g_, "F_g.txt");
    SaveMatrix(G_mat_, "G_mat_.txt");
    arma::mat gm_delt = g_m.t() - G_mat_;
    SaveMatrix(gm_delt, "gm_delt.txt");
    arma::mat gm_sum = gm_delt % gm_delt;
    SaveMatrix(gm_sum, "gm_sum.txt");
    gm_sum = gm_sum % alpha_g_;
    double g_energy = arma::accu(gm_sum);
    //SaveMatrix(g_energy, "g_energy.txt");
    return g_energy;

    //arma::mat G_mat_apla = G_mat_ % alpha_g_;
    //arma::mat g_m_t = g_m.t() % alpha_g_;
    //std::cout << " g_m size " << g_m.n_rows <<" " <<g_m.n_cols << endl;
    //std::cout << " G_mat_ size " << G_mat_.n_rows << " " << G_mat_.n_cols << endl;
    //arma::mat energy0 = g_m_t * g_m - 2 * G_mat_apla * g_m + G_mat_apla * G_mat_.t();
    // cout << "gradient Energy before rbf gradient normalized: " << energy0[0] << endl;

    // for (size_t i = 0; i < pt_n_; ++i)
    // {
    //     double dx = g_m[i];
    //     double dy = g_m[ pt_n_ + i];
    //     double dz = g_m[2 * pt_n_ + i];
    //     double len = sqrt(dx * dx + dy * dy + dz * dz);
    //     g_m[i] = g_m[i] / len;
    //     g_m[pt_n_ + i] = g_m[pt_n_ + i] / len;
    //     g_m[2 * pt_n_ + i] = g_m[2 * pt_n_ + i] / len;
    // }

   /* SaveMatrix(g_m, "g_m.txt");
    SaveMatrix(F_g_, "F_g.txt");
    SaveMatrix(G_mat_, "G_mat_.txt");*/
    // arma::mat energy = g_m.t() * g_m - 2 * G_mat_ * g_m + G_mat_ * G_mat_.t();
    //std::cout << " GradientEnergy size " << energy.size() << endl;
    //return energy0[0];
}

double RBF_Energy::CalculateSurfaceEnergy() const
{
    arma::mat s_m = F_s_ * X_;
    arma::mat s_m_t = s_m.t();
    s_m_t = s_m_t.each_row() % alpha_s_;
    arma::mat energy = s_m.t() * s_m;
    //std::cout << " SurfaceEnergy size " << energy.size() << endl;
    return energy[0];
}

double RBF_Energy::CalculateAllEnergy() const
{
    /*std::cout << " H_ size " << H_.n_rows << " " << H_.n_cols << endl;
    std::cout << " X_ size " << X_.n_rows << " " << X_.n_cols << endl;*/
    //arma::mat e_all = X_.t() * H_ * X_;
    //arma::mat e_all = X_.t() * H_ * X_ - 2 * B_ * X_ + G_mat_ % alpha_g_ * G_mat_.t();
    //return e_all[0];
    return 0;
}

void RBF_Energy::SolveIncreVipss()
{
    /*RBF_Core rbf_core;*/
    this->SetRBFPara();
    vector<double>Vs;
    std::vector<double> vn;
    readPLYFile(e_para_.mesh_points_path, Vs, vn);
    normals_ = vn;
    pts_ = Vs;
    NormalizePts();
    Vs = pts_;
    std::string out_mesh_path = "normalized_pt.ply"; 
    writePLYFile_VN(out_mesh_path, Vs, vn);

    vector<double> key_vs;
    vector<double> auxi_vs;

    
    size_t sample_num = 50;


    if(e_para_.vipss_incre_init_pt_num < 1.0)
    {
        sample_num = size_t(Vs.size() / 3 * e_para_.vipss_incre_init_pt_num);
    } else {
        sample_num = size_t(e_para_.vipss_incre_init_pt_num); 
    }
    
    PTSample::FurthestSamplePointCloud(Vs, sample_num, key_vs, auxi_vs);

    rbf_core_.apply_sample = e_para_.vipss_apply_sample;
    rbf_core_.user_beta = e_para_.vipss_beta;
    
    rbf_para_.user_lamnbda = e_para_.v_lambda;
    rbf_core_.sample_iter = true;
    rbf_core_.sample_threshold = e_para_.vipss_incre_shreshold;
    // rbf_core.incre_num = size_t(auxi_vs.size() / 3  * 0.02);

    rbf_core_.incre_num = e_para_.vipss_incre_pt_num;
    rbf_core_.max_sample_iter = e_para_.vipss_incre_max_iter;
    // size_t i =0;
    bool enable_debug = e_para_.vipss_incre_debug;
    double re_time = 0;

    auto t_begin = Clock::now();
    for(size_t i =0; i < rbf_core_.max_sample_iter; ++i)
    {
        auto t0 = Clock::now();
        std::cout << " current lopp id " << i << endl;
        rbf_core_.pts = key_vs;
        rbf_core_.auxi_npt = auxi_vs.size() / 3;
        rbf_core_.auxi_pts = auxi_vs;
        rbf_core_.InjectData(rbf_para_);

        rbf_core_.BuildK(rbf_para_);
        rbf_core_.InitNormal(rbf_para_);
        rbf_core_.opt_incre = true;
        // cout << "optimize normal ...... " << std::endl;
        rbf_core_.OptNormal(0);
        rbf_core_.opt_incre = false;

        std::string out_normal_dir = e_para_.out_dir + std::to_string(i) + "_normal";
        // rbf_core.Write_Hermite_NormalPrediction(out_normal_dir, 1);

        size_t n_voxel_line = e_para_.volumn_dim;
        if(e_para_.is_surfacing && enable_debug){
            rbf_core_.Surfacing(0, n_voxel_line);
            std::string out_surface_dir = e_para_.out_dir + std::to_string(i) +"_surface";
            rbf_core_.Write_Surface(out_surface_dir);
        }
        bool is_outputtime = true;
        if(is_outputtime && enable_debug){
            std::string out_time_dir = e_para_.out_dir + std::to_string(i) +"_time.txt";
            rbf_core_.Print_TimerRecord_Single(out_time_dir);
        }
        if(!rbf_core_.apply_sample) return;
        std::string out_pt_color_dir = e_para_.out_dir + std::to_string(i) + "_auxi_color";
        rbf_core_.CalculateAuxiDistanceVal(out_pt_color_dir, enable_debug);

        if(rbf_core_.auxi_dist_mat.n_cols == 0)
        {
            continue;
        }
        // cout << "  rbf_core.auxi_dist_mat  " <<  rbf_core.auxi_dist_mat.n_cols << endl;
        arma::vec dist_vec = rbf_core_.auxi_dist_mat.col(0);
        // cout << " dist_vec size "<< dist_vec.size() << endl;
        arma::uvec indices = arma::sort_index(dist_vec, "descend");
        // cout << " indices size "<< indices.size() << endl;

        std::vector<double> new_pts;
        std::vector<double> left_auxi_pts;
        for(size_t i = 0; i < indices.size(); ++i)
        {
            size_t id = indices(i);
            if( i < indices.size() && i < rbf_core_.incre_num * 4)
            {
                new_pts.push_back(auxi_vs[id*3]);
                new_pts.push_back(auxi_vs[id*3 + 1]);
                new_pts.push_back(auxi_vs[id*3 + 2]);
            } else {
                left_auxi_pts.push_back(auxi_vs[id*3]);
                left_auxi_pts.push_back(auxi_vs[id*3 + 1]);
                left_auxi_pts.push_back(auxi_vs[id*3 + 2]);
            }            
        }
        std::vector<double> incre_key_pts;
        std::vector<double> new_auxi_pts;
        // cout << " start to sample large error points " << endl;
        sample_num = rbf_core_.incre_num;
        PTSample::FurthestSamplePointCloud(new_pts, sample_num, incre_key_pts, new_auxi_pts);
        // cout << " add new points num " << incre_key_pts.size()<< endl;
        for(auto val : incre_key_pts)
        {
            key_vs.push_back(val);
        }
        for(auto val : new_auxi_pts)
        {
            left_auxi_pts.push_back(val);
        }
        auxi_vs = left_auxi_pts;

        if(!rbf_core_.sample_iter)
        {
            break;
        }
        auto t1 = Clock::now();
        cout << "vipss incre iter " << i << " time: " << (re_time = std::chrono::nanoseconds(t1 - t0).count() / 1e9) << endl;
    }
    rbf_core_.As_.clear();
    
    auto t_end = Clock::now();
    cout << "--------------vipss incre iter all time: " << (re_time = std::chrono::nanoseconds(t_end - t_begin).count() / 1e9) << endl;
    /*rbf_core_ = std::make_shared<RBF_Core>(); */
   
    
    pts_ = Vs;
    pt_n_ = pts_.size()/3;
    std::cout << " EstimateRBFNormals " << std::endl;

    auto t_p0 = Clock::now();
    EstimateRBFNormals();
    auto t_p1 = Clock::now();
    cout << " --------------  normal estimation time: " << (re_time = std::chrono::nanoseconds(t_p1 - t_p0).count() / 1e9) << endl;

    rbf_core_.clearMemory();
    
}

void RBF_Energy::SolveVipss()
{
    /*RBF_Core rbf_core;*/
    this->SetRBFPara();
    vector<double>Vs;
    std::vector<double> vn;
    readPLYFile(e_para_.mesh_points_path, Vs, vn);
    pts_ = Vs;
    normals_ = vn;
    //NormalizePts();
  
    std::string out_mesh_path = e_para_.out_dir + "normalized_pt.ply";
    writePLYFile_VN(out_mesh_path, Vs, vn);
    rbf_core_.apply_sample = e_para_.vipss_apply_sample;
    rbf_core_.user_beta = e_para_.vipss_beta;

    rbf_para_.user_lamnbda = e_para_.v_lambda;
    rbf_core_.sample_iter = true;

    // rbf_core.incre_num = size_t(auxi_vs.size() / 3  * 0.02);

    rbf_core_.incre_num = e_para_.vipss_incre_pt_num;
    rbf_core_.max_sample_iter = e_para_.vipss_incre_max_iter;
    // size_t i =0;
    bool enable_debug = e_para_.vipss_incre_debug;
    double re_time = 0;

    auto t_begin = Clock::now();
    auto t0 = Clock::now();

    rbf_core_.pts = pts_;
    rbf_core_.InjectData(rbf_para_);
    rbf_core_.BuildK(rbf_para_);
    
    // cout << "optimize normal ...... " << std::endl;
    if (e_para_.use_input_normal)
    {
        rbf_core_.AssignInitNormals(vn);
    }
    else {
        rbf_core_.InitNormal(rbf_para_);
    }
    rbf_core_.opt_incre = false;

    if (e_para_.only_vipss_hrbf)
    {
        rbf_core_.Opt_Hermite_With_InputNormal();
    }
    else {
        rbf_core_.OptNormal(0);
    }
    
    std::string out_normal_dir = e_para_.out_dir +  "_normal";
    rbf_core_.Write_Hermite_NormalPrediction(out_normal_dir, 1);
    /* writePLYFile_VN(out_normal_dir, rbf_core.pts, rbf_core.init);*/

    VisualFuncValues();
    //VisualSamplePtsFuncValues();

    EstimateRBFNormals();

    size_t n_voxel_line = e_para_.volumn_dim;
    if (e_para_.is_surfacing ) {
        rbf_core_.Surfacing(0, n_voxel_line);
        std::string out_surface_dir = e_para_.out_dir +  "_surface";
        rbf_core_.Write_Surface(out_surface_dir);
    }
    bool is_outputtime = true;
    if (is_outputtime ) {
        std::string out_time_dir = e_para_.out_dir + "_time.txt";
        rbf_core_.Print_TimerRecord_Single(out_time_dir);
    }
    auto t1 = Clock::now();
    cout << "vipss time: " << (re_time = std::chrono::nanoseconds(t1 - t0).count() / 1e9) << endl;
    
}

// apply incremental vipss to optimize pts normal

void RBF_Energy::SolveWithVipssOptNormal()
{        
    SolveIncreVipss();
    RunTestWithOptNormal();

    if(e_para_.save_estimate_normal)
    {
        std::string out_normal_dir = e_para_.out_dir + "_opt_input_normal";
        writePLYFile_VN(out_normal_dir, pts_, gradients_);
    }
    
    // if(e_para_.is_surfacing){
    //     rbf_core.Surfacing(0,n_voxel_line);
    //     rbf_core.Write_Surface("_surface");
    // }

    // if(is_outputtime){
    //     rbf_core.Print_TimerRecord_Single(outpath+pcname+"_time.txt");
    // }
}