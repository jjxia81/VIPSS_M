#include "rbfcore.h"
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

struct RBF_Energy_PARA{
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
    int volumn_dim = 50;
    double e_lambda = 0.1;
    double e_beta = 0.1;
    std::string mesh_points_path;
    std::string gradients_path;
    std::string out_dir;
    void loadYamlFile(const std::string& yaml_path);
};

class RBF_Energy
{
    public:
        RBF_Energy();
        ~RBF_Energy();
        // void InitPara();
        void InitRBFCore();
        
        void SetPts(const std::string& ply_path);
        
        void SolveConditionMatSVD();
        // void ConstructEnergyMatrix();

        
        void LoadPtsAndNormals(const std::string& ply_path);
        void RunTest(const std::string& ply_path, const std::string gradient_path=" ", bool is_gradient=false);
        void RunTest();
        void LoadPtsGradients(const std::string& gradient_path);
        void ProcessGradientsAndConfidenceMat(); 
        void SetOutdir(const std::string& dir);
        void SetEnergyParameters(const RBF_Energy_PARA& rbf_e_para);
        void EstimateRBFNormals();

    private:
        void SetRBFPara();
        void DebugLog(const std::string& log_str);
        void BuildSurfaceTermMat();
        void BuildGradientTermMat();
        void BuildMatrixB();
        void BuildConditionMat();
        void BuildHessianMat();
        void BuildEnergyMatrixAndSolve();
        void BuildConfidenceMat();
        void ReduceBAndHMatWithSVD();
        void SolveReducedLinearSystem();
        void SaveFuncInputs();
        // void SaveMatrix();
        // void ConvertArmaMatToEigenMat(const arma::mat& in_mat,  );
    
        double CalculateDuchonEnergy() const;
        double CalculateGradientEnergy() const;
        double CalculateSurfaceEnergy() const;

    public: 
        
        arma::mat F_s_; // surface term matrix F_s
        arma::mat F_g_; // gradient term matrix F_g
        arma::mat C_a_; // condition constraint matrix A
        arma::mat B_;
        arma::mat B_reduced_;
        arma::mat H_;
        arma::mat D_M_; // Duchon energy mat
        arma::mat alpha_s_;
        arma::mat alpha_g_;
        arma::mat reduced_solution_;
        arma::mat original_solution_;
        size_t pt_n_; 
        arma::vec grad_lens_;
        arma::mat confidence_; 
        std::vector<double> tangents_;
        std::vector<double> gradients_;
        std::vector<double> normals_;
        std::vector<double> pts_;
        arma::mat SVD_V_;
        arma::mat X_;
        arma::mat X_reduced_;
        

    private:
        std::shared_ptr<RBF_Core> rbf_core_;
        RBF_Paras rbf_para_;
        RBF_Energy_PARA e_para_;
        // std::shared_ptr<RBF_Energy_PARA> para_ptr_;
        // bool enable_debug_;
        // bool is_surfacing_;
        // bool use_gradient_;
        // bool use_confidence_;
        // bool solve_with_Eigen_;
        // int n_voxel_line_ = 100;
        // double e_lambda_;
        // double e_beta_;
        // std::string out_dir_;
        // std::string pt_path_;
        // std::string gd_path_;

};