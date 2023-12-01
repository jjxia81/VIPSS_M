#include "rbfcore.h"
#include <memory>
#include <string>

class RBF_Energy
{
    public:
        RBF_Energy();
        ~RBF_Energy();
        void InitPara();
        void InitRBFCore();
        void SetRBFPARA();

        // arma::mat& getM() const;
        // arma::mat getM00() const;
        // arma::mat getM01() const;
        // arma::mat getM11() const;

        void SetPts(const std::string& ply_path);
        
        void SolveConditionMatSVD();
        void ConstructEnergyMatrix();

        void TextLog(const std::string& log_str);
        void LoadPtsAndGradients(const std::string& ply_path);
        void RunTest(const std::string& ply_path);

    private:
        void BuildSurfaceTermMat();
        void BuildGradientTermMat();
        void BuildMatrixB();
        void BuildConditionMat();
        void BuildHessianMat();
        void BuildEnergyMatrix();
        void BuildConfidenceMat();
        void ReduceBAndHMatWithSVD();
        void SolveReducedLinearSystem();


    public: 
        bool enable_debug_;
        double e_lambda_;
        double e_beta_;
        double* e_alpha;
        arma::mat F_s_; // surface term matrix F_s
        arma::mat F_g_; // gradient term matrix F_g
        arma::mat C_a_; // condition constraint matrix A
        arma::mat B_;
        arma::mat H_;
        arma::mat D_M_; // Duchon energy mat
        arma::mat confidence_mat_s_;
        arma::mat confidence_mat_g_;
        arma::mat reduced_solution_;
        arma::mat original_solution_;
        size_t pt_n_; 
        std::vector<double> gradients_;
        std::vector<double> pts_;
        arma::mat SVD_V_;




    private:
        std::shared_ptr<RBF_Core> rbf_core_;
        RBF_Paras rbf_para_;

};