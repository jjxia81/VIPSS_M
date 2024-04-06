#pragma once
#include "rbf_energy.h"
#include "rbf_energy_para.h"
#include <set>
#include <map>
// #include "gurobi/gurobi_c++.h"

class NormalOptimizer
{
public:
	
	NormalOptimizer(){};
	~NormalOptimizer() {};
	void Init();
	void SetRBFPara();
	void LoadAllContours(const std::string& contour_path);
	void CombineContoursAndSigns();
	void SetRBFEnergyInput();
	void OptimizeNormalSigns();
	// void NormalizePtCoords();
	void NormalizePts();
	void CombineContoursAndSigns(size_t slice_id);
	void CombineNormalAndSigns();

public:
	RBF_Energy_PARA e_para_;
	RBF_Energy rbf_e_;
	// GRBEnv* env_ = 0;
	// GRBVar* grb_x = 0;
	// GRBModel grb_model_;
	std::vector<std::vector<double>> slice_points_;
	std::vector<std::vector<double>> slice_normals_;
	std::vector<double> normal_signs_;
	std::vector<double> pts_all_;
	std::vector<double> normals_all_;
	arma::mat g_;
	size_t pt_num_;
	std::unordered_map<size_t, size_t> sign_index_map_;
};

