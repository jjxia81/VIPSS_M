#pragma once
#include "rbf_energy.h"
#include "rbf_energy_para.h"
#include <set>

class NormalOptimizer
{
public:
	
	NormalOptimizer() {};
	~NormalOptimizer() {};
	void Init();
	void SetRBFPara();
	void LoadAllContours(const std::string& contour_path);
	void CombineContoursAndSigns();
	void SetRBFEnergyInput();
	void OptimizeNormalSigns();
	void NormalizePtCoords();
	void CombineContoursAndSigns(size_t slice_id);

public:
	RBF_Energy_PARA e_para_;
	RBF_Energy rbf_e_;
	std::vector<std::vector<double>> slice_points_;
	std::vector<std::vector<double>> slice_normals_;
	std::vector<double> normal_signs_;
	std::vector<double> pts_all_;
	std::vector<double> normals_all_;

};

