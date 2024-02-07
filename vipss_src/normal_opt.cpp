#include "normal_opt.h"
#include <filesystem>
#include <limits>
#include "rbfcore.h"
#include "readers.h"

namespace fs = std::filesystem;

void NormalOptimizer::Init()
{

}

void NormalOptimizer::SetRBFPara()
{

}

void NormalOptimizer::LoadAllContours(const std::string& contour_dir)
{
	this->slice_points_.clear();
	this->slice_normals_.clear();
	this->normal_signs_.clear();
	for (const auto& entry : fs::directory_iterator(contour_dir))
	{
		std::string pts_path = entry.path().string();
		std::vector<double> pts;
		std::vector<double> normals;
		readPLYFile(pts_path, pts, normals);
		std::cout <<"Read points path: " << pts_path << std::endl;
		std::cout << "current point num : " << pts.size() << std::endl;
		slice_points_.push_back(pts);
		slice_normals_.push_back(normals);
		normal_signs_.push_back(1.0);
	}	
}

void NormalOptimizer::CombineContoursAndSigns()
{
	pts_all_.clear();
	normals_all_.clear();
	for (size_t i = 0; i < slice_points_.size(); ++i)
	{
		auto& cur_normals = slice_normals_[i];
		auto& cur_points = slice_points_[i];
		double normal_sign = normal_signs_[i];
		for (size_t p_id = 0; p_id < cur_points.size(); ++p_id)
		{
			pts_all_.push_back(cur_points[p_id]);
			normals_all_.push_back(normal_sign * cur_normals[p_id]);
		}
	}
	std::cout << "All points num : " << pts_all_.size() << std::endl;
}

void NormalOptimizer::CombineContoursAndSigns(size_t slice_id)
{
	pts_all_.clear();
	normals_all_.clear();
	for (size_t i = 0; i < slice_id; ++i)
	{
		auto& cur_normals = slice_normals_[i];
		auto& cur_points = slice_points_[i];
		double normal_sign = normal_signs_[i];
		for (size_t p_id = 0; p_id < cur_points.size(); ++p_id)
		{
			pts_all_.push_back(cur_points[p_id]);
			normals_all_.push_back(normal_sign * cur_normals[p_id]);
		}
	}
	std::cout << "All points num : " << pts_all_.size() << std::endl;
}

//void NormalOptimizer::FlipNormals(size_t contour_id)
//{
//	auto& normals = slice_normals_[contour_id];
//	arma::Mat normals_mat((double*) normals.data(), normals.size());
//}


void NormalOptimizer::SetRBFEnergyInput()
{
	rbf_e_.SetPts(this->pts_all_, this->normals_all_);
}


void NormalOptimizer::OptimizeNormalSigns()
{
	LoadAllContours(e_para_.mesh_points_path);
	NormalizePtCoords();
	rbf_e_.SetRBFPara();

	e_para_.is_surfacing = false;
	e_para_.e_lambda = 0.0;
	rbf_e_.SetEnergyParameters(e_para_);
	for (size_t i = 1; i < normal_signs_.size(); ++i)
	{
		CombineContoursAndSigns(i);
		SetRBFEnergyInput();
		rbf_e_.InitRBFCore();
		/*std::cout << "--------pt number : " << pts_all_.size() << std::endl;*/
		rbf_e_.SolveRBF();
		double duchon_energy = rbf_e_.CalculateDuchonEnergy();
		std::cout << "duchon_energy : " << i << " " << duchon_energy << std::endl;

		normal_signs_[i] = -1;
		CombineContoursAndSigns(i);
		SetRBFEnergyInput();
		rbf_e_.InitRBFCore();
		rbf_e_.SolveRBF();
		double duchon_energy2 = rbf_e_.CalculateDuchonEnergy();
		std::cout << "duchon_energy2 : " << i << " "  << duchon_energy2 << std::endl;

		if (duchon_energy < duchon_energy2)
		{
			normal_signs_[i] = 1;
		}
	}
	e_para_.is_surfacing = true;
	e_para_.e_lambda = 0.00001;
	rbf_e_.SetEnergyParameters(e_para_);
	CombineContoursAndSigns(this->slice_points_.size()-1);
	SetRBFEnergyInput();
	rbf_e_.InitRBFCore();
	rbf_e_.SolveRBF();

	for (auto sign : normal_signs_)
	{
		std::cout << sign << std::endl;
	}
	
}

void NormalOptimizer::NormalizePtCoords()
{
	double DMIN = std::numeric_limits<double>::min();
	double DMAX = std::numeric_limits<double>::max();
	double max_x = DMIN, max_y = DMIN, max_z = DMIN;
	double min_x = DMAX, min_y = DMAX, min_z = DMAX;

	for(const auto& pts : slice_points_)
	{
		for(size_t i = 0; i < pts.size()/3; ++i)
		{
			max_x = std::max(pts[3*i], max_x);
			max_y = std::max(pts[3*i + 1], max_y);
			max_z = std::max(pts[3*i + 2], max_z);

			min_x = std::min(pts[3*i], min_x);
			min_y = std::min(pts[3*i + 1], min_y);
			min_z = std::min(pts[3*i + 2], min_z);
		}
	} 
	double delt_x = max_x - min_x;
	double delt_y = max_y - min_y;
	double delt_z = max_z - min_z;
	std::string out_path = "E:\\projects\\All_Elbows\\output\\";
	for(size_t s_id = 0; s_id < slice_points_.size(); ++s_id)
	{
		auto& pts = slice_points_[s_id];
		for(size_t i = 0; i < pts.size()/3; ++i)
		{
			pts[3*i] = (pts[3*i] - min_x) / delt_x;
			pts[3*i + 1] = (pts[3*i + 1] - min_y) / delt_y;
			pts[3*i + 2] = (pts[3*i + 2] - max_z) / delt_z;
		}
		std::string slice_out_path = out_path + std::to_string(s_id) + ".ply";
		writePLYFile(slice_out_path, pts);
	}
}

// void solve_vipss()
// {
// 	vector<double>Vs;
//     RBF_Core rbf_core;
//     RBF_Paras para = Set_RBF_PARA();
//     para.user_lamnbda = user_lambda;

//     readXYZ(infilename,Vs);
//     rbf_core.InjectData(Vs,para);
//     rbf_core.BuildK(para);
//     rbf_core.InitNormal(para);
//     rbf_core.OptNormal(0);

//     rbf_core.Write_Hermite_NormalPrediction(outpath+pcname+"_normal", 1);

//     if(is_surfacing){
//         rbf_core.Surfacing(0,n_voxel_line);
//         rbf_core.Write_Surface(outpath+pcname+"_surface");
//     }

//     if(is_outputtime){
//         rbf_core.Print_TimerRecord_Single(outpath+pcname+"_time.txt");
//     }
//     return 0;
// }