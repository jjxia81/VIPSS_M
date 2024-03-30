#include "normal_opt.h"
#include <filesystem>
#include <limits>
#include "rbfcore.h"
#include "readers.h"
#include "gurobi/gurobi_c++.h"
#include <chrono>
namespace fs = std::filesystem;
typedef std::chrono::high_resolution_clock Clock;

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
	pt_num_ = 0;
	size_t c_count = 0;
	for (const auto& entry : fs::directory_iterator(contour_dir))
	{
		std::string pts_path = entry.path().string();
		std::vector<double> pts;
		std::vector<double> normals;
		readPLYFile(pts_path, pts, normals);
		std::cout <<"Current contour id: " << c_count << std::endl;
		std::cout <<"Read points path: " << pts_path << std::endl;
		std::cout << "current point num : " << pts.size() << std::endl;
		slice_points_.push_back(pts);
		slice_normals_.push_back(normals);
		pt_num_ += pts.size()/3;
		normal_signs_.push_back(1.0);
		c_count ++;
	}
	std::cout << "Input data Contour number : " << slice_normals_.size() << std::endl;
	normals_all_.resize(pt_num_ * 3);
	g_.resize(1, pt_num_ * 3);
}

void NormalOptimizer::CombineContoursAndSigns()
{
	size_t id = 0;
	for (size_t i = 0; i < slice_points_.size(); ++i)
	{
		auto& cur_normals = slice_normals_[i];
		auto& cur_points = slice_points_[i];
		for (size_t p_id = 0; p_id < cur_points.size()/3; ++p_id)
		{
			sign_index_map_[id] = i;
			sign_index_map_[id + pt_num_] = i;
			sign_index_map_[id + pt_num_ * 2] = i;

			pts_all_.push_back(cur_points[p_id * 3]);
			pts_all_.push_back(cur_points[p_id * 3 + 1]);
			pts_all_.push_back(cur_points[p_id * 3 + 2]);
			normals_all_[id] = cur_normals[p_id * 3];
			normals_all_[id + pt_num_ ] = cur_normals[p_id * 3 + 1];
			normals_all_[id + pt_num_ * 2] = cur_normals[p_id * 3 + 2];

			//normals_all_.push_back(cur_normals[p_id]);
			g_(0, id) = cur_normals[p_id * 3];
			g_(0, id + pt_num_) = cur_normals[p_id * 3 + 1];
			g_(0, id + pt_num_ * 2) = cur_normals[p_id * 3 + 2];
			id++;
		}
	}
	std::cout << "All points num : " << pts_all_.size() << std::endl;
}

void NormalOptimizer::CombineNormalAndSigns()
{
	size_t id = 0;
	normals_all_.clear();
	for (size_t i = 0; i < slice_points_.size(); ++i)
	{
		auto& cur_normals = slice_normals_[i];
		double n_sign = normal_signs_[i];
		// n_sign = 1.0;
		for (size_t p_id = 0; p_id < cur_normals.size()/3; ++p_id)
		{
			normals_all_.push_back(cur_normals[p_id * 3] * n_sign);
			normals_all_.push_back(cur_normals[p_id * 3 + 1] * n_sign);
			normals_all_.push_back(cur_normals[p_id * 3 + 2] * n_sign);
		}
	}
}

// void NormalOptimizer::CombineContoursAndSigns(size_t slice_id)
// {
// 	pts_all_.clear();
// 	normals_all_.clear();
// 	for (size_t i = 0; i < slice_id; ++i)
// 	{
// 		auto& cur_normals = slice_normals_[i];
// 		auto& cur_points = slice_points_[i];
// 		double normal_sign = normal_signs_[i];
// 		for (size_t p_id = 0; p_id < cur_points.size(); ++p_id)
// 		{
// 			pts_all_.push_back(cur_points[p_id]);
// 			normals_all_.push_back(normal_sign * cur_normals[p_id]);
// 		}
// 	}
// 	std::cout << "All points num : " << pts_all_.size() << std::endl;
// }


void NormalOptimizer::SetRBFEnergyInput()
{
	// rbf_e_.SetPts(this->pts_all_, this->normals_all_);
}


void NormalOptimizer::OptimizeNormalSigns()
{
	std::cout << " file dir " << e_para_.mesh_points_path << endl;
	LoadAllContours(e_para_.mesh_points_path);
	GRBEnv* env = 0;
	GRBVar* x = 0;
	env = new GRBEnv();
	GRBModel model = GRBModel(*env);
	model.set(GRB_StringAttr_ModelName, "normal_opt");
	// Add variables, set bounds and obj coefficients
	size_t var_num = slice_points_.size();
	x = model.addVars(var_num, GRB_BINARY);

	double re_time;
	auto t0 = Clock::now();
	CombineContoursAndSigns();
	NormalizePts();
	SetRBFEnergyInput();
	rbf_e_.e_para_ = e_para_;
	rbf_e_.rbf_para_.user_lamnbda = e_para_.e_lambda;

	std::cout << "------------------- lambda 0 : " << rbf_e_.rbf_para_.user_lamnbda << std::endl;
	rbf_e_.InitRBFCore();
	rbf_e_.rbf_core_.BuildK(rbf_e_.rbf_para_);
	arma::mat& final_H = rbf_e_.rbf_core_.finalH;
	auto t1 = Clock::now();
	cout << "UpdateGradient time: " << (re_time = std::chrono::nanoseconds(t1 - t0).count() / 1e9) << endl;

	arma::mat loss_mat = g_ * final_H * g_.t();
	std::cout << "-------------------loss_mat : " << loss_mat(0) << endl;
	auto t2 = Clock::now();
	
	GRBQuadExpr obj_sum = 0;
	std::vector<GRBLinExpr> signed_normals;
	for (size_t i = 0; i < normals_all_.size(); ++i)
	{
		// std::cout << " i " << i << " idex " << sign_index_map_[i] << endl;
		GRBLinExpr obj_n = (1 - x[sign_index_map_[i]] * 2) * normals_all_[i];
		signed_normals.push_back(obj_n);
	}

	for (size_t i = 0; i < normals_all_.size(); ++i)
	{
		GRBLinExpr row_e = 0;
		for (size_t j = 0; j < normals_all_.size(); ++j)
		{
			row_e += signed_normals[j] * final_H(j, i);
		}
		obj_sum += row_e * signed_normals[i];
	}

	model.setObjective(obj_sum, GRB_MINIMIZE);
	model.optimize();
	auto t3 = Clock::now();
	cout << "---- Normal Direction time: " << (re_time = std::chrono::nanoseconds(t3 - t2).count() / 1e9) << endl;
	normal_signs_.clear();
	for (int j = 0; j < var_num; j++) {
		cout << "x[" << j << "] = " << x[j].get(GRB_DoubleAttr_X) << endl;
		if(x[j].get(GRB_DoubleAttr_X) == 1)
		{
			normal_signs_.push_back(-1.0);
		} else {
			normal_signs_.push_back(1.0);
		}
	}
	CombineNormalAndSigns();
	SetRBFEnergyInput();
	// rbf_e_.SetEnergyParameters(e_para_);
	// rbf_e_.pts_ = pts_all_;
	// rbf_e_.gradients_ = normals_all_;
	// rbf_e_.pt_n_ = pts_all_.size() /3;

	// rbf_e_.RunTestWithOptNormal();
}

void NormalOptimizer::NormalizePts()
{
    double DMIN = std::numeric_limits<double>::min();
    double DMAX = std::numeric_limits<double>::max();
    double max_x = DMIN, max_y = DMIN, max_z = DMIN;
    double min_x = DMAX, min_y = DMAX, min_z = DMAX;
    for (size_t i = 0; i < pts_all_.size() / 3; ++i)
    {
        max_x = std::max(pts_all_[3 * i], max_x);
        max_y = std::max(pts_all_[3 * i + 1], max_y);
        max_z = std::max(pts_all_[3 * i + 2], max_z);

        min_x = std::min(pts_all_[3 * i], min_x);
        min_y = std::min(pts_all_[3 * i + 1], min_y);
        min_z = std::min(pts_all_[3 * i + 2], min_z);
    }
    double delt_x = (max_x - min_x) / 2.0;
    double delt_y = (max_y - min_y) / 2.0;
    double delt_z = (max_z - min_z) / 2.0;

    double scale = std::max(std::max(delt_x, delt_y), delt_z);

    double origin_x = (max_x + min_x) / 2;
    double origin_y = (max_y + min_y) / 2;
    double origin_z = (max_z + min_z) / 2;

    for (size_t i = 0; i < pts_all_.size() / 3; ++i)
    {
        pts_all_[3 * i] = (pts_all_[3 * i] - origin_x) / scale;
        pts_all_[3 * i + 1] = (pts_all_[3 * i + 1] - origin_y) / scale;
        pts_all_[3 * i + 2] = (pts_all_[3 * i + 2] - origin_z) / scale;
    }

    std::string normalized_pts_path = e_para_.out_dir + "_opt_normalized_pts.ply";
    writePLYFile_VN(normalized_pts_path, pts_all_, normals_all_);
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