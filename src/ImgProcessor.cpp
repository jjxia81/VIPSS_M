#include "ImgProcessor.hpp"
#include <fstream>
#include <iostream>


void ShowImg(cv::Mat img, const std::string& imgName)
{
    double min,max;
    cv::minMaxLoc(img,&min,&max);
    auto new_img = img / max;
    // std::cout << "min max : " << min << " " << max << std::endl;
    cv::imshow(imgName, new_img);
    cv::waitKey(0);
}

void ImgProcessor::LoadImgStacks(const std::string& img_stack_path)
{
    this->img_stacks_.clear();
	cv::imreadmulti(img_stack_path, this->img_stacks_, cv::IMREAD_ANYDEPTH);
	// for (const auto& img : this->img_stacks_)
	// {
	// 	double min,max;
	// 	cv::minMaxLoc(img,&min,&max);
	// 	auto new_img = img / max;
	// 	// std::cout << "min max : " << min << " " << max << std::endl;
	// 	cv::imshow("first img", new_img);
	// 	cv::waitKey(0);
	// 	break;
	// }
    volume_z_max_ = img_stacks_.size();
    std::cout << "img stack num count : " << volume_z_max_ << std::endl;
    if(volume_z_max_ > 0)
    {
        volume_x_max_ = img_stacks_[0].cols;
        volume_y_max_ = img_stacks_[0].rows;
        std::cout << "img cols : " << volume_x_max_ << std::endl;
        std::cout << "img rows : " << volume_y_max_ << std::endl;
    }
}


void ImgProcessor::CalVolumeGradients()
{
    this->volume_gradient_z_.clear();
    size_t img_stack_num = img_stacks_.size();
    if(img_stack_num == 0) return;
    size_t img_rows = img_stacks_[0].rows;
    size_t img_cols = img_stacks_[0].cols;
    auto img_type = img_stacks_[0].type();
    for(size_t i = 0; i < img_stack_num; ++i)
    {
        // std::cout << " img id : " << i << std::endl;
        auto front_delt = cv::Mat::zeros(img_rows, img_cols, img_type);
        if (i > 0){
            front_delt = img_stacks_[i] - img_stacks_[i -1];
        }
        // std::cout << "start back end !" << std::endl;
        auto back_delt = cv::Mat::zeros(img_rows, img_cols, img_type);
        if (i < img_stack_num- 1){
            // std::cout << "start back end 00 !" << std::endl;
            back_delt = img_stacks_[i+ 1] - img_stacks_[i];
            // std::cout << "start back end 01!" << std::endl;
        }
        // if(i == 0)
        // {
        //     ShowImg(back_delt, "back_delt");
        // }
        auto gradient_z = (back_delt + front_delt) / 2.0;
        // std::cout << "gradient z success !" << std::endl;
        volume_gradient_z_.push_back(gradient_z);

        auto cur_img = img_stacks_[i];
        auto padding_col = cur_img.colRange(0, 1);
        auto left_block = cur_img.colRange(0, cur_img.cols -2);
        auto right_block = cur_img.colRange(2, cur_img.cols);
        auto x_gradient_block = (right_block - left_block) / 2.0;
        cv::Mat gradient_x;
        // auto padding_col = cv::Mat(img_rows, 1, img_type);
        cv::hconcat(padding_col, x_gradient_block, gradient_x);
        cv::hconcat(gradient_x, padding_col, gradient_x);
        

        

        volume_gradient_x_.push_back(gradient_x);
        // std::cout << "gradient x success !" << std::endl;

        auto padding_row = cur_img.rowRange(0, 1);
        auto top_block = cur_img.rowRange(0, cur_img.rows -2);
        auto bot_block = cur_img.rowRange(2, cur_img.rows);
        auto y_gradient_block = (bot_block - top_block) / 2.0;
        cv::Mat gradient_y;
        // auto padding_col = cv::Mat(img_rows, 1, img_type);
        cv::vconcat(padding_row, y_gradient_block, gradient_y);
        cv::vconcat(gradient_y, padding_row, gradient_y);

        // auto up_img = img_stacks_[i];
        // up_img(cv::Range(1, img_cols), cv::Range::all()) = img_stacks_[i](cv::Range(0, img_cols-1), cv::Range::all());
        // auto up_delt = img_stacks_[i] - up_img;

        // auto bot_img = img_stacks_[i];
        // bot_img(cv::Range(0, img_cols-1), cv::Range::all()) = img_stacks_[i](cv::Range(1, img_cols), cv::Range::all());
        // auto bot_delt = bot_img - img_stacks_[i];
        // auto gradient_y = (up_delt + bot_delt) / 2.0;
        volume_gradient_y_.push_back(gradient_y); 

        // if(i == 0)
        // {
        //     ShowImg(gradient_x, "gradient_x");
        //     ShowImg(gradient_y, "gradient_y");

        // }
        // std::cout << "gradient y success !" << std::endl;
    }

    // ShowImg(volume_gradient_z_[0], "volume_gradient_z_");
       
}



Vec3 ImgProcessor::GetGradient(size_t x_id, size_t y_id, size_t z_id)
{
    auto img_type = volume_gradient_z_[0].type();
    // std::cout << "img type " << img_type<< std::endl;
    if(x_id < volume_x_max_ && y_id < volume_y_max_ && z_id < volume_z_max_)
    {
        // std::cout << " ids :  "  << x_id << " " << y_id << " " << z_id << std::endl;
        auto grad_z_val = this->volume_gradient_z_[z_id].at<double>(y_id, x_id);
        auto grad_x_val = this->volume_gradient_x_[z_id].at<double>(y_id, x_id);
        auto grad_y_val = this->volume_gradient_y_[z_id].at<double>(y_id, x_id);
        auto gradient = Vec3(grad_x_val, grad_y_val, grad_z_val);
       
        return gradient;
    }
    return Vec3(0, 0,0);
}

Vec3 ImgProcessor::GetGradient(const Point3& point)
{
    size_t x_id = size_t(point.x + 0.5);
    size_t y_id = size_t(point.y + 0.5);
    size_t z_id = size_t(point.z + 0.5);
    return GetGradient(x_id, y_id, z_id);
}

void ImgProcessor::SampleVolumeGradients(size_t sample_step)
{
    volume_points_.clear();
    volume_gradients_.clear();
    max_gradient_norm_ = 0;
    for(size_t z_id = 0; z_id < volume_z_max_; z_id += sample_step)
    {
        for(size_t y_id = 0; y_id < volume_y_max_; y_id += sample_step)
        {
            for (size_t x_id = 0; x_id < volume_x_max_; x_id += sample_step)
            {
                auto new_p = Point3(x_id, y_id, z_id);
                auto new_grad = GetGradient(x_id, y_id, z_id);
                volume_points_.push_back(new_p);
                volume_gradients_.push_back(new_grad);
                float grad_norm = cv::norm(new_grad);
                max_gradient_norm_ = max_gradient_norm_ > grad_norm ? max_gradient_norm_ : grad_norm;
            }
        }
        // break;
    }
}


void ImgProcessor::SaveGradients(const std::string& path)
{
    // float sample_step = 2.0;
    // float gradient_scale = max_gradient_norm_ / sample_step_;

    std::fstream s_file{ path, s_file.out };

    for (size_t i = 0; i < volume_points_.size(); ++i)
    {
        auto point = volume_points_[i];
        // auto gradient = volume_gradients_[i] ;
        auto gradient = cv::normalize(volume_gradients_[i]);
        // if(cv::norm(volume_gradients_[i]) > 0)
        // {
        //     std::cout << "gradient value : " << gradient << std::endl;
        // }
        gradient = cv::normalize(gradient) * cv::norm(gradient) / max_gradient_norm_ * sample_step_ ;
        auto px = point.x + gradient[0];
        auto py = point.y + gradient[1];
        auto pz = point.z + gradient[2];
        auto new_p = Point3(px, py, pz);

        std::string p_str = std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z);
        s_file << "v " << p_str << std::endl;

        std::string e_str = std::to_string(new_p.x) + " " + std::to_string(new_p.y) + " " + std::to_string(new_p.z);
        s_file << "v " << e_str << std::endl;

       
    }

    for (size_t i = 0; i < volume_points_.size(); ++i)
    {
        std::string new_l = "l " + std::to_string(i *2 + 1) + " " + std::to_string(i *2 + 2);
        s_file << new_l << std::endl;
    }
}

void ImgProcessor::RunGradientsPipeline(const std::string& img_stack_path, const std::string& out_path)
{
    LoadImgStacks(img_stack_path);
    std::cout << " Load Img stacks success !" << std::endl;
    CalVolumeGradients();
    std::cout << " CalVolumeGradients success !" << std::endl;
    SampleVolumeGradients(sample_step_);
     std::cout << " SampleVolumeGradients success !" << std::endl;
    SaveGradients(out_path);
}


