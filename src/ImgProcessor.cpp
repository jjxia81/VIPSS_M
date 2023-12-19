#include "ImgProcessor.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <set>
#include "opencv2/imgproc.hpp"

#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;

void ShowImg(cv::Mat img, const std::string& imgName)
{
    // double min,max;
    // cv::minMaxLoc(img,&min,&max);
    // auto new_img = img / max;
    // std::cout << "min max : " << min << " " << max << std::endl;
    cv::imshow(imgName, img);
    cv::waitKey(0);
}

void ImgProcessor::LoadImgStacks(const std::string& img_stack_path)
{
    this->img_stacks_.clear();
	cv::imreadmulti(img_stack_path, this->img_stacks_, cv::IMREAD_ANYDEPTH);
    volume_z_max_ = img_stacks_.size();
    std::cout << "img stack num count : " << volume_z_max_ << std::endl;
    if(volume_z_max_ > 0)
    {
        volume_x_max_ = img_stacks_[0].cols;
        volume_y_max_ = img_stacks_[0].rows;
        std::cout << "img cols : " << volume_x_max_ << std::endl;
        std::cout << "img rows : " << volume_y_max_ << std::endl;
    }
    this->img_stacks_grey_.clear();
    for(const auto& img : img_stacks_)
    {
        double minVal; 
        double maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        cv::Mat new_img = (img - minVal) / (maxVal - minVal) * 255;
        new_img.convertTo(new_img, CV_8U);
        cv::Mat blur_img;
        blur( new_img, blur_img, cv::Size(3,3));
        this->img_stacks_grey_.push_back(blur_img);
    }
    
}

void ImgProcessor::FilterImgStacksWithGaussian()
{
    int kenerl_size = 3;
    for(auto & img : this->img_stacks_)
    {
        cv::GaussianBlur(img, img, cv::Size(kenerl_size, kenerl_size), 0);
    }
}


void ImgProcessor::LoadImgStacksMask(const std::string& img_stack_mask_path)
{
    this->img_mask_stacks_.clear();
    // std::cout << "mask path : " << img_stack_mask_path <<
	cv::imreadmulti(img_stack_mask_path, this->img_mask_stacks_, cv::IMREAD_ANYDEPTH);
    if(this->img_mask_stacks_.empty())
    {
        std::cout << " Load image mask stack failed !" << std::endl;
    } else {
        std::cout << " Load image mask stack succeed !" << std::endl;
        std::cout << " Image mask size : "<< this->img_mask_stacks_.size() << std::endl;
    }
}

void ImgProcessor::LoadImgContourSamplePoints(const std::string& img_contour_sample_points_path)
{
    this->contour_sample_points_.clear();
    std::string line;
    std::ifstream myfile(img_contour_sample_points_path);
    if (myfile.is_open())
    {
        std::cout << " Load Contour Sample Points succeed !" << std::endl;
        this->contour_sample_points_.clear();
        while ( getline (myfile,line) )
        {
            // std::cout << line << '\n';
            std::string word = "";
            std::vector<int> coords;
            for (auto s_i : line)
            {
                if (s_i == ' ')
                {
                   
                    coords.push_back(std::stoi(word));
                    word = "";
                }
                else {
                    word = word + s_i;
                }
            }
            coords.push_back(std::stoi(word));

            // std::cout << "coords size : " << coords.size() << std::endl;
            if(coords.size() == 3)
            {
                Point3i new_p(coords[0], coords[1], coords[2]);
                this->contour_sample_points_.push_back(new_p);
            }
        
        }
        myfile.close();
        std::cout << " Load Contour Sample Points number: "<<  this->contour_sample_points_.size() << std::endl;
    }
}

Mat ImgProcessor::CalStructureTensor(const Vec3& gradeint)
{
    Mat tensor(3,3, CV_32F);
    Mat grad(gradeint);
    std:: cout << "grad " << grad << std::endl;
    tensor =  grad * grad.t();
    return tensor;
}


void ImgProcessor::CalSamplePointsStructureTensor()
{
    for(size_t i = 0; i < this->contour_sample_points_gradients_.size(); ++i)
    {
        const auto& gradient = contour_sample_points_gradients_[i];
        std::cout << "gradient " << gradient << std::endl;
        Mat tensor = CalStructureTensor(gradient);
        std::cout << "tensor " << tensor << std::endl;
    }

}

Mat ImgProcessor::CalStructureTensorGaussain(const Point3i& point)
{
    int k_size = 3;
    int half = k_size / 2;
    auto cur_gradient = CalPointGradient(point);

    Mat tensor_avg(3, 3, CV_32F);
    float params[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    Mat Gaussian2D(3, 3, CV_32F, params);
    // Gaussian2D = Gaussian2D / cv::sum(Gaussian2D);
    std::vector<Mat> Gaussian3D;
    float sum = cv::sum(Gaussian2D)[0] * 2;
    Gaussian3D.push_back(Gaussian2D * 0.5 / sum);
    Gaussian3D.push_back(Gaussian2D / sum);
    Gaussian3D.push_back(Gaussian2D * 0.5 / sum);

    for(int z_d = -half; z_d <= half; ++z_d)
    {
        for(int y_d = -half; y_d <= half; ++y_d)
        {
            for(int x_d = -half; x_d <= half; ++ x_d)
            {
                int x = std::max(0, x_d + point.x);
                x = x < volume_x_max_ ? x : volume_x_max_;

                int y = std::max(0, y_d + point.y);
                y = y < volume_y_max_ ? y : volume_y_max_;

                int z = std::max(0, z_d + point.z);
                z = z < volume_z_max_ ? z : volume_z_max_;
                Point3i new_p(x, y, z);
                auto gradient = CalPointGradient(new_p);
                Mat tensor = CalStructureTensor(gradient);
                float g_param = Gaussian3D[z_d + half].at<float> (y_d + half, x_d + half);
                tensor_avg = tensor_avg + tensor *  g_param;
            }
        }
    }
    return tensor_avg;     
}




Vec3 ImgProcessor::CalPointGradient(const Point3i& point)
{
        const auto& img_p = point; 
        // std::cout << img_p.x << " " << img_p.y << " " << img_p.z<< std::endl;
        double d_x = 0;
        double d_y = 0;
        double d_z = 0;
        if(img_p.z == 0)
        {
            double z_cur  = this->img_stacks_[img_p.z].at<double>(img_p.y, img_p.x);
            double z_back = this->img_stacks_[img_p.z + 1].at<double>(img_p.y, img_p.x);
            d_z =  z_back - z_cur;
        } else if (img_p.z == this->img_stacks_.size() - 1){
            double z_cur  = this->img_stacks_[img_p.z].at<double>(img_p.y, img_p.x);
            double z_pre = this->img_stacks_[img_p.z - 1].at<double>(img_p.y, img_p.x);
            d_z = z_cur - z_pre;
        } else {
           
            double z_pre = this->img_stacks_[img_p.z - 1].at<double>(img_p.y, img_p.x);
            double z_back = this->img_stacks_[img_p.z + 1].at<double>(img_p.y, img_p.x);
            d_z = (z_back - z_pre)/ 2.0;
        }
        const auto& cur_img = this->img_stacks_[img_p.z];
        if(img_p.x == 0)
        {
            double x_cur  =cur_img.at<double>(img_p.y, img_p.x);
            double x_right = cur_img.at<double>(img_p.y, img_p.x + 1);
            d_x =  x_right - x_cur;
        } else if (img_p.x == cur_img.cols - 1){
            double x_cur  = cur_img.at<double>(img_p.y, img_p.x);
            double x_left  = cur_img.at<double>(img_p.y, img_p.x - 1);
            d_x = x_left - x_cur;
        } else {
            double x_left = cur_img.at<double>(img_p.y, img_p.x - 1 );
            double x_right = cur_img.at<double>(img_p.y, img_p.x + 1);
            d_x = (x_right - x_left)/ 2.0;
        }

        if(img_p.y == 0)
        {
            double y_cur  =cur_img.at<double>(img_p.y, img_p.x);
            double y_top = cur_img.at<double>(img_p.y + 1, img_p.x);
            d_y =  y_top - y_cur;
        } else if (img_p.y == cur_img.rows - 1){
            double y_cur  =cur_img.at<double>(img_p.y, img_p.x);
            double y_bot = cur_img.at<double>(img_p.y - 1, img_p.x);
            d_y =  y_cur - y_bot;
        } else {
            double y_bot = cur_img.at<double>(img_p.y - 1, img_p.x);
            double y_top = cur_img.at<double>(img_p.y + 1, img_p.x);
            d_y = (y_top - y_bot)/ 2.0;
        }
     
        Vec3 gradient(d_x, d_y, d_z);
        return gradient;
}

void ImgProcessor::CalSamplePointsGradients()
{
    // std::cout << " img stack size : " << img_stacks_.size() << std::endl;
    // std::cout << " img stack  data type  : "  << img_stacks_[0].type();
    this->contour_sample_points_gradients_.clear();
    max_gradient_norm_ = 0;
    for(size_t i = 0; i < this->contour_sample_points_.size(); ++i)
    {
        const auto& img_p = contour_sample_points_[i]; 
        // std::cout << img_p.x << " " << img_p.y << " " << img_p.z<< std::endl;
        Vec3 gradient = CalPointGradient(img_p);
        this->contour_sample_points_gradients_.push_back(gradient);
        float g_len = cv::norm(gradient);
        max_gradient_norm_ = max_gradient_norm_ > g_len ? max_gradient_norm_ : g_len;
    }
    std::cout << " Cal Contour Sample Points  gradient succeed, with gradient number: "<<  this->contour_sample_points_gradients_.size() << std::endl;

}

void ImgProcessor::CalSamplePointsTangentVector()
{
    this->contour_sample_points_tangents_.clear();
    size_t p_size = this->contour_sample_points_.size();
    if(p_size < 3) return; 
    for(size_t i = 0; i < p_size; ++i)
    {
        size_t pre_id = i - 1 < 0 ? i - 1 + p_size : i - 1;
        size_t next_id = i + 1 >= p_size ? i + 1 - p_size : i + 1;
        auto tangent_dir = this->contour_sample_points_[next_id] - this->contour_sample_points_[pre_id];
        Vec3 tangent(tangent_dir.x, tangent_dir.y, tangent_dir.z);
        tangent = tangent / sqrt(tangent.dot(tangent));
        this->contour_sample_points_tangents_.push_back(tangent);
    }
}

void ImgProcessor::SaveImgContourSamplePoints(const std::string& img_contour_sample_points_path)
{
    std::fstream s_file{ img_contour_sample_points_path, s_file.out };
    const auto& c_points = this->contour_sample_points_;
    for (size_t i = 0; i < c_points.size(); ++i)
    {
        auto point = c_points[i];
        // auto gradient = volume_gradients_[i] ;
        std::string p_str = std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z);
        s_file << "v " << p_str << std::endl;
    }
}


void ImgProcessor::CalSamplePointsGradientsTangentProjection()
{
    this->contour_sample_points_projection_.clear();
    if(this->contour_sample_points_gradients_.empty() || this->contour_sample_points_tangents_.empty())
    {

        std::cout << " sample_points_gradients or sample_points_tangents are empty !!!" << std::endl;
        return;
    }
    
    if(this->contour_sample_points_gradients_.size() != this->contour_sample_points_tangents_.size())
    {
        std::cout << " sample_points_gradients size : " << this->contour_sample_points_gradients_.size() << std::endl;
        std::cout << " sample_points_tangents size : " << this->contour_sample_points_tangents_.size() << std::endl;
        
        std::cout << " sample_points_gradients and sample_points_tangents do not have same size !!!" << std::endl;
        return;
    }
    for(size_t i =0; i < this->contour_sample_points_tangents_.size(); ++i)
    {
        auto project = this->contour_sample_points_tangents_[i].dot(contour_sample_points_gradients_[i]);
        this->contour_sample_points_projection_.push_back(fabs(project));
    }
}


void ImgProcessor::SaveImgContourSamplePointsWithGradients(const std::string& img_contour_sample_points_path)
{
    std::fstream s_file{ img_contour_sample_points_path, s_file.out };
    const auto& c_points = this->contour_sample_points_;
    const auto& c_gradients = this->contour_sample_points_gradients_; 

    if(this->contour_sample_points_projection_.empty()) return;
    float min_project = this->contour_sample_points_projection_[0];
    float max_project = this->contour_sample_points_projection_[0];
    for(auto project : this->contour_sample_points_projection_)
    {
        min_project = min_project < project ? min_project : project;
        max_project = max_project > project ? max_project : project;
    }

    for (size_t i = 0; i < c_points.size(); ++i)
    {
        auto point = c_points[i];
        // auto gradient = volume_gradients_[i] ;
        auto gradient = (c_gradients[i]);
        // if(cv::norm(volume_gradients_[i]) > 0)
        // {
        //     std::cout << "gradient value : " << gradient << std::endl;
        // }
        gradient = cv::normalize(gradient) * cv::norm(gradient) / this->max_gradient_norm_ * 10;
        auto px = point.x + gradient[0];
        auto py = point.y + gradient[1];
        auto pz = point.z + gradient[2];
        auto new_p = Point3(px, py, pz);

        auto project = this->contour_sample_points_projection_[i];
        float color_r = (project - min_project) / (max_project - min_project) ;
        float color_b = (max_project - project) / (max_project - min_project) ;
        std::string color_str =  " " + std::to_string(color_r) + " " + std::to_string(color_b) + " " + std::to_string(0.1);

        std::string p_str = std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z);
        s_file << "v " << p_str << color_str << std::endl;

        std::string e_str = std::to_string(new_p.x) + " " + std::to_string(new_p.y) + " " + std::to_string(new_p.z);
        s_file << "v " << e_str << color_str << std::endl;

        std::string m_str = std::to_string((point.x + new_p.x)/2) + " " + std::to_string((point.y + new_p.y)/2) 
            + " " + std::to_string((point.z + new_p.z)/2);
        s_file << "v " << m_str << color_str << std::endl;
    }

    for (size_t i = 0; i < c_points.size(); ++i)
    {
        // std::string new_l = "l " + std::to_string(i *2 + 1) + " " + std::to_string(i *2 + 2);
        std::string new_l = "f " + std::to_string(i *3 + 1) + " " + std::to_string(i *3 + 2) + " " + std::to_string(i *3 + 3);
        s_file << new_l << std::endl;
    }
}


void ImgProcessor::SaveOriginalImgs(const std::string& path)
{
    for(size_t i =0; i <this->img_stacks_.size(); ++i)
    {
        std::string out_img_path = path + std::to_string(i) + ".png";
        auto img = img_stacks_[i];
        double minVal; 
        double maxVal;
        minMaxLoc(img, &minVal, &maxVal);
        cv::Mat new_img =  (img - minVal) / (maxVal - minVal) * 255;
        // cv::Mat new_img = img;
        
        new_img.convertTo(new_img, CV_8U);
        // ShowImg(new_img, "new img");
        cv::imwrite(out_img_path, new_img);
        
    }
}

void ImgProcessor::SaveOriginalImgsWithContourPoints(const std::string& path)
{
    std::set<int> image_ids;
    for(const auto& img_p : this->contour_sample_points_)
    {
        image_ids.insert(img_p.z);
    }
    std::vector<cv::Mat> new_images;
    for(size_t i =0; i <this->img_stacks_.size(); ++i)
    {
        auto img = img_stacks_[i];
        double minVal; 
        double maxVal;
        minMaxLoc(img, &minVal, &maxVal);
        cv::Mat new_img =  (img - minVal) / (maxVal - minVal) * 255;
        new_img.convertTo(new_img, CV_8U);
        cv::Mat color_img;
        cvtColor(new_img, color_img, cv::COLOR_GRAY2BGR);
        new_images.push_back(color_img);
    }
    float min_score = 0; 
    float max_score = 400;
    for(size_t p_id = 0; p_id < this->contour_sample_points_.size(); ++p_id)
    {
        const auto& cur_p = contour_sample_points_[p_id];
        auto& cur_img = new_images[cur_p.z];
        cv::Vec3b& color = cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x);
        // float score = this->contour_points_sobel_scores_[p_id];

        // float new_score = std::min(std::max(min_score, score), max_score);
        float score = img_stacks_sobels_[cur_p.z].at<float>(cur_p.y, cur_p.x);
        float new_score = std::min(std::max(min_score, score), max_score);
        float ratio = (new_score - min_score) / max_score;
        // std::cout << "score  " << score << std::endl;
        // std::cout << "new_score  " << new_score << std::endl;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[0] = 0;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[1] = uchar(255 * ratio);;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[2] = uchar(255 * (1 -ratio));

    }

    for(auto id : image_ids)
    {
        std::string out_img_path = path + "_" + std::to_string(id) + ".png";
        cv::imwrite(out_img_path, new_images[id]);
    }
}

void ImgProcessor::SaveSobelImgsWithContourPoints(const std::string& path)
{
    std::set<int> image_ids;
    for(const auto& img_p : this->contour_sample_points_)
    {
        image_ids.insert(img_p.z);
    }
    std::vector<cv::Mat> new_images;
    for(size_t i =0; i <this->img_stacks_sobels_.size(); ++i)
    {
        auto img = img_stacks_sobels_[i];
        double minVal; 
        double maxVal;
        minMaxLoc(img, &minVal, &maxVal);
        cv::Mat new_img =  (img - minVal) / (maxVal - minVal) * 255;
        new_img.convertTo(new_img, CV_8U);
        cv::Mat color_img;
        cvtColor(new_img, color_img, cv::COLOR_GRAY2BGR);
        new_images.push_back(color_img);
    }
    
    float min_score = 0; 
    float max_score = 300;
    for(size_t p_id = 0; p_id < this->contour_sample_points_.size(); ++p_id)
    {
        const auto& cur_p = contour_sample_points_[p_id];
        auto& cur_img = new_images[cur_p.z];

        float sum_score = 0;
        for(int ri = -1; ri < 2; ++ri)
        {
            for(int ci = -1; ci < 2; ++ci)
            {
                int y_i = std::min(cur_img.rows-1, std::max(0, cur_p.y + ri));
                int x_i = std::min(cur_img.cols-1, std::max(0, cur_p.x + ci));
                sum_score += img_stacks_sobels_[cur_p.z].at<float>(y_i, x_i);
            }    
        }
        float score = sum_score / 9.0;
        // float score = img_stacks_sobels_[cur_p.z].at<float>(cur_p.y, cur_p.x);
        float new_score = std::min(std::max(min_score, score), max_score);
        float ratio = (new_score - min_score) / max_score;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[0] = 0;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[1] = uchar(255 * ratio);;
        cur_img.at<cv::Vec3b>(cur_p.y, cur_p.x)[2] = uchar(255 * (1 -ratio));
    }

    for(auto id : image_ids)
    {
        std::string out_img_path = path + "_" + std::to_string(id) + ".png";
        cv::imwrite(out_img_path, new_images[id]);
    }
}

void ImgProcessor::test_sobel_score()
{
    auto& cur_img = this->img_stacks_grey_[0];
    cv::Mat new_img = cur_img;
    cv::imshow("new image 00",  new_img);
    cv::waitKey(0);
    float min_score = 0; 
    float max_score = 500;
    // for(size_t i = 0; i < cur_img.rows; ++i)
    // {
    //     for(size_t j =0; j < cur_img.cols; ++j)
    //     {
    //         Point3i new_p(j, i, 0);
    //         float score = CalSobelScoreByPoint(new_p);
    //         float new_score = std::min(std::max(min_score, score), max_score);
    //         float ratio = (new_score - min_score) / max_score;
    //         new_img.at<uchar>(i, j) = uchar(int(ratio * 255));
    //     }
    // }
    Mat gradX, gradY;
    Sobel(new_img, gradX, CV_32F, 1, 0, 3);
    Sobel(new_img, gradY, CV_32F, 0, 1, 3);
     // Compute the magnitude of the gradient
    Mat magnitude;
    magnitude = abs(gradX) + abs(gradY);

    // Normalize the magnitude to display it as an image
    Mat normalizedMagnitude;
    normalize(magnitude, normalizedMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display the original image and the edge-detected image
    cv::imshow("Original Image", new_img);
    cv::imshow("Edge-Detected Image", normalizedMagnitude);
    cv::waitKey(0);

}

cv::RNG rng(12345);

void ImgProcessor::ScanImageEdgeDetection()
{
    size_t img_id = 0; 
    for(size_t img_id = 0; img_id < this->img_stacks_.size(); ++ img_id)
    {
        if(img_id != 52) continue;
        const auto& img = this->img_stacks_[img_id];
        double minVal; 
        double maxVal; 
        cv::Point minLoc; 
        cv::Point maxLoc;

        minMaxLoc(img, &minVal, &maxVal);
        

        cv::Mat new_img = img; //(img - minVal) / (maxVal - minVal) * 255;
        int lowThreshold = 0;
        const int max_lowThreshold = 150;
        const int ratio = 3;
        const int kernel_size = 3;
        const char* window_name = "Edge Map";

        // new_img.convertTo(new_img, CV_8U);
        auto img_type = new_img.type();


        cv::Mat data;
        new_img.convertTo(data,CV_32F);
        data = data.reshape(1,data.total());

        int K = 3;
        // Term criteria for K-means algorithm
        cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.2);
        cv::Mat labels, centers;
        cv::kmeans(data, K, labels, termCriteria, 3, cv::KMEANS_RANDOM_CENTERS, centers);
        // Reshape the labels back to the original image size
        labels = labels.reshape(0, new_img.rows);
        cv::Mat colormap(K, 1, CV_8U);
        cv::RNG rng(12345);
        for (int i = 0; i < K; ++i) {
            colormap.at<uchar>(i, 0) = i * (256 / K);
        }
        cv::Mat segmentedImage(new_img.size(), CV_8U);
        for (int i = 0; i < new_img.rows; ++i) {
            for (int j = 0; j < new_img.cols; ++j) {
                int clusterLabel = labels.at<int>(i, j);
                segmentedImage.at<uchar>(i, j) = colormap.at<uchar>(clusterLabel, 0);
            }
        }

        ShowImg(segmentedImage, "new segmentedImage");

        segmentedImage.convertTo(new_img, CV_8U);
        ShowImg(new_img, "new new_img");

        std::cout << " first img id  " << std::endl;
        // auto front_delt = cv::Mat::zeros(img_rows, img_cols, img_type);
        cv::Mat detected_edges = cv::Mat::zeros(img.size(), img_type);
        cv::Canny( new_img, detected_edges, max_lowThreshold, max_lowThreshold*ratio, kernel_size );

        const auto& img_mask = this->img_mask_stacks_[img_id];
        // ShowImg(img_mask, "new mask");
        cv::Mat new_mask =  img_mask * 255;
        new_mask.convertTo(new_mask, CV_8U);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        // ShowImg(new_mask, "new mask");
        findContours( new_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        std::cout << " find contour succeed " << std::endl;

        // detected_edges.convertTo(detected_edges, cv::CV_8UC3);
        // cvtColor(detected_edges, detected_edges, cv::CV_g);
        cv::Mat new_edges = cv::Mat::zeros(img.size(), img_type);
        cv::cvtColor(detected_edges, new_edges, cv::COLOR_GRAY2BGR);
        for( size_t i = 0; i< contours.size(); i++ )
        {
            cv::Scalar color = cv::Scalar(255, 122, 242,  0.5);
            drawContours(new_edges, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0 );
        }

        // dst = Scalar::all(0);
        std::cout << " visual img " << std::endl;
        // src.copyTo( dst, detected_edges);
        ShowImg(detected_edges, "new edges");
        ShowImg(new_edges, "contours");
        break;

        // std::string out_img_path = this->contour_mask_overlay_output_path_ + "contour_mask_" + std::to_string(img_id) +".png";
        // cv::imwrite(out_img_path,new_edges);

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
        gradient = cv::normalize(gradient) * cv::norm(gradient) / 1000 * sample_step_ ;
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

void ImgProcessor::CalSobelScore()
{
    this->volume_gradient_z_.clear();
    size_t img_stack_num = img_stacks_.size();
    if(img_stack_num == 0) return;
    // auto img_type = img_stacks_[0].type();
    this->img_stacks_sobels_.clear();

    for(size_t i = 0; i < img_stacks_grey_.size(); ++i)
    {
        const auto& cur_img = img_stacks_grey_[i];
        Mat gradX, gradY;
        Sobel(cur_img, gradX, CV_32F, 1, 0, 3);
        Sobel(cur_img, gradY, CV_32F, 0, 1, 3);
        // Compute the magnitude of the gradient
        Mat magnitude;
        magnitude = abs(gradX) + abs(gradY);
        this->img_stacks_sobels_.push_back(magnitude);
    }  
}


float ImgProcessor::CalSobelScoreByPoint(const Point3i& point)
{
    // auto cur_img = img_stacks_[point.z];
    // std::cout << "img stack grey num : " << img_stacks_grey_.size() << std::endl;
    auto cur_img = img_stacks_grey_[point.z];
    float g_x = 0;
    float g_y = 0;
    std::vector<std::vector<float>>  gradient_x = {{-1,  0,  1}, {-2, 0, 2}, {-1, 0, 1}};
    std::vector<std::vector<float>>  gradient_y = {{-1, -2, -1}, {0, 0, 0},  {1, 2, 1}};

    // std::vector<std::vector<float>>  gradient_x = {{-3,  0,  3}, {-10, 0, 10}, {-3, 0, 3}};
    // std::vector<std::vector<float>>  gradient_y = {{-3, -10, -3}, {0, 0, 0},  {3, 10, 3}};

    
    // double minVal; 
    // double maxVal;
    // cv::minMaxLoc(cur_img, &minVal, &maxVal);
    // cv::Mat new_img = (cur_img - minVal) / (maxVal - minVal) * 255;
    // new_img.convertTo(new_img, CV_8U);
    for(int m_r = 0; m_r < 3; ++m_r)
    {
        for(int n_r = 0; n_r < 3; ++n_r)
        {
            float param_x = gradient_x[m_r][n_r]; 
            float param_y = gradient_y[m_r][n_r];
            int r_i = point.y - m_r + 1;
            int c_j = point.x - n_r + 1;
            r_i = std::min(std::max(r_i, 0), cur_img.rows-1);
            c_j = std::min(std::max(c_j, 0), cur_img.cols-1);
            // g_x += param_x * cur_img.at<double>(r_i, c_j);  
            // g_y += param_y * cur_img.at<double>(r_i, c_j);  
            g_x += param_x * cur_img.at<uchar>(r_i, c_j);  
            g_y += param_y * cur_img.at<uchar>(r_i, c_j);
        }
    }
    float score = sqrt(g_x * g_x + g_y * g_y);
    // std::cout << "score : " << score << std::endl;
    return score;
}

void ImgProcessor::CalContourPointsSobelScore()
{
    this->contour_points_sobel_scores_.clear();
    // std::cout << "!!! contour_sample_points_ size " << this->contour_sample_points_.size() << std::endl;
    for(const auto& p : this->contour_sample_points_)
    {
        float score = CalSobelScoreByPoint(p);
        this->contour_points_sobel_scores_.push_back(score);
    }
    // std::cout << "!!! score number " << this->contour_points_sobel_scores_.size() << std::endl;
}

void ImgProcessor::GetContourPointSobelScores()
{
    this->contour_points_sobel_scores_.clear();
    float min_score = 0; 
    float max_score = 0;
    for(size_t p_id = 0; p_id < this->contour_sample_points_.size(); ++p_id)
    {
        const auto& cur_p = contour_sample_points_[p_id];
        auto& cur_img = this->img_stacks_sobels_[cur_p.z];

        float score = cur_img.at<float>(cur_p.y, cur_p.x);
        this->contour_points_sobel_scores_.push_back(score);
    }    
}

void ImgProcessor::SaveGradientAndSobelScorePair(const std::string& out_csv_path)
{
    std::fstream s_file{ out_csv_path, s_file.out };
    for(size_t i = 0; i < this->contour_points_sobel_scores_.size(); ++ i)
    {
        float score = this->contour_points_sobel_scores_[i];
        auto gradient = this->contour_sample_points_gradients_[i];
        float gradient_len = sqrt(gradient.dot(gradient));
        std::string p_str = std::to_string(score) + "," + std::to_string(gradient_len);
        s_file <<  p_str << std::endl;
    }
}



void ImgProcessor::RunSinglePipeline(const std::string& img_stack_path, const std::string& point_path, const std::string& out_gradient_path)
{
    LoadImgStacks(img_stack_path);
    // test_sobel_score();
    FilterImgStacksWithGaussian();
    LoadImgContourSamplePoints(point_path);
    CalSamplePointsGradients();
    CalSamplePointsTangentVector();
    CalSamplePointsGradientsTangentProjection();
    // std::string gradient_path = out_subdir + "/gradients.obj";
    SaveImgContourSamplePointsWithGradients(out_gradient_path);

    
    // std::string out_point_path = out_subdir + "/contour_points.obj";
    // SaveImgContourSamplePoints(out_point_path);

}

void ImgProcessor::RunBatchesPipeline(const std::string& img_stack_dir, const std::string& point_dir, const std::string& out_dir)
{

    const std::string gradient_dir = out_dir + "/gradient";
    if(!fs::exists(gradient_dir))
    {
        fs::create_directory(gradient_dir);
    }
    const std::string out_point_dir = out_dir + "/point";
    if(!fs::exists(out_point_dir))
    {
        fs::create_directory(out_point_dir);
    }

    const std::string img_dir = out_dir + "/img_contours";
    if(!fs::exists(img_dir))
    {
        fs::create_directory(img_dir);
    }

    const std::string sobel_dir = out_dir + "/img_sobels";
    if(!fs::exists(sobel_dir))
    {
        fs::create_directory(sobel_dir);
    }

    const std::string csv_dir = out_dir + "/csv_sobels";
    if(!fs::exists(csv_dir))
    {
        fs::create_directory(csv_dir);
    }

    for(const auto & entry : fs::directory_iterator(img_stack_dir))
    {
        max_gradient_norm_ = 0.0;
        std::string file = fs::path(entry).filename();
        // std::cout << file << std::endl;
        auto found = file.find('_');
        if (found!=std::string::npos)
        {
            std::string file_name = file.substr(0, found);
            std::cout << file_name << std::endl; 
            std::string point_path = point_dir + "/" + file_name + "/contour_points/opt_cv_contour_5/opt_cv_contour_4.xyz";
            std::string out_gradient_path = gradient_dir + "/" + file_name + "_gradient.obj";
            std::string out_point_path = out_point_dir + "/" + file_name + "_points.obj";
            if(fs::exists(point_path))
            {
                RunSinglePipeline(fs::path(entry), point_path, out_gradient_path);
                SaveImgContourSamplePoints(out_point_path);
            }
            std::string contour_img_dir = img_dir + "/" + file_name + "_imgs";
            std::cout << "save contour dir 00: " << contour_img_dir << std::endl;
            if( !fs::exists(contour_img_dir))
            {
                fs::create_directory(contour_img_dir);
            }
            std::cout << "save contour dir : " << contour_img_dir << std::endl;
            std::cout << "CalContourPointsSobelScore start " << std::endl;
            // CalContourPointsSobelScore();
            // std::cout << "CalContourPointsSobelScore succeed " << std::endl;

            // std::cout << "save contour image dir : " << contour_img_dir << std::endl;
           
            CalSobelScore();

            SaveOriginalImgsWithContourPoints(contour_img_dir + "/" + file_name);
            std::string sobel_img_dir = sobel_dir + "/" + file_name + "_imgs";
            std::cout << "save sobel_img_dir dir 00: " << sobel_img_dir << std::endl;
            if( !fs::exists(sobel_img_dir))
            {
                fs::create_directory(sobel_img_dir);
            }
            SaveSobelImgsWithContourPoints(sobel_img_dir + "/" + file_name);

            CalSamplePointsGradients();
            GetContourPointSobelScores();
            std::string out_path = csv_dir + "/" + file_name + ".csv";
            SaveGradientAndSobelScorePair(out_path);
          
        }
        // break;
    }
}

void ImgProcessor::RunGradientsPipeline(const std::string& img_stack_path, const std::string& img_mask_stack_path, const std::string& out_path)
{
    LoadImgStacks(img_stack_path);
    FilterImgStacksWithGaussian();
    // LoadImgStacksMask(img_mask_stack_path);
    std::cout << " Load Img stacks success !" << std::endl;
    this->contour_mask_overlay_output_path_ = "../data/contour_mask_overlay/"; 
    // ScanImageEdgeDetection();
    std::string point_path = "/home/jjxia/Documents/projects/SparseRecon/IMG/6007/contour_points/opt_cv_contour_5/opt_cv_contour_4.xyz";
    LoadImgContourSamplePoints(point_path);
    CalSamplePointsGradients();
    CalSamplePointsTangentVector();
    CalSamplePointsGradientsTangentProjection();
    std::string gradient_path = "../data/gradients4_0.obj";
    SaveImgContourSamplePointsWithGradients(gradient_path);

    std::string out_point_path = "../data/contour_point4_0.obj";
    SaveImgContourSamplePoints(out_point_path);
    // CalSamplePointsStructureTensor();
    // std::string original_path = "../data/original_data/";
    // SaveOriginalImgs(original_path);
    //     CalVolumeGradients();
    //     std::cout << " CalVolumeGradients success !" << std::endl;
    //     SampleVolumeGradients(sample_step_);
    //      std::cout << " SampleVolumeGradients success !" << std::endl;
    //     SaveGradients(out_path);
}


