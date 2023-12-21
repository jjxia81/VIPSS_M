
#include <iostream>
#include "src/ImgProcessor.hpp"

// void LoadImgStacks(const std::string& img_stack_path)
// {
// 	std::vector<cv::Mat> img_stacks;
// 	cv::imreadmulti(img_stack_path, img_stacks, cv::IMREAD_ANYDEPTH);
// 	for (const auto& img : img_stacks)
// 	{
// 		double min,max;
// 		cv::minMaxLoc(img,&min,&max);
// 		auto new_img = img / max;
// 		std::cout << "min max : " << min << " " << max << std::endl;
// 		cv::imshow("first img", new_img);
// 		cv::waitKey(0);
// 		break;
// 	}
// 	//ret, images = cv2.imreadmulti(self.img_stack_path_, images, cv2.IMREAD_ANYDEPTH)
// }

int main()
{
    const std::string img_stack_path = "/home/jjxia/Documents/projects/All_Elbow/scans_tiffstacks/6007_scan_midsaggital.tif";
    const std::string img_mask_stack_path = "/home/jjxia/Documents/projects/All_Elbow/segmentations_tiffstack/6007_segmentation_midsaggital.tif";
    const std::string out_path = "../data/6007_gradient_scale.obj";
    ImgProcessor ImgPro;
	// ImgPro.LoadImgStacks(img_stack_path);
    // ImgPro.RunGradientsPipeline(img_stack_path, img_mask_stack_path, out_path);
    const std::string img_stack_dir = "/home/jjxia/Documents/projects/All_Elbow/scans_tiffstacks";
    const std::string point_dir = "/home/jjxia/Documents/projects/All_Elbow/contours";
    const std::string out_dir = "/home/jjxia/Documents/projects/All_Elbow/out";
    ImgPro.RunBatchesPipeline(img_stack_dir, point_dir, out_dir);
    // std::cout << " hello " << std::endl;
}