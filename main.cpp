
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
    const std::string img_stack_path = "/media/jjxia/D/data/All_Elbows/All_Elbows_Anterior_Capsule_Only_20230417/scans_tiffstacks/6007_scan_midsaggital.tif";
    const std::string out_path = "../data/6007_gradient_scale.obj";
    ImgProcessor ImgPro;
	// ImgPro.LoadImgStacks(img_stack_path);
    ImgPro.RunGradientsPipeline(img_stack_path, out_path);
    // std::cout << " hello " << std::endl;
}