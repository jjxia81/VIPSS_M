#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


typedef cv::Point3f Point3;
typedef cv::Vec3f   Vec3;

class ImgProcessor{

    public:
        ImgProcessor(){};
        ~ImgProcessor(){};
        
        void LoadImgStacks(const std::string& img_stack_path);
        void CalVolumeGradients();
        void SampleVolumeGradients(size_t sample_step);
        Vec3 GetGradient(size_t x_id, size_t y_id, size_t z_id);
        Vec3 GetGradient(const Point3& point);
        void SaveGradients(const std::string& path);
        void RunGradientsPipeline(const std::string& img_stack_path, const std::string& out_path);

    public:
        size_t volume_x_max_;
        size_t volume_y_max_;
        size_t volume_z_max_;

    private:

        std::vector<cv::Mat> img_stacks_;
        std::vector<cv::Mat> volume_gradient_z_;
        std::vector<cv::Mat> volume_gradient_x_;
        std::vector<cv::Mat> volume_gradient_y_;
        std::vector<Point3>  volume_points_;
        std::vector<Vec3>    volume_gradients_;
        float max_gradient_norm_;
        float sample_step_ = 4.0;
};


