#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


typedef cv::Point3f Point3;
typedef cv::Vec3f   Vec3;
typedef cv::Point3i Point3i;
typedef cv::Mat     Mat;

class ImgProcessor{

    public:
        ImgProcessor(){};
        ~ImgProcessor(){};
        
        void LoadImgStacks(const std::string& img_stack_path);
        void LoadImgStacksMask(const std::string& img_stack_mask_path);
        void LoadImgContourSamplePoints(const std::string& img_contour_sample_points_path);
        void SaveImgContourSamplePoints(const std::string& img_contour_sample_points_path);
        void SaveImgContourSamplePointsWithGradients(const std::string& img_contour_sample_points_path);

        void CalSamplePointsGradients();
        void CalSamplePointsTangentVector();
        void CalSamplePointsGradientsTangentProjection();
        void CalSamplePointsStructureTensor();
        void FilterImgStacksWithGaussian();
        void ScanImageEdgeDetection();
        void SaveOriginalImgs(const std::string& path);
        void SaveOriginalImgsWithContourPoints(const std::string& path);
        void SaveSobelImgsWithContourPoints(const std::string& path);
        void GetContourPointSobelScores();
        void CalSobelScore();
        
        void CalContourPointsSobelScore();

        void CalVolumeGradients();
        void SampleVolumeGradients(size_t sample_step);
        Vec3 GetGradient(size_t x_id, size_t y_id, size_t z_id);
        Vec3 GetGradient(const Point3& point);
        void SaveGradients(const std::string& path);
        
        void RunGradientsPipeline(const std::string& img_stack_path, const std::string& img_mask_stack_path, const std::string& out_path);
        void RunSinglePipeline(const std::string& img_stack_path, const std::string& point_path, const std::string& out_path);
        void RunBatchesPipeline(const std::string& img_stack_dir, const std::string& point_dir, const std::string& out_dir);
        void SaveGradientAndSobelScorePair(const std::string& out_csv_path);


    private:
        Mat CalStructureTensor(const Vec3& gradeint);
        Mat CalStructureTensorGaussain(const Point3i& point);
        Vec3 CalPointGradient(const Point3i& point);
        float CalSobelScoreByPoint(const Point3i& point);
        void test_sobel_score();

    public:
        size_t volume_x_max_;
        size_t volume_y_max_;
        size_t volume_z_max_;

    private:

        std::vector<cv::Mat> img_stacks_;
        std::vector<cv::Mat> img_stacks_grey_;
        std::vector<cv::Mat> img_mask_stacks_;
        std::vector<cv::Mat> img_stacks_sobels_;
        std::vector<cv::Mat> volume_gradient_z_;
        std::vector<cv::Mat> volume_gradient_x_;
        std::vector<cv::Mat> volume_gradient_y_;
        std::vector<Point3>  volume_points_;
        std::vector<Vec3>    volume_gradients_;
        std::vector<Point3i> contour_sample_points_;
        std::vector<Vec3> contour_sample_points_tangents_;
        std::vector<Vec3> contour_sample_points_gradients_;
        std::vector<float> contour_sample_points_projection_;
        std::vector<float> contour_points_sobel_scores_;
        std::string contour_mask_overlay_output_path_;

        float max_gradient_norm_;
        float sample_step_ = 4.0;
};


