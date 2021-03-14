#include "frontend/SuperPointExtractor.h"
#include "Frame.h"
#include <opencv2/opencv.hpp>

#include "Settings.h"

using namespace ldso;

bool test_featureExtraction(std::string imagePath){
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.rows * img.cols == 0){
        std::cerr << "Could not read image " << imagePath << std::endl;
        return false;
    }
    int numFeatures = 1000;
    std::vector<shared_ptr<SuperPoint>> features;
    shared_ptr<Frame> frame(new Frame());
    unique_ptr<SuperPointExtractor> spDetector(new SuperPointExtractor());
    spDetector->DetectAndDescribe(numFeatures, img, frame, features);

    cv::Mat imgCopy;
    cv::cvtColor(img, imgCopy, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < features.size(); i++){
        cv::circle(imgCopy, cv::Point(features[i]->uv[0], features[i]->uv[1]), 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Detection result", imgCopy);
    cv::waitKey(0);

    return true;
}

bool test_tensorAllocation(){
    int array[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor tensorArr = torch::from_blob(array, {2, 5}, options);
    int* ptr = tensorArr.data_ptr<int>();

    for (int i = 0; i < 10; i++){
        std::cout << *(ptr++) << " ";
    }
    std::cout << std::endl;
    return true;
}


int main(int argc, char** argv){
    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << " path_to_pretrained_model path_to_sample_image" << std::endl;
        return 1;
    }

    setting_superPointModelPath = std::string(argv[1]);

    test_featureExtraction(std::string(argv[2]));

    test_tensorAllocation();
    return 0;
}
