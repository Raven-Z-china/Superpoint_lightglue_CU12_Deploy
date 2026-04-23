#include <iostream>
#include <memory>
#include "super_point.h"
#include "light_glue.h"

int main() {
    cv::Mat image0 = cv::imread("image/image0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread("image/image1.png", cv::IMREAD_GRAYSCALE);

    if (image0.empty() || image1.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return -1;
    }

    Configs configs("config/config.yaml", "weights/");
    
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    // auto superpoint_lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);

    std::cout << "Building SuperPoint engine..." << std::endl;
    if (!superpoint->build()) {
        std::cerr << "SuperPoint build failed" << std::endl;
        return -1;
    }

    // std::cout << "Building LightGlue engine..." << std::endl;
    // if (!superpoint_lightglue->build()) {
    //     std::cerr << "LightGlue build failed" << std::endl;
    //     return -1;
    // }

    Eigen::Matrix<double, 258, Eigen::Dynamic> feature0, feature1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> score0, score1;
    
    std::cout << "Inferring image 0..." << std::endl;
    superpoint->infer(image0, feature0, score0);
    
    std::cout << "Inferring image 1..." << std::endl;
    superpoint->infer(image1, feature1, score1);

    std::cout << "Image 0 keypoints: " << feature0.cols() << std::endl;
    std::cout << "Image 1 keypoints: " << feature1.cols() << std::endl;

    // std::vector<cv::DMatch> matches;
    // superpoint_lightglue->matching_points(feature0, feature1, matches);
    // std::cout << "Matches: " << matches.size() << std::endl;

    return 0;
}
