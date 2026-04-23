//
// Created by haoyuefan on 2023/11/02.
//

#include <chrono>
#include <memory>

#include "light_glue.h"
#include "super_point.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "./superpoint_lightglue_image config_path model_dir first_image_absolutely_path second_image_absolutely_path" << std::endl;
    return 0;
  }

  std::string config_path = argv[1];
  std::string model_dir = argv[2];
  std::string image0_path = argv[3];
  std::string image1_path = argv[4];

  cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

  if (image0.empty() || image1.empty()) {
    std::cerr << "Input image is empty. Please check the image path." << std::endl;
    return 0;
  }

  Configs configs(config_path, model_dir);
  int width = configs.superpoint_lightglue_config.image_width;
  int height = configs.superpoint_lightglue_config.image_height;

  cv::resize(image0, image0, cv::Size(width, height));
  cv::resize(image1, image1, cv::Size(width, height));

  auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
  if (!superpoint->build()) {
    std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
    return 0;
  }
  auto superpoint_lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);
  if (!superpoint_lightglue->build()) {
    std::cerr << "Error in SuperPoint-LightGlue building engine. Please check your onnx model path." << std::endl;
    return 0;
  }

  Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
  Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
  std::vector<cv::DMatch> lightglue_matches;

  // Run inference once
  if (!superpoint->infer(image0, feature_points0, feature_scores0)) {
    std::cerr << "Failed when extracting features from first image." << std::endl;
    return 0;
  }

  if (!superpoint->infer(image1, feature_points1, feature_scores1)) {
    std::cerr << "Failed when extracting features from second image." << std::endl;
    return 0;
  }

  superpoint_lightglue->matching_points(feature_points0, feature_points1, lightglue_matches);

  // Benchmark: run 100 times for timing
  const int num_runs = 100;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) {
    superpoint->infer(image0, feature_points0, feature_scores0);
    superpoint->infer(image1, feature_points1, feature_scores1);
    superpoint_lightglue->matching_points(feature_points0, feature_points1, lightglue_matches);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  std::cout << "Feature points: " << feature_points0.cols() << " / " << feature_points1.cols() << std::endl;
  std::cout << "Matches: " << lightglue_matches.size() << std::endl;
  std::cout << "Average time: " << duration.count() / static_cast<double>(num_runs) << " ms" << std::endl;

  return 0;
}
