//
// Created by haoyuefan on 2023/11/02.
//

#include <chrono>
#include <memory>

#include "light_glue.h"
#include "super_point.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "./superpoint_lightglue_sequence config_path model_dir image_folder_absolutely_path output_folder_path" << std::endl;
    return 0;
  }

  std::string config_path = argv[1];
  std::string model_dir = argv[2];
  std::string image_path = argv[3];
  std::string output_path = argv[4];
  std::vector<std::string> image_names;
  GetFileNames(image_path, image_names);
  Configs configs(config_path, model_dir);
  int width = configs.superpoint_lightglue_config.image_width;
  int height = configs.superpoint_lightglue_config.image_height;

  std::cout << "Building Inference Engine......" << std::endl;
  auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
  if (!superpoint->build()) {
    std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
    return 0;
  }
  auto superpoint_lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);
  if (!superpoint_lightglue->build()) {
    std::cerr << "Error in LightGlue building engine. Please check your onnx model path." << std::endl;
    return 0;
  }
  std::cout << "SuperPoint and LightGlue inference engine build success." << std::endl;

  Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0;
  Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0;
  cv::Mat image0 = cv::imread(image_names[0], cv::IMREAD_GRAYSCALE);
  if (image0.empty()) {
    std::cerr << "First image in the image folder is empty." << std::endl;
    return 0;
  }
  cv::resize(image0, image0, cv::Size(width, height));
  
  if (!superpoint->infer(image0, feature_points0, feature_scores0)) {
    std::cerr << "Failed when extracting features from first image." << std::endl;
    return 0;
  }
  std::vector<cv::DMatch> init_matches;
  superpoint_lightglue->matching_points(feature_points0, feature_points0, init_matches);
  std::string mkdir_cmd = "mkdir -p " + output_path;
  system(mkdir_cmd.c_str());

  for (int index = 1; index < image_names.size(); ++index) {
    Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores1;
    std::vector<cv::DMatch> lightglue_matches;
    cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_GRAYSCALE);
    if (image1.empty()) continue;
    cv::resize(image1, image1, cv::Size(width, height));

    // SuperPoint inference timing
    auto sp_start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image1, feature_points1, feature_scores1)) {
      std::cerr << "Failed when extracting features from second image." << std::endl;
      return 0;
    }
    auto sp_end = std::chrono::high_resolution_clock::now();
    auto sp_duration = std::chrono::duration_cast<std::chrono::microseconds>(sp_end - sp_start);

    // LightGlue matching timing
    auto lg_start = std::chrono::high_resolution_clock::now();
    superpoint_lightglue->matching_points(feature_points0, feature_points1, lightglue_matches);
    auto lg_end = std::chrono::high_resolution_clock::now();
    auto lg_duration = std::chrono::duration_cast<std::chrono::microseconds>(lg_end - lg_start);

    // Output: image_idx, keypoints, matches, superpoint_time_ms, lightglue_time_ms
    std::cout << index << " | " 
              << feature_points0.cols() << " / " << feature_points1.cols() << " | "
              << lightglue_matches.size() << " | "
              << sp_duration.count() / 1000.0 << " ms | "
              << lg_duration.count() / 1000.0 << " ms" << std::endl;

    cv::Mat match_image;
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    for (size_t i = 0; i < feature_points0.cols(); ++i) {
      double score = feature_scores0(0, i);
      double x = feature_points0(0, i);
      double y = feature_points0(1, i);
      keypoints0.emplace_back(x, y, 8, -1, score);
    }
    for (size_t i = 0; i < feature_points1.cols(); ++i) {
      double score = feature_scores1(0, i);
      double x = feature_points1(0, i);
      double y = feature_points1(1, i);
      keypoints1.emplace_back(x, y, 8, -1, score);
    }
    double cost_time = (sp_duration.count() + lg_duration.count()) / 1000.0;
    VisualizeMatching(image0, keypoints0, image1, keypoints1, lightglue_matches, match_image, cost_time);
    cv::imwrite(output_path + "/" + std::to_string(index) + ".png", match_image);
  }

  return 0;
}
