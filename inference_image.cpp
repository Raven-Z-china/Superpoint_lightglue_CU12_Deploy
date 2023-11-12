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
  std::cout << "First image size: " << std::to_string(image0.cols) << "x" << std::to_string(image0.rows) << std::endl;
  std::cout << "Second image size: " << std::to_string(image1.cols) << "x" << std::to_string(image1.rows) << std::endl;

  std::cout << "Building Inference Engine......" << std::endl;
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
  std::cout << "SuperPoint and SuperPoint-LightGlue inference engine build success." << std::endl;

  Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
  Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
  std::vector<cv::DMatch> lightglue_matches;

  long image0_time_count = 0;
  long image1_time_count = 0;
  long match_time_count = 0;
  std::cout << "SuperPoint and LightGlue test in 100 times." << std::endl;
  for (int i = 0; i < 100; ++i) {
    std::cout << "---------------------------------------------------------" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image0, feature_points0, feature_scores0)) {
      std::cerr << "Failed when extracting features from first image." << std::endl;
      return 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      std::cout << "First image feature points number: " << feature_points0.cols() << std::endl;
      image0_time_count += duration.count();
      std::cout << "First image infer cost " << image0_time_count / i / 1000.0 << " MS" << std::endl;
    }
    start = std::chrono::high_resolution_clock::now();
    if (!superpoint->infer(image1, feature_points1, feature_scores1)) {
      std::cerr << "Failed when extracting features from second image." << std::endl;
      return 0;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      std::cout << "Second image feature points number: " << feature_points1.cols() << std::endl;
      image1_time_count += duration.count();
      std::cout << "Second image infer cost " << image1_time_count / i / 1000.0 << " MS" << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    superpoint_lightglue->matching_points(feature_points0, feature_points1, lightglue_matches);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      match_time_count += duration.count();
      std::cout << "Match image cost " << match_time_count / i / 1000.0 << " MS" << std::endl;
    }
  }

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

  //  cv::drawMatches(image0, keypoints0, image1, keypoints1, lightglue_matches, match_image);
  //  cv::imwrite("match_image.png", match_image);
  //  visualize
  //  cv::imshow("match_image", match_image);
  //  cv::waitKey(-1);

  return 0;
}
