//
// Created by haoyuefan on 2021/9/22.
//

#include "light_glue.h"

#include <cfloat>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperPointLightGlue::SuperPointLightGlue(const SuperPointLightGlueConfig &superpoint_lightglue_config) : superpoint_lightglue_config_(superpoint_lightglue_config), engine_(nullptr) {
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPointLightGlue::build() {
  if (deserialize_engine()) {
    return true;
  }

  auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }

  const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    return false;
  }

  auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
  if (!parser) {
    return false;
  }

  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    return false;
  }
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 2));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 2));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 256));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 1024, 256));

  config->addOptimizationProfile(profile);

  auto constructed = construct_network(builder, network, config, parser);
  if (!constructed) {
    return false;
  }

  auto profile_stream = makeCudaStream();
  if (!profile_stream) {
    return false;
  }
  config->setProfileStream(*profile_stream);

  TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  TensorRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime) {
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) {
    return false;
  }

  save_engine();

  ASSERT(network->getNbInputs() == 4);
  keypoints_0_dims_ = network->getInput(0)->getDimensions();
  keypoints_1_dims_ = network->getInput(1)->getDimensions();
  descriptors_0_dims_ = network->getInput(2)->getDimensions();
  descriptors_1_dims_ = network->getInput(3)->getDimensions();
  assert(keypoints_0_dims_.d[1] == -1);
  assert(keypoints_1_dims_.d[1] == -1);
  assert(descriptors_0_dims_.d[1] == -1);
  assert(descriptors_1_dims_.d[1] == -1);
  return true;
}

bool SuperPointLightGlue::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                            TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  auto parsed = parser->parseFromFile(superpoint_lightglue_config_.onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }
  //    config->setMaxWorkspaceSize(512);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  enableDLA(builder.get(), config.get(), superpoint_lightglue_config_.dla_core);
  return true;
}

bool SuperPointLightGlue::infer(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1,
                                Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score) {
  if (!context_) {
    context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
      return false;
    }
  }

  assert(engine_->getNbBindings() == 5);

  const int keypoints_0_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[0].c_str());
  const int keypoints_1_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[1].c_str());
  const int descriptors_0_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[2].c_str());
  const int descriptors_1_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[3].c_str());
  //    const int scores_index = engine_->getBindingIndex(
  //            superpoint_lightglue_config_.output_tensor_names[0].c_str());

  context_->setBindingDimensions(keypoints_0_index, nvinfer1::Dims3(1, features0.cols(), 2));
  context_->setBindingDimensions(keypoints_1_index, nvinfer1::Dims3(1, features1.cols(), 2));
  context_->setBindingDimensions(descriptors_0_index, nvinfer1::Dims3(1, features0.cols(), 256));
  context_->setBindingDimensions(descriptors_1_index, nvinfer1::Dims3(1, features1.cols(), 256));
  //    context_->setBindingDimensions(scores_index, nvinfer1::Dims3(1, features0.cols(), features1.cols()));

  keypoints_0_dims_ = context_->getBindingDimensions(keypoints_0_index);
  keypoints_1_dims_ = context_->getBindingDimensions(keypoints_1_index);
  descriptors_0_dims_ = context_->getBindingDimensions(descriptors_0_index);
  descriptors_1_dims_ = context_->getBindingDimensions(descriptors_1_index);
  //    scores_dims_ = context_->getBindingDimensions(scores_index);

  BufferManager buffers(engine_, 0, context_.get());

  ASSERT(superpoint_lightglue_config_.input_tensor_names.size() == 4);
  if (!process_input(buffers, features0, features1)) {
    return false;
  }

  buffers.copyInputToDevice();

  bool status = context_->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers.copyOutputToHost();

  if (!process_output(buffers, matches_index, matches_score)) {
    return false;
  }

  return true;
}

bool SuperPointLightGlue::process_input(const BufferManager &buffers, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1) {
  auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[0]));
  auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[1]));
  auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[2]));
  auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[3]));

  for (int colk0 = 0; colk0 < features0.cols(); ++colk0) {
    for (int rowk0 = 0; rowk0 < 2; ++rowk0) {
      keypoints_0_buffer[colk0 * 2 + rowk0] = features0(rowk0, colk0);
    }
  }

//  *keypoints_0_buffer = features0.data()[0];


  for (int colk1 = 0; colk1 < features1.cols(); ++colk1) {
    for (int rowk1 = 0; rowk1 < 2; ++rowk1) {
      keypoints_1_buffer[colk1 * 2 + rowk1] = features1(rowk1, colk1);
    }
  }

//  *keypoints_1_buffer = features1.data()[0];

  for (int cold0 = 0; cold0 < features0.cols(); ++cold0) {
    for (int rowd0 = 2; rowd0 < features0.rows(); ++rowd0) {
      descriptors_0_buffer[cold0 * 256 + (rowd0 - 2)] = features0(rowd0, cold0);
    }
  }

//  *descriptors_0_buffer = features0.data()[features0.cols() * 2];

  for (int cold1 = 0; cold1 < features1.cols(); ++cold1) {
    for (int rowd1 = 2; rowd1 < features1.rows(); ++rowd1) {
      descriptors_1_buffer[cold1 * 256 + (rowd1 - 2)] = features1(rowd1, cold1);
    }
  }

//  *descriptors_1_buffer = features1.data()[features1.cols() * 2];

  return true;
}

void where_negative_one(const int *flag_data, const int *data, int size, std::vector<int> &indices) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      indices.push_back(data[i]);
    } else {
      indices.push_back(-1);
    }
  }
}

void max_matrix(const float *data, int *indices, float *values, int h, int w, int dim) {
  if (dim == 2) {
    for (int i = 0; i < h - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < w - 1; ++j) {
        if (max_value < data[i * w + j]) {
          max_value = data[i * w + j];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  } else if (dim == 1) {
    for (int i = 0; i < w - 1; ++i) {
      float max_value = -FLT_MAX;
      int max_indices = 0;
      for (int j = 0; j < h - 1; ++j) {
        if (max_value < data[j * w + i]) {
          max_value = data[j * w + i];
          max_indices = j;
        }
      }
      values[i] = max_value;
      indices[i] = max_indices;
    }
  }
}

void equal_gather(const int *indices0, const int *indices1, int *mutual, int size) {
  for (int i = 0; i < size; ++i) {
    if (indices0[indices1[i]] == i) {
      mutual[i] = 1;
    } else {
      mutual[i] = 0;
    }
  }
}

void where_exp(const int *flag_data, float *data, std::vector<double> &mscores0, int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores0.push_back(std::exp(data[i]));
    } else {
      mscores0.push_back(0);
    }
  }
}

void where_gather(const int *flag_data, int *indices, std::vector<double> &mscores0, std::vector<double> &mscores1, int size) {
  for (int i = 0; i < size; ++i) {
    if (flag_data[i] == 1) {
      mscores1.push_back(mscores0[indices[i]]);
    } else {
      mscores1.push_back(0);
    }
  }
}

void and_threshold(const int *mutual0, int *valid0, const std::vector<double> &mscores0, double threhold) {
  for (int i = 0; i < mscores0.size(); ++i) {
    if (mutual0[i] == 1 && mscores0[i] > threhold) {
      valid0[i] = 1;
    } else {
      valid0[i] = 0;
    }
  }
}

void and_gather(const int *mutual1, const int *valid0, const int *indices1, int *valid1, int size) {
  for (int i = 0; i < size; ++i) {
    if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
      valid1[i] = 1;
    } else {
      valid1[i] = 0;
    }
  }
}

void decode(float *scores, int h, int w, std::vector<int> &indices0, std::vector<int> &indices1, std::vector<double> &mscores0, std::vector<double> &mscores1) {
  auto *max_indices0 = new int[h - 1];
  auto *max_indices1 = new int[w - 1];
  auto *max_values0 = new float[h - 1];
  auto *max_values1 = new float[w - 1];
  max_matrix(scores, max_indices0, max_values0, h, w, 2);
  max_matrix(scores, max_indices1, max_values1, h, w, 1);
  auto *mutual0 = new int[h - 1];
  auto *mutual1 = new int[w - 1];
  equal_gather(max_indices1, max_indices0, mutual0, h - 1);
  equal_gather(max_indices0, max_indices1, mutual1, w - 1);
  where_exp(mutual0, max_values0, mscores0, h - 1);
  where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);
  auto *valid0 = new int[h - 1];
  auto *valid1 = new int[w - 1];
  and_threshold(mutual0, valid0, mscores0, 0.2);
  and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
  where_negative_one(valid0, max_indices0, h - 1, indices0);
  where_negative_one(valid1, max_indices1, w - 1, indices1);
  delete[] max_indices0;
  delete[] max_indices1;
  delete[] max_values0;
  delete[] max_values1;
  delete[] mutual0;
  delete[] mutual1;
  delete[] valid0;
  delete[] valid1;
}

void log_sinkhorn_iterations(float *couplings, float *Z, int m, int n, float *log_mu, float *log_nu, int iters) {
  auto *u = new float[m]();
  auto *v = new float[n]();
  for (int k = 0; k < iters; ++k) {
    for (int ki = 0; ki < m; ++ki) {
      float nu_expsum = 0.0;
      for (int kn = 0; kn < n; ++kn) {
        nu_expsum += std::exp(couplings[ki * n + kn] + v[kn]);
      }
      u[ki] = log_mu[ki] - std::log(nu_expsum);
    }
    for (int kj = 0; kj < n; ++kj) {
      float nu_expsum = 0.0;
      for (int km = 0; km < m; ++km) {
        nu_expsum += std::exp(couplings[km * n + kj] + u[km]);
      }
      v[kj] = log_nu[kj] - std::log(nu_expsum);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Z[i * n + j] = couplings[i * n + j] + u[i] + v[j];
    }
  }
  delete[] u;
  delete[] v;
}

void log_optimal_transport(float *scores, float *Z, int m, int n, float alpha = 2.3457, int iters = 100) {
  auto *couplings = new float[(m + 1) * (n + 1)];
  for (int i = 0; i < m + 1; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      if (i == m || j == n) {
        couplings[i * (n + 1) + j] = alpha;
      } else {
        couplings[i * (n + 1) + j] = scores[i * n + j];
      }
    }
  }

  float norm = -std::log(m + n);

  auto *log_mu = new float[m + 1];
  auto *log_nu = new float[n + 1];
  for (int ii = 0; ii < m; ++ii) {
    log_mu[ii] = norm;
  }
  log_mu[m] = std::log(n) + norm;

  for (int jj = 0; jj < n; ++jj) {
    log_nu[jj] = norm;
  }
  log_nu[n] = std::log(m) + norm;

  log_sinkhorn_iterations(couplings, Z, m + 1, n + 1, log_mu, log_nu, iters);
  for (int ii = 0; ii < m + 1; ++ii) {
    for (int jj = 0; jj < n + 1; ++jj) {
      Z[ii * (n + 1) + jj] = Z[ii * (n + 1) + jj] - norm;
    }
  }
  delete[] couplings;
  delete[] log_mu;
  delete[] log_nu;
}

// def filter_matches(scores: torch.Tensor, th: float):
//"""obtain matches from a log assignment matrix [BxMxN]"""
// max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
// max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
// m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]
// indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
// # indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
// mutual0 = indices0 == m1.gather(1, m0)
// # mutual1 = indices1 == m0.gather(1, m1)
// max0_exp = max0.values[:, :, 0].exp()
// zero = max0_exp.new_tensor(0)
// mscores0 = torch.where(mutual0, max0_exp, zero)
// # mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
// valid0 = mscores0 > th
// # valid1 = mutual1 & valid0.gather(1, m1)
// # m0 = torch.where(valid0, m0, -1)
// # m1 = torch.where(valid1, m1, -1)
// # return m0, m1, mscores0, mscores1
//
// m_indices_0 = indices0[valid0]
// m_indices_1 = m0[0][m_indices_0]
//
// matches = torch.stack([m_indices_0, m_indices_1], -1)
// mscores = mscores0[0][m_indices_0]
// return matches, mscores

void filter_matches(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &scores, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score,
                    double threshold = 0.1) {
  std::vector<std::pair<int, double>> row_max;
  row_max.resize(scores.rows());
  for (int row = 0; row < scores.rows(); ++row) {
    double max_value = -FLT_MAX;
    for (int col = 0; col < scores.cols(); ++col) {
      if (scores(row, col) > max_value) {
        row_max[row] = std::make_pair(col, scores(row, col));
        max_value = scores(row, col);
      }
    }
//            Eigen::MatrixXd::Index max_row, max_col;
//            double max_score = scores.block(row, 0, 1, scores.cols()).maxCoeff(&max_row, &max_col);
//            row_max[row] = std::make_pair(max_col, max_score);
  }

  std::vector<std::pair<int, double>> col_max;
  col_max.resize(scores.cols());

  for (int col = 0; col < scores.cols(); ++col) {
    double max_value = -FLT_MAX;
    for (int row = 0; row < scores.rows(); ++row) {
      if (scores(row, col) > max_value) {
        col_max[col] = std::make_pair(row, scores(row, col));
        max_value = scores(row, col);
      }
    }
//            Eigen::MatrixXd::Index max_row, max_col;
//            double max_score = scores.block(0, col, scores.rows(), 1).maxCoeff(&max_row, &max_col);
//            row_max[col] = std::make_pair(max_row, max_score);
  }
  std::vector<int> matches_index0_vec;
  std::vector<int> matches_index1_vec;
  std::vector<double> matches_score_vec;
  for (int row = 0; row < row_max.size(); ++row) {
    if (row == col_max[row_max[row].first].first) {
      double score_exp = std::exp(row_max[row].second);
      if (score_exp > threshold) {
        matches_index0_vec.push_back(row);
        matches_index1_vec.push_back(row_max[row].first);
        matches_score_vec.push_back(score_exp);
      }
    }
  }
  matches_index.resize(matches_index0_vec.size(), 2);
  matches_score.resize(matches_score_vec.size(), 1);
  for (int i = 0; i < matches_index0_vec.size(); ++i) {
    matches_index(i, 0) = matches_index0_vec[i];
    matches_index(i, 1) = matches_index1_vec[i];
    matches_score(i) = matches_score_vec[i];
  }
}

bool SuperPointLightGlue::process_output(const BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score) {
  auto *output_scores = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.output_tensor_names[0]));
  int scores_rows = keypoints_0_dims_.d[1];
  int scores_cols = keypoints_1_dims_.d[1];
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> scores_matrix;
  scores_matrix.resize(scores_rows, scores_cols);
  for (int row = 0; row < scores_rows; ++row) {
    for (int col = 0; col < scores_cols; ++col) {
      scores_matrix(row, col) = output_scores[row * scores_cols + col];
    }
  }
  filter_matches(scores_matrix, matches_index, matches_score);
  return true;
}

void SuperPointLightGlue::save_engine() {
  if (superpoint_lightglue_config_.engine_file.empty()) return;
  if (engine_ != nullptr) {
    nvinfer1::IHostMemory *data = engine_->serialize();
    std::ofstream file(superpoint_lightglue_config_.engine_file, std::ios::binary);
    ;
    if (!file) return;
    file.write(reinterpret_cast<const char *>(data->data()), data->size());
  }
}

bool SuperPointLightGlue::deserialize_engine() {
  std::ifstream file(superpoint_lightglue_config_.engine_file, std::ios::binary);
  if (file.is_open()) {
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    char *model_stream = new char[size];
    file.read(model_stream, size);
    file.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
      delete[] model_stream;
      return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
    if (engine_ == nullptr) {
      delete[] model_stream;
      return false;
    }
    delete[] model_stream;
    return true;
  }
  return false;
}

int SuperPointLightGlue::matching_points(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
                                         bool outlier_rejection) {
  matches.clear();
  auto norm_features0 = normalize_keypoints(features0, superpoint_lightglue_config_.image_width, superpoint_lightglue_config_.image_height);
  auto norm_features1 = normalize_keypoints(features1, superpoint_lightglue_config_.image_width, superpoint_lightglue_config_.image_height);

  Eigen::VectorXi indices0, indices1;
  Eigen::Matrix<int, Eigen::Dynamic, 2> matches_index;
  Eigen::Matrix<double, Eigen::Dynamic, 1> matches_score;
  infer(norm_features0, norm_features1, matches_index, matches_score);

  std::vector<cv::Point> points0, points1;
  std::vector<int> point_indexes;
  for (size_t i = 0; i < matches_index.rows(); i++) {
    points0.emplace_back(features0(0, matches_index(i, 0)), features0(1, matches_index(i, 0)));
    points1.emplace_back(features1(0, matches_index(i, 1)), features1(1, matches_index(i, 1)));
    matches.emplace_back(matches_index(i, 0), matches_index(i, 1), matches_score(i));
  }

  if (outlier_rejection) {
    std::vector<uchar> inliers;
    cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
    int j = 0;
    for (int i = 0; i < matches.size(); i++) {
      if (inliers[i]) {
        matches[j++] = matches[i];
      }
    }
    matches.resize(j);
  }

  return matches.size();
}

Eigen::Matrix<double, 258, Eigen::Dynamic> SuperPointLightGlue::normalize_keypoints(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features, int width, int height) {
  Eigen::Matrix<double, 258, Eigen::Dynamic> norm_features;
  norm_features.resize(258, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(0, col) = (features(0, col) - width / 2) / (std::max(width, height) * 0.5);
    norm_features(1, col) = (features(1, col) - height / 2) / (std::max(width, height) * 0.5);
  }
  return norm_features;
}
