SuperPoint LightGlue TensorRT

[![TensorRT](https://img.shields.io/badge/TensorRT-10.8.0-green)] [![CUDA](https://img.shields.io/badge/CUDA-12.1+-blue)] [![OpenCV](https://img.shields.io/badge/OpenCV-%3E%3D4.0-red)]

SuperPoint and LightGlue inference with TensorRT in C++.
Based on SuperPoint-LightGlue-TensorRT by yuefanhao (https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT)

---

Benchmark

FP32 Inference

#   Keypoints (1st / curr frame)   Matches   SP extract (ms)   LG match (ms)
1   418 / 444                      336       4.584             5.445
2   418 / 416                      291       3.845             5.652
3   418 / 393                      229       3.715             6.395
4   418 / 370                      187       3.685             5.274
5   418 / 432                      155       3.958             5.333
6   418 / 449                      129       3.917             5.762
7   418 / 419                      97        3.627             6.224
8   418 / 474                      74        3.714             5.267
9   418 / 427                      52        3.745             5.318
10  418 / 415                      43        3.728             6.342
11  418 / 458                      35        3.718             5.205
12  418 / 474                      23        3.897             5.438
13  418 / 426                      34        3.798             6.406
14  418 / 457                      23        3.828             5.207
15  418 / 505                      27        3.974             5.402
16  418 / 486                      30        3.781             6.350

FP16 Inference

#   Keypoints (1st / curr frame)   Matches   SP extract (ms)   LG match (ms)
1   422 / 439                      335       3.530             4.679
2   422 / 413                      292       2.681             5.402
3   422 / 392                      231       2.572             4.636
4   422 / 367                      188       2.747             4.769
5   422 / 430                      157       2.676             5.170
6   422 / 445                      133       2.649             4.651
7   422 / 418                      94        2.732             4.741
8   422 / 472                      74        2.673             4.734
9   422 / 427                      50        2.529             4.654
10  422 / 416                      42        2.620             4.623
11  422 / 457                      35        2.637             4.621
12  422 / 474                      24        2.631             5.491
13  422 / 424                      37        2.674             4.618
14  422 / 461                      26        2.762             4.944
15  422 / 503                      30        2.631             4.641
16  422 / 487                      28        2.863             4.770

---

Requirements

- CUDA >= 12.1
- TensorRT == 10.8.0.43
- OpenCV >= 4.0
- Eigen3
- yaml-cpp

---

Model Conversion (Optional)

Pretrained models are already provided in the weights/ folder.
If you use the official models from SuperPoint (https://github.com/magicleap/SuperGluePretrainedNetwork) and LightGlue (https://github.com/cvg/LightGlue), you can skip this step.

# Convert SuperPoint
python3 convert2onnx/convert_superpoint_to_onnx.py \
    --weight_file /path/to/superpoint.pth \
    --output_dir /path/to/superpoint_onnx

# Convert LightGlue
python3 convert2onnx/convert_lightglue_to_onnx.py \
    --weight_file /path/to/lightglue.pth \
    --output_dir /path/to/lightglue_onnx

# (Optional) Validate ONNX with Polygraphy
polygraphy run --trt --onnxrt onnx_file_name.onnx

---

Build & Run

git clone https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT.git
cd SuperPoint-LightGlue-TensorRT
mkdir build && cd build
cmake ..
make
export CUDA_MODULE_LOADING=LAZY

# Test on a pair of images (100 runs, result saved in build/)
./superpoint_lightglue_image ../config/config.yaml ../weights/ ../image/image0.png ../image/image1.png

# Test on a sequence folder
./superpoint_lightglue_sequence ../config/config.yaml ../weights/ ../image/freiburg_sequence/ ../image/freiburg_sequence/match_images/

---

Usage Example

#include "super_point.h"
#include "light_glue.h"

// Load images
cv::Mat image0 = cv::imread("../image/image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image1 = cv::imread("../image/image1.png", cv::IMREAD_GRAYSCALE);

// Load config
Configs configs("../config/config.yaml", "../weights/");

// Create detector and matcher
auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
auto superpoint_lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);

// Build TensorRT engines
superpoint->build();
superpoint_lightglue->build();

// Extract features
Eigen::Matrix<double, 258, Eigen::Dynamic> feat0, feat1;
Eigen::Matrix<double, 1, Eigen::Dynamic> score0, score1;
superpoint->infer(image0, feat0, score0);
superpoint->infer(image1, feat1, score1);

// Match features
std::vector<cv::DMatch> matches;
superpoint_lightglue->matching_points(feat0, feat1, matches);

---

Acknowledgements

- SuperPoint (https://github.com/magicleap/SuperPointPretrainedNetwork)
- LightGlue (https://github.com/cvg/LightGlue)
- TensorRT (https://github.com/NVIDIA/TensorRT)
- SuperPoint-LightGlue-TensorRT by yuefanhao (https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT)