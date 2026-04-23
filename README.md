# SuperPoint LightGlue TensorRT

<div align="center">

[![TensorRT](https://img.shields.io/badge/TensorRT-10.8.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![C++](https://img.shields.io/badge/C++-14-blue?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)

*A high-performance implementation of SuperPoint and LightGlue for feature point matching using NVIDIA TensorRT.*

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Benchmark](#-benchmark)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Model Conversion](#-model-conversion-optional)
- [Usage Example](#-usage-example)
- [Project Structure](#-project-structure)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- 🚀 **TensorRT Optimized**: FP32/FP16 inference with CUDA acceleration
- 📦 **Easy Integration**: Simple C++ API for embedding into your project
- 🔄 **Dynamic Shapes**: Support for variable image sizes
- 📊 **Detailed Benchmarks**: Performance metrics for real-world scenarios

---

## 📊 Benchmark

### FP32 Inference

| # | Keypoints (1st / curr) | Matches | SP Extract (ms) | LG Match (ms) |
|---|-------------------------|---------|-----------------|---------------|
| 1  | 418 / 444  | 336 | 4.58 | 5.45 |
| 2  | 418 / 416  | 291 | 3.85 | 5.65 |
| 3  | 418 / 393  | 229 | 3.72 | 6.40 |
| 4  | 418 / 370  | 187 | 3.69 | 5.27 |
| 5  | 418 / 432  | 155 | 3.96 | 5.33 |
| 6  | 418 / 449  | 129 | 3.92 | 5.76 |
| 7  | 418 / 419  | 97  | 3.63 | 6.22 |
| 8  | 418 / 474  | 74  | 3.71 | 5.27 |
| 9  | 418 / 427  | 52  | 3.75 | 5.32 |
| 10 | 418 / 415  | 43  | 3.73 | 6.34 |

### FP16 Inference

| # | Keypoints (1st / curr) | Matches | SP Extract (ms) | LG Match (ms) |
|---|-------------------------|---------|-----------------|---------------|
| 1  | 422 / 439  | 335 | 3.53 | 4.68 |
| 2  | 422 / 413  | 292 | 2.68 | 5.40 |
| 3  | 422 / 392  | 231 | 2.57 | 4.64 |
| 4  | 422 / 367  | 188 | 2.75 | 4.77 |
| 5  | 422 / 430  | 157 | 2.68 | 5.17 |
| 6  | 422 / 445  | 133 | 2.65 | 4.65 |
| 7  | 422 / 418  | 94  | 2.73 | 4.74 |
| 8  | 422 / 472  | 74  | 2.67 | 4.73 |
| 9  | 422 / 427  | 50  | 2.53 | 4.65 |
| 10 | 422 / 416  | 42  | 2.62 | 4.62 |

> **Note**: FP16 provides ~25% speedup over FP32

---

## 📦 Requirements

| Dependency | Version | Description |
|------------|---------|-------------|
| CUDA | >= 12.1 | NVIDIA CUDA Toolkit |
| TensorRT | 10.8.0.43 | NVIDIA TensorRT SDK |
| OpenCV | >= 4.0 | Computer vision library |
| Eigen3 | - | Linear algebra library |
| yaml-cpp | - | YAML configuration parser |

---

## 🚀 Quick Start

### Build

```bash
git clone https://github.com/Raven-Z-china/Superpoint_lightglue_CU12_Deploy.git
cd Superpoint_lightglue_CU12_Deploy
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run

```bash
export CUDA_MODULE_LOADING=LAZY

# Test on image pair
./superpoint_lightglue_image \
    ../config/config.yaml \
    ../weights/ \
    ../image/image0.png \
    ../image/image1.png

# Test on image sequence
./superpoint_lightglue_sequence \
    ../config/config.yaml \
    ../weights/ \
    ../image/freiburg_sequence/ \
    ../image/freiburg_sequence/match_images/
```

---

## 🔄 Model Conversion (Optional)

Pretrained models are already provided in the `weights/` folder. If you want to use custom models:

```bash
# Convert SuperPoint
python3 convert2onnx/convert_superpoint_to_onnx.py \
    --weight_file /path/to/superpoint.pth \
    --output_dir /path/to/output

# Convert LightGlue
python3 convert2onnx/convert_lightglue_to_onnx.py \
    --weight_file /path/to/lightglue.pth \
    --output_dir /path/to/output

# Validate ONNX model (optional)
polygraphy run --trt --onnxrt your_model.onnx
```

---

## 💻 Usage Example

```cpp
#include "super_point.h"
#include "light_glue.h"

int main() {
    // Load images
    cv::Mat image0 = cv::imread("image0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread("image1.png", cv::IMREAD_GRAYSCALE);

    // Load config
    Configs configs("config/config.yaml", "weights/");

    // Create detector and matcher
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    auto lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);

    // Build TensorRT engines
    superpoint->build();
    lightglue->build();

    // Extract features
    Eigen::Matrix<double, 258, Eigen::Dynamic> feat0, feat1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> score0, score1;
    superpoint->infer(image0, feat0, score0);
    superpoint->infer(image1, feat1, score1);

    // Match features
    std::vector<cv::DMatch> matches;
    lightglue->matching_points(feat0, feat1, matches);

    return 0;
}
```

---

## 📁 Project Structure

```
SuperPoint-LightGlue-TensorRT/
├── CMakeLists.txt           # Build configuration
├── config/
│   └── config.yaml          # Model configuration
├── convert2onnx/
│   ├── convert_superpoint_to_onnx.py
│   ├── convert_lightglue_to_onnx.py
│   ├── custom_superpoint.py  # Custom SuperPoint model
│   └── custom_lightglue.py   # Custom LightGlue model
├── include/
│   ├── super_point.h
│   ├── light_glue.h
│   └── utils.h
├── src/
│   ├── super_point.cpp
│   ├── light_glue.cpp
│   └── utils.cpp
├── weights/                  # Pretrained models
│   ├── offical_sp.engine
│   └── offical_lg.engine
├── image/                    # Test images
└── build/                    # Build output
```

---

## 🙏 Acknowledgements

- [SuperPoint](https://github.com/magicleap/SuperGluePretrainedNetwork) - Magic Leap
- [LightGlue](https://github.com/cvg/LightGlue) - CVG Group
- [TensorRT](https://github.com/NVIDIA/TensorRT) - NVIDIA
- [SuperPoint-LightGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT) - yuefanhao

---

<div align="center">

Made with ❤️ for Computer Vision

</div>
