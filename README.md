# SuperPoint LightGlue TensorRT

SuperPoint and LightGlue with TensorRT. Deploy with C++. (Release tag v0.1.0 is available, the main branch has some bugs to fix.)

## Demo

- [x] TODO

## Baseline

|           Image Size: 320 x 240           | RTX3080 |
|:-----------------------------------------:|:-------:|
|  SuperPoint (250 points)<sub>Time</sub>   | 0.95 MS |             
|  SuperPoint (256 points)<sub>Time</sub>   | 0.96 MS |             
|    SuperGlue (256 dims)<sub>Time</sub>    | 7.47 MS |             
|    LightGlue (256 dims)<sub>Time</sub>    | 2.54 MS |
| (SuperPoint Once+SuperGlue)<sub>FPS</sub> | 118 FPS |             
| (SuperPoint Once+SuperGlue)<sub>FPS</sub> | 286 FPS |

## Docker(Recommand)

```bash
docker pull yuefan2022/tensorrt-ubuntu22.04-cuda12.1:latest
docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --privileged --runtime nvidia --gpus all --volume ${PWD}:/workspace --workdir /workspace --name tensorrt yuefan2022/tensorrt-ubuntu22.04-cuda12.1:latest /bin/bash
```

## Environment Required

* CUDA==12.1
* TensorRT==8.6.1.6
* OpenCV>=4.0
* EIGEN
* yaml-cpp

## Convert Model(Optional)

The converted model is already provided in the [weights](./weights) folder, if you are using the pretrained model
officially provided by [SuperPoint](https://github.com/magicleap/SuperGluePretrainedNetwork) and [LightGlue](https://github.com/cvg/LightGlue), you do not
need to go through this step.

```bash
python3 convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python3 convert2onnx/convert_lightglue_to_onnx.py --weight_file lightglue_pth_file_path --output_dir lightglue_onnx_file_dir
# use polygraphy check onnx model(optional)
polygraphy run --trt --onnxrt onnx_file_name.onnx
```

## Build and Run

```bash
git clone https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT.git
cd SuperPoint-LightGlue-TensorRT
mkdir build
cd build
cmake ..
make
export CUDA_MODULE_LOADING=LAZY
# test on image pairs 100 times, the output image will be saved in the build dir
./build/superpoint_lightglue_image config/config.yaml weights/ image/image0.png image/image1.png
# test on the folder with image sequence, output images will be saved in the param assigned dir
./build/superpoint_lightglue_sequence config/config.yaml weights/ image/freiburg_sequence/ image/freiburg_sequence/match_images/
```

## Samples
```c++
#include "super_point.h"
#include "light_glue.h"

// read image
cv::Mat image0 = cv::imread("../image/image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image1 = cv::imread("../image/image1.png", cv::IMREAD_GRAYSCALE);

// read config from file
Configs configs("../config/config.yaml", "../weights/");

// create superpoint detector and superpoint-lightglue matcher
auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
auto superpoint_lightglue = std::make_shared<SuperPointLightGlue>(configs.superpoint_lightglue_config);

// build engine
superpoint->build();
superpoint_lightglue->build();

// infer superpoint
Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
superpoint->infer(image0, feature_points0, feature_scores0);
superpoint->infer(image1, feature_points1, feature_scores1)

// infer lightglue
std::vector<cv::DMatch> lightglue_matches;
superpoint_lightglue->matching_points(feature_points0, feature_points1, lightglue_matches);
 
```

## Acknowledgements
[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) [LightGlue](https://github.com/cvg/LightGlue) [TensorRT](https://github.com/NVIDIA/TensorRT)


FP32推理结果
首图特征点数/连续帧特征点数|匹配点数|单帧sp提取时间|lg匹配时间
1 | 418 / 444 | 336 | 4.584 ms | 5.445 ms
2 | 418 / 416 | 291 | 3.845 ms | 5.652 ms
3 | 418 / 393 | 229 | 3.715 ms | 6.395 ms
4 | 418 / 370 | 187 | 3.685 ms | 5.274 ms
5 | 418 / 432 | 155 | 3.958 ms | 5.333 ms
6 | 418 / 449 | 129 | 3.917 ms | 5.762 ms
7 | 418 / 419 | 97 | 3.627 ms | 6.224 ms
8 | 418 / 474 | 74 | 3.714 ms | 5.267 ms
9 | 418 / 427 | 52 | 3.745 ms | 5.318 ms
10 | 418 / 415 | 43 | 3.728 ms | 6.342 ms
11 | 418 / 458 | 35 | 3.718 ms | 5.205 ms
12 | 418 / 474 | 23 | 3.897 ms | 5.438 ms
13 | 418 / 426 | 34 | 3.798 ms | 6.406 ms
14 | 418 / 457 | 23 | 3.828 ms | 5.207 ms
Warning: Keypoints exceed engine capacity, clamping to 500
15 | 418 / 505 | 27 | 3.974 ms | 5.402 ms
16 | 418 / 486 | 30 | 3.781 ms | 6.35 ms

FP16推理结果
首图特征点数/连续帧特征点数|匹配点数|单帧sp提取时间|lg匹配时间
1 | 422 / 439 | 335 | 3.53 ms | 4.679 ms
2 | 422 / 413 | 292 | 2.681 ms | 5.402 ms
3 | 422 / 392 | 231 | 2.572 ms | 4.636 ms
4 | 422 / 367 | 188 | 2.747 ms | 4.769 ms
5 | 422 / 430 | 157 | 2.676 ms | 5.17 ms
6 | 422 / 445 | 133 | 2.649 ms | 4.651 ms
7 | 422 / 418 | 94 | 2.732 ms | 4.741 ms
8 | 422 / 472 | 74 | 2.673 ms | 4.734 ms
9 | 422 / 427 | 50 | 2.529 ms | 4.654 ms
10 | 422 / 416 | 42 | 2.62 ms | 4.623 ms
11 | 422 / 457 | 35 | 2.637 ms | 4.621 ms
12 | 422 / 474 | 24 | 2.631 ms | 5.491 ms
13 | 422 / 424 | 37 | 2.674 ms | 4.618 ms
14 | 422 / 461 | 26 | 2.762 ms | 4.944 ms
Warning: Keypoints exceed engine capacity, clamping to 500
15 | 422 / 503 | 30 | 2.631 ms | 4.641 ms
16 | 422 / 487 | 28 | 2.863 ms | 4.77 ms