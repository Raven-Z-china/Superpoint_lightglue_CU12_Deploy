# SuperPoint LightGlue TensorRT

SuperPoint and LightGlue with TensorRT.

## Demo

<img src="image/superpoint_lightglue_tensorrt.gif" width = "640" height = "240"  alt="match_image" border="10" />

* This demo was tested on the RTX3080 GPU.

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
officially provided by [SuperPoint and SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), you do not
need to go through this step.

```bash
python3 convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python3 convert2onnx/convert_lightglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir lightglue_onnx_file_dir
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
./superpoint_lightglue_image  ${PWD}/../config/config.yaml ${PWD}/../weights/ ${PWD}/../image/image0.png ${PWD}/../image/image1.png
# test on the folder with image sequence, output images will be saved in the param assigned dir
./superpoint_lightglue_sequence  ${PWD}/../config/config.yaml ${PWD}/../weights/ ${PWD}/../image/freiburg_sequence/ ${PWD}/../image/freiburg_sequence/match_images/
```
