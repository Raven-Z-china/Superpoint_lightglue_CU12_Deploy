#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
import cv2

import superpoint
from utils import to_numpy


def main():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="../weights/superpoint_v1.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="../weights/", help="onnx model file output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    superpoint_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'weights': weight_file,
    }

    # load model
    superpoint_model = superpoint.SuperPoint(superpoint_config).eval()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # image0 = cv2.imread("../image/image0.png", cv2.IMREAD_GRAYSCALE)
    # image1 = cv2.imread("../image/image1.png", cv2.IMREAD_GRAYSCALE)
    #
    # image0 = cv2.resize(image0, (320, 240))
    # image1 = cv2.resize(image1, (320, 240))
    #
    # image0 = torch.from_numpy(image0.astype(np.float32).reshape(-1, 1, 240, 320) / 255.0)
    # image1 = torch.from_numpy(image1.astype(np.float32).reshape(-1, 1, 240, 320) / 255.0)
    #
    # image0_superpoint_output = superpoint_model(image0, True)
    # image1_superpoint_output = superpoint_model(image1, True)

    # torch.save(image0_superpoint_output, 'image0_superpoint_output.pt')
    # torch.save(image1_superpoint_output, 'image1_superpoint_output.pt')

    # create random input to the model for onnx trace
    superpoint_input = torch.randn(1, 1, 240, 320)

    torch_infer_output = superpoint_model(superpoint_input)
    onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + ".onnx")

    # export the model
    torch.onnx.export(superpoint_model,  # model being run
                      superpoint_input,  # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['image'],  # the model input names
                      output_names=['scores', 'descriptors'],  # the model output names
                      dynamic_axes={'image': {2: 'image_height', 3: "image_width"}}  # dynamic input names
                      )

    # check onnx conversion
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_input = {onnxruntime_session.get_inputs()[0].name: to_numpy(superpoint_input)}
    onnxruntime_infer_output = onnxruntime_session.run(None, onnxruntime_input)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_infer_output[0]), onnxruntime_infer_output[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_infer_output[1]), onnxruntime_infer_output[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
