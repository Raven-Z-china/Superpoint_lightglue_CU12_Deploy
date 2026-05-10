#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
import cv2

import custom_superpoint
from utils import to_numpy


def main():
    parser = argparse.ArgumentParser(
        description='script to convert model from pytorch to onnx')
    parser.add_argument('--weight', default=None,
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="../weights/", help="onnx model file output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight

    superpoint_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'weights': weight_file,
    }

    state_dict = torch.load(weight_file)['model']

    # load model
    superpoint_model = custom_superpoint.SuperPoint(superpoint_config)

    superpoint_model.load_state_dict(state_dict,strict=True)
    superpoint_model.eval()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # create random input to the model for onnx trace
    superpoint_input = torch.randn(1, 1, 240, 320)

    torch_infer_output = superpoint_model(superpoint_input)
    if weight_file is not None:
        onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + "_sp.onnx")
    else:
        onnx_filename = os.path.join(output_dir, "offical_sp.onnx")

    # export the model - use dynamo=False to avoid PyTorch 2.x ONNX export bug
    torch.onnx.export(
        superpoint_model,  # model being run
        superpoint_input,  # model input (or a tuple for multiple inputs)
        onnx_filename,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['image'],  # the model input names
        output_names=['scores', 'descriptors'],  # the model output names
        dynamic_axes={'image': {2: 'image_height', 3: "image_width"}},  # dynamic input names
    )

    print(f"save path:{onnx_filename}")

    # check onnx conversion
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_input = {onnxruntime_session.get_inputs()[0].name: to_numpy(superpoint_input)}
    onnxruntime_infer_output = onnxruntime_session.run(None, onnxruntime_input)

    # compare ONNX Runtime and PyTorch results (relaxed tolerance)
    np.testing.assert_allclose(to_numpy(torch_infer_output[0]), onnxruntime_infer_output[0], rtol=1e-02, atol=1e-03)
    np.testing.assert_allclose(to_numpy(torch_infer_output[1]), onnxruntime_infer_output[1], rtol=1e-02, atol=1e-03)

    print("Exported superpoint has been tested with ONNXRuntime, and the result looks good.")

    # convert lightglue

    lightglue_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (auto-selected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": weight_file,
    }

    lightglue_model = custom_lightglue.LightGlue("superpoint", depth_confidence=lightglue_conf["depth_confidence"],
                                          width_confidence=lightglue_conf["width_confidence"],
                                          weights=lightglue_conf["weights"])

    lightglue_model.load_state_dict(state_dict,strict=True)
    lightglue_model.eval()

    pytorch_total_params = sum(p.numel() for p in lightglue_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # Create input to the model for onnx trace.
    x0 = torch.from_numpy(np.random.randint(low=0, high=320, size=(1, 512)))
    y0 = torch.from_numpy(np.random.randint(low=0, high=240, size=(1, 512)))
    keypoints0 = torch.stack((x0, y0), 2).float()
    keypoints0 = normalize_keypoints(keypoints0, (1, 1, 240, 320))
    descriptors0 = torch.randn(1, 512, 256)
    x1 = torch.from_numpy(np.random.randint(low=0, high=320, size=(1, 512)))
    y1 = torch.from_numpy(np.random.randint(low=0, high=240, size=(1, 512)))
    keypoints1 = torch.stack((x1, y1), 2).float()
    keypoints1 = normalize_keypoints(keypoints1, (1, 1, 240, 320))
    descriptors1 = torch.randn(1, 512, 256)

    torch_infer_output = lightglue_model(keypoints0, keypoints1, descriptors0, descriptors1)

    if weight_file is not None:
        onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + "_lg.onnx")
    else:
        onnx_filename = os.path.join(output_dir, "offical_lg.onnx")

    torch.onnx.export(lightglue_model,
                      (keypoints0, keypoints1, descriptors0, descriptors1),
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["keypoints0",  # batch x number0 x 2
                                   "keypoints1",  # batch x number1 x 2
                                   "descriptors0",  # batch x number0 x 256
                                   "descriptors1"],  # batch x number1 x 256
                      output_names=["scores"],
                      dynamic_axes={
                          "keypoints0": {1: "feature_number_0"},
                          "keypoints1": {1: "feature_number_1"},
                          "descriptors0": {1: "feature_number_0"},
                          "descriptors1": {1: "feature_number_1"},
                      },
                      )
    

    print(f"save path:{onnx_filename}")

    # Check onnx conversion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(keypoints0),
                          onnxruntime_session.get_inputs()[1].name: to_numpy(keypoints1),
                          onnxruntime_session.get_inputs()[2].name: to_numpy(descriptors0),
                          onnxruntime_session.get_inputs()[3].name: to_numpy(descriptors1)}
    onnxruntime_infer_out = onnxruntime_session.run(None, onnxruntime_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_infer_output), onnxruntime_infer_out[0], rtol=1e-03, atol=1e-05)
    print("Exported lightglue has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
