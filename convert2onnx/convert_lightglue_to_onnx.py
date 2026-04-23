#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
import custom_lightglue
from utils import to_numpy, normalize_keypoints


def main():
    parser = argparse.ArgumentParser(description='script to convert lightglue model from pytorch to onnx')
    parser.add_argument('--weight_file', default=None,
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="../weights/", help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

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
                                          weights=lightglue_conf["weights"]).eval()
    pytorch_total_params = sum(p.numel() for p in lightglue_model.parameters())
    print('Total number of params: ', pytorch_total_params)

    # image0_superpoint_output = torch.load('image0_superpoint_output.pt')
    # image1_superpoint_output = torch.load('image1_superpoint_output.pt')
    #
    # keypoints0 = image0_superpoint_output["keypoints"][0]
    # descriptors0 = torch.transpose(image0_superpoint_output["descriptors"][0], 0, 1)
    # keypoints0 = keypoints0.view(-1, keypoints0.shape[0], keypoints0.shape[1])
    # keypoints0 = normalize_keypoints(keypoints0, (1, 1, 240, 320))
    # descriptors0 = descriptors0.view(-1, descriptors0.shape[0], descriptors0.shape[1])
    #
    # keypoints1 = image1_superpoint_output["keypoints"][0]
    # descriptors1 = torch.transpose(image1_superpoint_output["descriptors"][0], 0, 1)
    # keypoints1 = keypoints1.view(-1, keypoints1.shape[0], keypoints1.shape[1])
    # keypoints1 = normalize_keypoints(keypoints1, (1, 1, 240, 320))
    # descriptors1 = descriptors1.view(-1, descriptors1.shape[0], descriptors1.shape[1])
    #
    # scores = np.squeeze(lightglue_model(keypoints0, keypoints1, descriptors0, descriptors1).detach().numpy())

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
                      dynamo=False, 
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
    print("Exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
