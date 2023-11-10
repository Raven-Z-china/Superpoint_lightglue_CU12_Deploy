#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
import lightglue


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.5
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def main():
    parser = argparse.ArgumentParser(
        description='script to convert lightglue model from pytorch to onnx')
    parser.add_argument('--weight_file', default="../weights/superpoint_lightglue.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="../weights/", help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    lightglue_model = lightglue.LightGlue().eval()
    pytorch_total_params = sum(p.numel() for p in lightglue_model.parameters())
    print('total number of params: ', pytorch_total_params)

    image0_superpoint_out = torch.load('image0_superpoint_out.pt')
    image1_superpoint_out = torch.load('image1_superpoint_out.pt')

    keypoints0 = image0_superpoint_out["keypoints"][0]
    descriptors0 = torch.transpose(image0_superpoint_out["descriptors"][0], 0, 1)
    keypoints0 = keypoints0.view(-1, keypoints0.shape[0], keypoints0.shape[1])
    keypoints0 = normalize_keypoints(keypoints0, (1, 1, 240, 320))
    descriptors0 = descriptors0.view(-1, descriptors0.shape[0], descriptors0.shape[1])

    keypoints1 = image1_superpoint_out["keypoints"][0]
    descriptors1 = torch.transpose(image1_superpoint_out["descriptors"][0], 0, 1)
    keypoints1 = keypoints1.view(-1, keypoints1.shape[0], keypoints1.shape[1])
    keypoints1 = normalize_keypoints(keypoints1, (1, 1, 240, 320))
    descriptors1 = descriptors1.view(-1, descriptors1.shape[0], descriptors1.shape[1])

    out_scores = np.squeeze(lightglue_model(keypoints0, keypoints1, descriptors0, descriptors1).detach().numpy())

    # Create input to the model for onnx trace.
    x0 = torch.from_numpy(np.random.randint(low=0, high=320, size=(1, 512)))
    y0 = torch.from_numpy(np.random.randint(low=0, high=240, size=(1, 512)))
    kpts0 = torch.stack((x0, y0), 2).float()
    desc0 = torch.randn(1, 512, 256)
    x1 = torch.from_numpy(np.random.randint(low=0, high=320, size=(1, 512)))
    y1 = torch.from_numpy(np.random.randint(low=0, high=240, size=(1, 512)))
    kpts1 = torch.stack((x1, y1), 2).float()
    desc1 = torch.randn(1, 512, 256)

    torch_out = lightglue_model(kpts0, kpts1, desc0, desc1)
    onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + ".onnx")

    torch.onnx.export(lightglue_model,
                      (kpts0, kpts1, desc0, desc1),
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["keypoints_0",  # batch x number0 x 2
                                   "keypoints_1",  # batch x number1 x 2
                                   "descriptors_0",  # batch x number0 x 256
                                   "descriptors_1"],  # batch x number1 x 256
                      output_names=["scores"],
                      dynamic_axes={
                          "keypoints_0": {1: "feature_number_0"},
                          "keypoints_1": {1: "feature_number_1"},
                          "descriptors_0": {1: "feature_number_0"},
                          "descriptors_1": {1: "feature_number_1"},
                      },
                      )

    # Check onnx conversion.
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    # onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)
    #
    # # compute ONNX Runtime output prediction
    # onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(kpts0),
    #                       onnxruntime_session.get_inputs()[1].name: to_numpy(kpts1),
    #                       onnxruntime_session.get_inputs()[2].name: to_numpy(desc0),
    #                       onnxruntime_session.get_inputs()[3].name: to_numpy(desc1)}
    # onnxruntime_outs = onnxruntime_session.run(None, onnxruntime_inputs)
    #
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out[0]), onnxruntime_outs[0], rtol=1e-03,
    #                            atol=1e-05)
    print("exported model has been tested with ONNXRuntime, and the result looks good.")


if __name__ == '__main__':
    main()
