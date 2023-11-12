import torch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def normalize_keypoints(keypoints, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = keypoints.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.5
    return (keypoints - center[:, None, :]) / scaling[:, None, :]
