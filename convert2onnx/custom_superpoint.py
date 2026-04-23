import torch
from torch import nn

AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


@AMP_CUSTOM_FWD_F32
def normalize_keypoints(
    kpts: torch.Tensor, size = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)[None]
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts

# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s:int = 8):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2.0, dim=1
    )
    return descriptors

class SuperPoint(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'weights': None,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.extractor = _SuperPoint(
            descriptor_dim = self.config['descriptor_dim'],
            nms_radius = self.config['nms_radius'],
        )



    '''
    灰度图转换
    scale = torch.tensor([0.299, 0.587, 0.114], device=image1.device, dtype=image1.dtype).view(1, 3, 1, 1)
    image1 = (image1 * scale).sum(1, keepdim=True)
    '''
    def forward(self, image):
        # image.shape = (1,1,H,W)
        scores, descriptors = self.extractor(image)
        return scores, descriptors

class _SuperPoint(nn.Module):

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"  # noqa: E501

    def __init__(self, descriptor_dim = 256,
                nms_radius = 3,
              ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, descriptor_dim, kernel_size=1, stride=1, padding=0
        )

        # ONNX-compatible MaxPool2d modules with fixed kernel size
        kernel_size = nms_radius * 2 + 1
        self.nms_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=nms_radius, dilation=1)

        self.load_state_dict(
            torch.hub.load_state_dict_from_url(str(self.checkpoint_url)), strict=True
        )

    def forward(self, image):

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, c, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        dense_desc = self.convDb(cDa)
        dense_desc = torch.nn.functional.normalize(dense_desc, p=2.0, dim=1)

        scores = scores.reshape(1,1,h * 8,w * 8)
        
        zeros = torch.zeros_like(scores)
        max_pooled = self.nms_pool(scores)
        max_mask = (scores == max_pooled)
        # NMS iteration 1
        supp_mask = self.nms_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == self.nms_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
        # NMS iteration 2
        supp_mask = self.nms_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == self.nms_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
        scores = torch.where(max_mask, scores, zeros)
        scores = scores.reshape(1,h * 8,w * 8)

        return scores, dense_desc


    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
    