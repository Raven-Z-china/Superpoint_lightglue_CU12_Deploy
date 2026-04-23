import warnings
from pathlib import Path
from typing import Callable, List, Optional
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.backends.cudnn.deterministic = True


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    xs = x.shape
    x = x.view(xs[0], xs[1], xs[2], xs[3]//2, 2)
    x1, x2 = x[..., 0] , x[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(xs[0], xs[1], xs[2], xs[3])


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0)
        emb = emb.repeat_interleave(2, dim=3).unsqueeze(1)
        return emb


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

    def forward(self, q, k, v) -> torch.Tensor:
        # ONNX-compatible implementation without einsum
        # q, k, v shape: (1, num_heads, seq_len, dim_head)
        s = q.shape[-1] ** -0.5
        b, h, seq_len, d = q.shape
        
        # Reshape for bmm: (b*h, seq_len, d)
        q_flat = q.reshape(b * h, seq_len, d)
        k_flat = k.reshape(b * h, seq_len, d)
        v_flat = v.reshape(b * h, seq_len, d)
        
        # Attention scores: (b*h, seq_len, seq_len)
        sim = torch.bmm(q_flat, k_flat.transpose(1, 2)) * s
        attn = F.softmax(sim, -1)
        
        # Output: (b*h, seq_len, d)
        out = torch.bmm(attn, v_flat)
        
        # Reshape back to (b, h, seq_len, d)
        return out.reshape(b, h, seq_len, d)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.reshape(1, -1, self.num_heads, self.dim_head, 3).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.dim_head = embed_dim // num_heads
        self.scale = self.dim_head**-0.5
        inner_dim = self.dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        b, seq_len, _ = qk0.shape
        h, d = self.heads, self.dim_head
        
        # Reshape: (b, seq, h*d) -> (b, h, seq, d)
        qk0 = qk0.reshape(b, seq_len, h, d).transpose(1, 2)
        qk1 = qk1.reshape(b, seq_len, h, d).transpose(1, 2)
        v0 = v0.reshape(b, seq_len, h, d).transpose(1, 2)
        v1 = v1.reshape(b, seq_len, h, d).transpose(1, 2)
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        
        # ONNX-compatible: bmm instead of einsum
        # sim[b,h,i,j] = sum_d qk0[b,h,i,d] * qk1[b,h,j,d]
        qk0_flat = qk0.reshape(b * h, seq_len, d)
        qk1_flat = qk1.reshape(b * h, seq_len, d)
        sim = torch.bmm(qk0_flat, qk1_flat.transpose(-2, -1)).reshape(b, h, seq_len, seq_len)
        
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        
        # m0 = attn01 @ v1, m1 = attn10 @ v0
        v1_flat = v1.reshape(b * h, seq_len, d)
        v0_flat = v0.reshape(b * h, seq_len, d)
        attn01_flat = attn01.reshape(b * h, seq_len, seq_len)
        attn10_flat = attn10.reshape(b * h, seq_len, seq_len)
        
        m0 = torch.bmm(attn01_flat, v1_flat).reshape(b, h, seq_len, d)
        m1 = torch.bmm(attn10_flat, v0_flat).reshape(b, h, seq_len, d)
        
        m0, m1 = m0.transpose(1, 2).flatten(start_dim=-2), m1.transpose(1, 2).flatten(start_dim=-2)
        m0, m1 = self.to_out(m0), self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads, bias)
        self.cross_attn = CrossBlock(embed_dim, num_heads, bias)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
    ):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        # ONNX-compatible: bmm instead of einsum "bmd,bnd->bmn"
        sim = torch.bmm(mdesc0, mdesc1.transpose(-2, -1))
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": 0.95,  # early stopping, disable with -1
        "width_confidence": 0.99,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, features="superpoint", **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})

        self.matcher = _LightGlue(
            input_dim=conf.input_dim,
            num_heads=conf.num_heads,
            n_layers=conf.n_layers,
            depth_confidence=conf.depth_confidence,
            filter_threshold=conf.filter_threshold,
            weights=conf.weights,
        )

    '''
    灰度图转换
    scale = torch.tensor([0.299, 0.587, 0.114], device=image1.device, dtype=image1.dtype).view(1, 3, 1, 1)
    image1 = (image1 * scale).sum(1, keepdim=True)
    '''
    def forward(self, kp1,kp2,desc1,desc2):
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            scores = self.matcher(kp1,kp2,desc1,desc2)
        # (m0,m1,mscores0,mscores1).shape=(1,2048),scores.shape=(1,2049,2049)
        return scores

class _LightGlue(nn.Module):

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"
    def __init__(self, input_dim=256,
                num_heads=4,
                n_layers=9,
                depth_confidence=0.95,
                filter_threshold=0.1,
                weights=None,
                 ) -> None:
        super().__init__()

        head_dim = input_dim // num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 , head_dim, head_dim
        )

        h, n, d = num_heads, n_layers, input_dim

        self.n_layers = n_layers
        self.depth_confidence = depth_confidence
        self.filter_threshold = filter_threshold

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n)]
        )
        nn.init.zeros_(self.token_confidence[8].token[0].weight)
        nn.init.zeros_(self.token_confidence[8].token[0].bias)

        state_dict = None

        if weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(weights).exists():
                state_dict = torch.load(weights, map_location="cpu")
            else:
                ValueError(f"weights {weights} not found")
        else:
            fname = f"superpoint_lightglue_v0.1_arxiv".replace(".", "-") + ".pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format('v0.1_arxiv', ''),
                file_name=fname,
            )

        if state_dict:
            # rename old state dict entries
            for i in range(self.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.n_layers)]
            ),
        )

    def compile(self, mode="reduce-overhead"):
        if self.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, kpts0, kpts1, desc0, desc1):
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)
        

        # GNN + final_proj + assignment
        scores = torch.zeros((1,m+1,n+1), device=device)
        tem_desc0, tem_desc = torch.zeros((1,m+1,n+1), device=device), torch.zeros((1,m+1,n+1), device=device)
        cnt=1.0
        ind0 = torch.arange(0, m, device=device)[None]
        ind1 = torch.arange(0, n, device=device)[None]
        # We store the index of the layer at which pruning is detected.
        token0, token1 = None, None
        for i in range(self.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

            token0, token1 = self.token_confidence[i](desc0, desc1)
            mask = torch.where(self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n),1.0,0.0)
            scores = scores * ((cnt - mask) / cnt) + (self.log_assignment[i](desc0, desc1)[0] / cnt) * mask
            cnt += mask

        scores = scores * ((cnt-1.0) / cnt) + (self.log_assignment[-1](desc0, desc1)[0] / cnt)

        return scores

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence

    def loss(self, pred, data):
        raise NotImplementedError


__main_model__ = LightGlue
