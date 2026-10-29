"""Two-sign LoRA; the legacy class name does not imply int8 quantization."""
import math

import torch
from torch import nn


class LoRAInt8Linear(nn.Module):
    def __init__(self, in_features=4, out_features=4, r=2, lora_alpha=2):
        super().__init__()
        if min(in_features, out_features, r) <= 0:
            raise ValueError("LoRA dimensions and rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r
        self.register_buffer("lora_A", torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x, lora_B, noise, eps):
        bsz, seq_len, _ = x.shape
        if bsz % 2:
            raise ValueError("Two-sign LoRA requires an even physical batch")
        pair = torch.stack((lora_B + eps * noise, lora_B - eps * noise))
        intermediate = x @ self.lora_A.T
        result = torch.matmul(
            intermediate.reshape(2, bsz // 2, seq_len, self.r),
            pair.reshape(2, 1, self.r, self.out_features),
        )
        return result.reshape(bsz, seq_len, self.out_features) * self.scaling


class MobileLoRALinear(nn.Module):
    def __init__(self, base, r, alpha, offset):
        super().__init__()
        self.base = base
        self.adapter = LoRAInt8Linear(base.in_features, base.out_features, r, alpha)
        self.offset = offset
        self.numel = r * base.out_features

    def forward(self, x, state, noise, eps):
        end = self.offset + self.numel
        shape = (self.adapter.r, self.adapter.out_features)
        return self.base(x) + self.adapter(
            x, state[self.offset:end].view(shape),
            noise[self.offset:end].view(shape), eps,
        )


def project(layer, x, state, noise, eps):
    if isinstance(layer, MobileLoRALinear):
        return layer(x, state, noise, eps)
    return layer(x)
