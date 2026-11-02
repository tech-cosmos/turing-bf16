"""Convert a bf16 model in place for Turing.

    tbf16.convert(model)               # inference
    tbf16.convert(model, train=True)   # training / fine-tuning (also after PEFT/LoRA wrapping)

Inference: every bf16 nn.Linear becomes a tbf16.Linear holding its weight as fp16 plus
one exact power-of-two scale per output row (same memory as bf16, same values).
Training: every nn.Linear becomes a tbf16.TrainLinear (an nn.Linear subclass, same
parameters) whose forward and backward run on fp16 tensor cores for bf16 inputs.
Both route F.scaled_dot_product_attention to tbf16's scaled attention for bf16 inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import autograd
from .attention import sdpa
from .ops import prepare_weight, scaled_linear

# Attention with fewer query rows (decoding) runs natively; it is as fast and skips the scaling.
MIN_QUERY_ROWS = 16


class Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        w16, ws = prepare_weight(lin.weight.data)
        self.register_buffer("weight16", w16)
        self.register_buffer("weight_scale", ws)
        self.bias = lin.bias

    @property
    def weight(self):  # bf16 view for code that reads .weight; materialized on demand
        return (self.weight16.float() * self.weight_scale[:, None]).bfloat16()

    def forward(self, x):
        if x.dtype != torch.bfloat16 or torch.is_grad_enabled() and x.requires_grad:
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        return scaled_linear(x, self.weight16, self.weight_scale, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class TrainLinear(nn.Linear):
    """nn.Linear whose bf16 forward and backward run on fp16 tensor cores. Swapped in by
    changing __class__, so parameters, hooks and PEFT wrappers are untouched."""

    def forward(self, x):
        w, b = self.weight, self.bias
        if x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
            # bf16 mixed precision (fp32 master weights): do autocast's casts here, differentiably
            x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
            b = None if b is None else b.to(torch.bfloat16)
        if x.dtype == w.dtype == torch.bfloat16 and x.is_cuda and x.shape[-1] % 8 == 0:
            if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
                return autograd.linear(x, w, b)
            return scaled_linear(x, *prepare_weight(w), b)
        return super().forward(x)


_native_sdpa = F.scaled_dot_product_attention


def _sdpa(q, k, v, *args, **kwargs):
    if q.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)  # what autocast would do
    if (q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.is_cuda and q.shape[-1] <= 256
            and q.shape[-2] >= MIN_QUERY_ROWS):
        if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
            return autograd.sdpa(q, k, v, *args, **kwargs)
        return sdpa(q, k, v, *args, **kwargs)
    return _native_sdpa(q, k, v, *args, **kwargs)


def install_sdpa():
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa


def uninstall_sdpa():
    F.scaled_dot_product_attention = _native_sdpa


def convert(model: nn.Module, train: bool = False, sdpa: bool = True):
    if train:
        for mod in model.modules():
            if type(mod) is nn.Linear:
                mod.__class__ = TrainLinear
        if sdpa:
            install_sdpa()
        return model
    # A weight shared with another module (e.g. lm_head tied to the embedding) would be
    # stored twice, so those layers stay native.
    owners = {}
    for mod in model.modules():
        for p in mod.parameters(recurse=False):
            owners[id(p)] = owners.get(id(p), 0) + 1
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if (type(child) is nn.Linear and child.weight.dtype == torch.bfloat16 and child.weight.is_cuda
                    and owners[id(child.weight)] == 1):
                setattr(mod, cname, Linear(child))
                child.weight = None  # free the bf16 copy now, not when convert() returns
    torch.cuda.empty_cache()
    if sdpa:
        install_sdpa()
    return model
