"""Convert a bf16 model in place for Turing.

    tbf16.convert(model)   # once, after loading; then run the model as usual
    tbf16.convert(pipe)    # diffusers pipeline: converts every nn.Module component

Every bf16 nn.Linear becomes a tbf16.Linear holding its weight as fp16 plus one exact
power-of-two scale per output row (same memory as bf16, same values), every bf16
nn.Conv1d/2d/3d (and subclasses) runs on fp16 cuDNN kernels the same way (tbf16/conv.py),
and F.scaled_dot_product_attention is routed to tbf16.sdpa for bf16 inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import sdpa
from .conv import convert_conv, convertible
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


_native_sdpa = F.scaled_dot_product_attention


def _sdpa(query, key, value, *args, **kwargs):  # torch's parameter names: callers pass them as keywords
    q, k, v = query, key, value
    if (q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.is_cuda and q.shape[-1] <= 256
            and q.shape[-2] >= MIN_QUERY_ROWS and not (torch.is_grad_enabled() and q.requires_grad)):
        return sdpa(q, k, v, *args, **kwargs)
    return _native_sdpa(q, k, v, *args, **kwargs)


def install_sdpa():
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa


def uninstall_sdpa():
    F.scaled_dot_product_attention = _native_sdpa


def convert(model, sdpa=True, conv=True):
    """Convert an nn.Module in place, or every nn.Module component of a diffusers pipeline."""
    if not isinstance(model, nn.Module) and isinstance(getattr(model, "components", None), dict):
        for c in model.components.values():
            if isinstance(c, nn.Module):
                convert(c, sdpa=False, conv=conv)
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
            elif conv and convertible(child) and owners[id(child.weight)] == 1:
                convert_conv(child)
    torch.cuda.empty_cache()
    if sdpa:
        install_sdpa()
    return model
