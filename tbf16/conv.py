"""bf16 convolutions on fp16 cuDNN kernels.

cuDNN has no fast bf16 convolution on Turing, so a bf16 VAE runs well below fp16 speed.
Same trick as the linear layers: the weight gets one exact power-of-two scale per output
channel, the input one per sample (a conv mixes positions, so rows cannot be scaled
separately), the conv runs in fp16 with fp32 accumulation, and the scales are undone in
one fused multiply-add that writes bf16.

The weight conversion is lossless. The input conversion is lossless for every element
within 2**14 of its sample's max; smaller ones become fp16 subnormals (absolute error
below 2**-25 of the max), far below the bf16 rounding of the output.

1x1 convolutions are per-pixel linear layers and go through scaled_linear, which also
scales each pixel separately. Depthwise convs are memory bound and get slightly slower
(the scaling passes cost more than fp16 saves), but they are converted too: cuDNN has no
sm_75 engine for some bf16 depthwise convs (e.g. 5x5 on channels-last input, in Sana's
DC-AE), which then fail outright.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import _target_exp, prepare_weight, scaled_linear

_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def _pow2_exponent(amax: torch.Tensor):
    """e with amax = m * 2**e, m in [0.5, 1); zeros give 1."""
    amax = amax.float()
    return torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))[1]


def _scaled_conv_forward(self, x, weight, bias):
    # `weight` is the (removed) bf16 parameter, i.e. None; the fp16 copy lives in weight16.
    if x.dtype != torch.bfloat16 or (torch.is_grad_enabled() and x.requires_grad):
        bias = None if bias is None else bias.to(x.dtype)
        return self._native_conv_forward(x, self.dequantized_weight().to(x.dtype), bias)
    nd = x.dim() - 2
    O = self.out_channels
    if self._tbf16_pointwise:
        return scaled_linear(x.movedim(1, -1), self.weight16.view(O, -1), self.weight_scale, bias).movedim(-1, 1)
    if self.padding_mode != "zeros":
        x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        padding = 0
    else:
        padding = self.padding
    B = x.shape[0]
    amax = torch.linalg.vector_norm(x, float("inf"), dim=tuple(range(1, x.dim())))
    e = _pow2_exponent(amax) - self._tbf16_target_exp
    one = torch.ones_like(amax, dtype=torch.float32)
    bshape = (B,) + (1,) * (nd + 1)
    x16 = torch.empty_like(x, dtype=torch.float16)
    torch.mul(x, torch.ldexp(one, -e).view(bshape), out=x16)  # exact power-of-two scale, fp32 math
    y16 = _CONV[nd](x16, self.weight16, None, self.stride, padding, self.dilation, self.groups)
    scale = (torch.ldexp(one, e)[:, None] * self.weight_scale[None, :]).view((B, O) + (1,) * nd)
    out = torch.empty_like(y16, dtype=torch.bfloat16)
    if bias is None:
        return torch.mul(y16, scale, out=out)
    return torch.addcmul(bias.detach().view((1, O) + (1,) * nd), y16, scale, out=out)


def _dequantized_weight(self):
    w = self.weight16.float() * self.weight_scale.view((-1,) + (1,) * (self.weight16.dim() - 1))
    return w.bfloat16()


_classes = {}


def _scaled_class(cls):
    if cls not in _classes:
        _classes[cls] = type(f"TBF16{cls.__name__}", (cls,), {
            "_native_conv_forward": cls._conv_forward,
            "_conv_forward": _scaled_conv_forward,
            "dequantized_weight": _dequantized_weight,
        })
    return _classes[cls]


def convertible(m: nn.Module) -> bool:
    return (isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and type(m) not in _classes.values()
            and isinstance(m.weight, torch.Tensor) and m.weight.dtype == torch.bfloat16 and m.weight.is_cuda)


def convert_conv(conv: nn.Module):
    """Convert one bf16 nn.Conv{1,2,3}d (or subclass) in place. Subclass forwards keep working:
    only _conv_forward is replaced. `.weight` becomes None; use .dequantized_weight()."""
    w = conv.weight.data
    O = w.shape[0]
    K = w[0].numel()
    t = _target_exp(K)
    w16, ws = prepare_weight(w.reshape(O, K))
    del conv._parameters["weight"]
    conv.register_parameter("weight", None)
    conv.register_buffer("weight16", w16.view_as(w))
    conv.register_buffer("weight_scale", ws)
    conv._tbf16_target_exp = t
    no_pad = conv.padding == "valid" or not isinstance(conv.padding, str) and all(p == 0 for p in conv.padding)
    conv._tbf16_pointwise = (all(k == 1 for k in conv.kernel_size) and all(s == 1 for s in conv.stride) and no_pad
                             and conv.groups == 1)
    conv.__class__ = _scaled_class(type(conv))
    return conv
