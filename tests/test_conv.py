"""tbf16 convolutions match an fp64 reference to bf16 accuracy, including where fp16 overflows."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tbf16
from tbf16.conv import convert_conv

torch.manual_seed(0)
CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def ref(conv, x):
    w = conv.dequantized_weight().double()
    b = None if conv.bias is None else conv.bias.double()
    if conv.padding_mode != "zeros":
        x = F.pad(x.double(), conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
        return CONV[x.dim() - 2](x, w, b, conv.stride, 0, conv.dilation, conv.groups)
    return CONV[x.dim() - 2](x.double(), w, b, conv.stride, conv.padding, conv.dilation, conv.groups)


def check(name, conv, shape, scale=1.0, channels_last=False):
    conv = conv.cuda().bfloat16()
    x = (torch.randn(shape, device="cuda") * scale).bfloat16()
    if channels_last:
        x = x.to(memory_format=torch.channels_last if x.dim() == 4 else torch.channels_last_3d)
    native_fp16 = CONV[x.dim() - 2](x.half(), conv.weight.half(), None if conv.bias is None else conv.bias.half(),
                                    conv.stride, conv.padding if conv.padding_mode == "zeros" else 0, conv.dilation,
                                    conv.groups) if conv.padding_mode == "zeros" else None
    tc = convert_conv(copy.deepcopy(conv))
    with torch.no_grad():
        y = tc(x)
    r = ref(tc, x)
    err = ((y.double() - r).norm() / r.norm()).item()
    fp16 = "n/a" if native_fp16 is None else ("NaN/inf" if not torch.isfinite(native_fp16).all() else
                                              f"{((native_fp16.double() - r).norm() / r.norm()).item():.1e}")
    print(f"{name:<44} x{scale:<6g} rel err {err:.2e}   (naive fp16: {fp16})")
    assert y.dtype == torch.bfloat16 and y.shape == r.shape and torch.isfinite(y).all() and err < 3e-3, err


check("conv2d 3x3", nn.Conv2d(64, 96, 3, padding=1), (2, 64, 40, 56))
check("conv2d 3x3 no bias stride 2", nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False), (2, 64, 40, 56))
check("conv2d 3x3 channels_last", nn.Conv2d(64, 96, 3, padding=1), (2, 64, 40, 56), channels_last=True)
check("conv2d 3x3 reflect padding", nn.Conv2d(32, 32, 3, padding=1, padding_mode="reflect"), (1, 32, 33, 47))
check("conv2d depthwise 5x5 channels_last", nn.Conv2d(96, 96, 5, padding=2, groups=96, bias=False), (1, 96, 16, 16),
      channels_last=True)
check("conv2d grouped 1x1", nn.Conv2d(96, 96, 1, groups=6, bias=False), (1, 96, 16, 16))
check("conv2d 1x1 (scaled_linear path)", nn.Conv2d(128, 320, 1), (2, 128, 24, 24))
check("conv2d 1x1 channels_last", nn.Conv2d(128, 320, 1), (2, 128, 24, 24), channels_last=True)
check("conv1d k=4", nn.Conv1d(48, 64, 4, padding=1), (3, 48, 100))
check("conv3d 3x3x3", nn.Conv3d(32, 48, 3, padding=1), (1, 32, 5, 24, 40))
check("conv3d 3x3x3 channels_last_3d", nn.Conv3d(32, 48, 3, padding=1), (1, 32, 5, 24, 40), channels_last=True)
for s in (3e4, 1e6, 1e-8):
    check("conv2d 3x3", nn.Conv2d(64, 96, 3, padding=1), (2, 64, 40, 56), scale=s)
    check("conv3d 3x3x3", nn.Conv3d(32, 48, 3, padding=1), (1, 32, 5, 24, 40), scale=s)


class CausalConv3d(nn.Conv3d):  # like diffusers' WanCausalConv3d: own forward, then super().forward
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._pad = (1, 1, 1, 1, 2, 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        return super().forward(F.pad(x, self._pad))


# convert() reaches nested convs and subclasses; their forward() still runs.
model = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.Sequential(nn.SiLU(), nn.Conv2d(32, 16, 1))).cuda().bfloat16()
x = torch.randn(1, 16, 20, 20, device="cuda").bfloat16()
with torch.no_grad():
    r = model.double()(x.double()); model.bfloat16()
    tbf16.convert(model, sdpa=False)
    y = model(x)
assert type(model[0]).__name__ == "TBF16Conv2d" and type(model[1][1]).__name__ == "TBF16Conv2d"
err = ((y.double() - r).norm() / r.norm()).item()
print(f"{'convert(nn.Sequential)':<44}         rel err {err:.2e}")
assert err < 1e-2

cc = CausalConv3d(8, 16, 3, padding=1).cuda().bfloat16()
x = torch.randn(1, 8, 5, 12, 12, device="cuda").bfloat16()
with torch.no_grad():
    r = F.conv3d(F.pad(x.double(), cc._pad), cc.weight.double(), cc.bias.double())
    holder = nn.Sequential(cc)
    tbf16.convert(holder, sdpa=False)
    y = holder(x)
assert isinstance(holder[0], CausalConv3d) and holder[0].weight is None
err = ((y.double() - r).norm() / r.norm()).item()
print(f"{'subclass with own forward (causal conv3d)':<44}         rel err {err:.2e}")
assert err < 3e-3

# Autograd / non-bf16 inputs fall back to the native conv on the dequantized weight.
x32 = torch.randn(1, 8, 5, 12, 12, device="cuda")
with torch.no_grad():
    y32 = holder(x32)
r32 = F.conv3d(F.pad(x32.double(), cc._pad), cc.dequantized_weight().double(), cc.bias.double())
assert y32.dtype == torch.float32 and ((y32.double() - r32).norm() / r32.norm()).item() < 1e-5


class FakePipe:  # diffusers pipelines expose their modules as .components
    def __init__(self):
        self.components = {"vae": nn.Sequential(nn.Conv2d(8, 8, 3, padding=1)).cuda().bfloat16(),
                           "transformer": nn.Sequential(nn.Linear(8, 8)).cuda().bfloat16(),
                           "scheduler": object(), "text_encoder": None}


pipe = FakePipe()
tbf16.convert(pipe, sdpa=False)
assert type(pipe.components["vae"][0]).__name__ == "TBF16Conv2d"
assert isinstance(pipe.components["transformer"][0], tbf16.Linear)
print("ok")
