"""Convolutions on Turing: torch bf16 vs fp16 vs fp32 vs tbf16.Conv, on diffusion VAE / DiT shapes."""
import copy

import torch
import torch.nn as nn
import triton.testing as tt

from tbf16.conv import convert_conv

# (name, conv factory, input shape)
CASES = [
    ("2d 3x3 512ch 128^2 (VAE)", lambda: nn.Conv2d(512, 512, 3, padding=1), (1, 512, 128, 128)),
    ("2d 3x3 256ch 256^2 (VAE)", lambda: nn.Conv2d(256, 256, 3, padding=1), (1, 256, 256, 256)),
    ("2d 3x3 128ch 512^2 (VAE)", lambda: nn.Conv2d(128, 128, 3, padding=1), (1, 128, 512, 512)),
    ("2d 1x1 2240->11200 32^2 (Sana FFN)", lambda: nn.Conv2d(2240, 11200, 1), (2, 2240, 32, 32)),
    ("2d dw3x3 11200ch 32^2 (Sana FFN)", lambda: nn.Conv2d(11200, 11200, 3, padding=1, groups=11200), (2, 11200, 32, 32)),
    ("3d 3x3x3 384ch 6x60x104 (Wan VAE)", lambda: nn.Conv3d(384, 384, 3, padding=1), (1, 384, 6, 60, 104)),
    ("3d 3x3x3 192ch 6x240x416 (Wan VAE)", lambda: nn.Conv3d(192, 192, 3, padding=1), (1, 192, 6, 240, 416)),
    ("3d 3x3x3 96ch 6x480x832 (Wan VAE)", lambda: nn.Conv3d(96, 96, 3, padding=1), (1, 96, 6, 480, 832)),
]


def main():
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(), "cudnn", torch.backends.cudnn.version())
    print(f"{'case':<38} {'bf16':>9} {'fp16':>9} {'fp32':>9} {'tbf16':>9}  (ms)   rel err vs fp32 conv of the bf16 inputs")
    with torch.no_grad():
        for name, make, shape in CASES:
            conv = make().cuda()
            x32 = torch.randn(shape, device="cuda")
            res = {}
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                c, x = conv.to(dt), x32.to(dt)
                try:
                    res[dt] = tt.do_bench(lambda: c(x), rep=100)
                except torch.OutOfMemoryError:
                    res[dt] = float("nan")  # printed as nan = OOM
                del c, x
                torch.cuda.empty_cache()
            c = conv.bfloat16()
            ref = nn.functional.conv2d if c.weight.dim() == 4 else nn.functional.conv3d
            xb = x32.bfloat16()
            r = ref(xb.float(), c.weight.float(), c.bias.float(), c.stride, c.padding, c.dilation, c.groups)
            tc = convert_conv(copy.deepcopy(c))
            res["t"] = tt.do_bench(lambda: tc(xb), rep=100)
            err = ((tc(xb).float() - r).norm() / r.norm()).item()
            try:
                errb = ((c(xb).float() - r).norm() / r.norm()).item()
            except torch.OutOfMemoryError:
                errb = float("nan")
            print(f"{name:<38} {res[torch.bfloat16]:9.2f} {res[torch.float16]:9.2f} {res[torch.float32]:9.2f} "
                  f"{res['t']:9.2f}   tbf16 {err:.2e}  bf16 {errb:.2e}", flush=True)
            del conv, c, tc, x32, xb, r
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
