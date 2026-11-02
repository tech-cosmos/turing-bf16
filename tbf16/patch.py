"""Drop-in: run unmodified bf16 PyTorch code on fp16 tensor cores.

    with tbf16.patch():
        out = model(**inputs)

Intercepts F.linear and F.scaled_dot_product_attention on bf16 CUDA tensors. Weight
conversions are cached per weight tensor and redone if the weight changes in place.
Everything else runs as before.
"""
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils.weak import WeakIdKeyDictionary

from .attention import sdpa
from .ops import prepare_weight, scaled_linear

# Below this many rows the GEMM is memory bound and native bf16 is as fast.
MIN_ROWS = 16


class TuringBF16Mode(TorchFunctionMode):
    def __init__(self, min_rows=MIN_ROWS):
        super().__init__()
        self.min_rows = min_rows
        self._weights = WeakIdKeyDictionary()
        self.stats = {"linear": 0, "linear_native": 0, "sdpa": 0, "sdpa_native": 0}

    def _weight(self, w):
        hit = self._weights.get(w)
        if hit is None or hit[0] != w._version:
            hit = (w._version, *prepare_weight(w.detach()))
            self._weights[w] = hit
        return hit[1], hit[2]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            if _ok(x, w) and not torch.is_grad_enabled():
                if x.numel() // x.shape[-1] >= self.min_rows:
                    self.stats["linear"] += 1
                    return scaled_linear(x, *self._weight(w), b)
                self.stats["linear_native"] += 1
        elif func is F.scaled_dot_product_attention:
            q, k, v = args[:3]
            # Few query rows (decoding): native bf16 is as fast and skips the scaling kernels.
            if _ok(q, k, v) and not torch.is_grad_enabled() and q.shape[-1] <= 128 and q.shape[-2] >= self.min_rows:
                self.stats["sdpa"] += 1
                return sdpa(q, k, v, *args[3:], **kwargs)
            self.stats["sdpa_native"] += 1
        return func(*args, **kwargs)


def _ok(*ts):
    return all(t.is_cuda and t.dtype == torch.bfloat16 for t in ts)


def patch(min_rows=MIN_ROWS):
    """Context manager. Inference only for now: calls with grad enabled run natively."""
    return TuringBF16Mode(min_rows)
