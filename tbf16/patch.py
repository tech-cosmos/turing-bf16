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

# Fewer query rows than this (decoding) run attention natively: it is as fast and skips
# the scaling kernels. Linear layers with a single row run natively too (memory bound).
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
                if x.numel() // x.shape[-1] >= 2:
                    self.stats["linear"] += 1
                    return scaled_linear(x, *self._weight(w), b)
                self.stats["linear_native"] += 1
        elif func is F.scaled_dot_product_attention:
            kwargs = dict(kwargs)  # q, k, v may come as keywords (diffusers passes query=, key=, value=)
            q, k, v = (args[i] if i < len(args) else kwargs.pop(n) for i, n in enumerate(("query", "key", "value")))
            rest = args[3:]
            # Few query rows (decoding): native bf16 is as fast and skips the scaling kernels.
            if _ok(q, k, v) and not torch.is_grad_enabled() and q.dim() == 4 and q.shape[-1] <= 256 and q.shape[-2] >= self.min_rows:
                self.stats["sdpa"] += 1
                return sdpa(q, k, v, *rest, **kwargs)
            self.stats["sdpa_native"] += 1
            return func(q, k, v, *rest, **kwargs)
        return func(*args, **kwargs)


def _ok(*ts):
    return all(t.is_cuda and t.dtype == torch.bfloat16 for t in ts)


def patch(min_rows=MIN_ROWS):
    """Context manager. Inference only for now: calls with grad enabled run natively."""
    return TuringBF16Mode(min_rows)
