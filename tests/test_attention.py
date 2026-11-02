import math
import torch, torch.nn.functional as F
import triton.testing as tt
from tbf16.attention import sdpa


def ref_attn(q, k, v, causal):
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        S = s.shape[-1]
        s = s.masked_fill(torch.ones(S, S, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v


def rel(y, r):
    return "NaN/inf" if not torch.isfinite(y).all() else f"{((y.double() - r).norm() / r.norm()).item():.2e}"


def main():
    torch.manual_seed(0)
    B, H, S, D = 1, 8, 2048, 128
    base = [torch.randn(B, H, S, D, device="cuda").bfloat16() for _ in range(3)]
    cases = {
        "gaussian": (1, 1, 1),
        "q,k x30 (peaky softmax)": (30, 30, 1),
        "v x1e5": (1, 1, 1e5),
        "v x1e-7": (1, 1, 1e-7),
        "q x1e3, k x1e-3": (1e3, 1e-3, 1),
        "q,k x300, v x1e6": (300, 300, 1e6),
    }
    for causal in (False, True):
        for name, (a, b, c) in cases.items():
            q, k, v = base[0] * a, base[1] * b, base[2] * c
            r = ref_attn(q, k, v, causal)
            outs = {"torch bf16": F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                    "naive fp16": F.scaled_dot_product_attention(q.half(), k.half(), v.half(), is_causal=causal),
                    "tbf16": sdpa(q, k, v, is_causal=causal)}
            print(f"{'causal ' if causal else ''}{name:<28}" + "  ".join(f"{n}: {rel(o, r):>8}" for n, o in outs.items()))
    print()
    for B, H, S, D in [(1, 24, 4096, 128), (1, 12, 8192, 128), (1, 24, 16384, 64), (4, 32, 2048, 128)]:
        q, k, v = (torch.randn(B, H, S, D, device="cuda").bfloat16() for _ in range(3))
        fl = 4 * B * H * S * S * D / 1e9
        t16 = tt.do_bench(lambda: F.scaled_dot_product_attention(q.half(), k.half(), v.half()), rep=200)
        torch.cuda.reset_peak_memory_stats(); t = tt.do_bench(lambda: sdpa(q, k, v), rep=200)
        print(f"{str((B, H, S, D)):>20}  fp16 {fl / t16:6.2f} TFLOPS   tbf16 {fl / t:6.2f} TFLOPS {t:8.2f} ms  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
