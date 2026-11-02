import torch, triton.testing as tt
from tbf16 import prepare_weight, scaled_linear
from tbf16.ops import small_m_linear

for M in (1, 2, 4, 8, 15, 16, 32):
    for K, N in ((2048, 6144), (4096, 11008), (11008, 4096)):
        x = torch.randn(M, K, device="cuda").bfloat16() * 100
        w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
        w16, ws = prepare_weight(w)
        ref = x.double() @ w.double().t()
        y = small_m_linear(x, w16, ws)
        err = ((y.double() - ref).norm() / ref.norm()).item()
        t_nat = tt.do_bench(lambda: x @ w.t()); t_s = tt.do_bench(lambda: small_m_linear(x, w16, ws))
        x16, w16b = x.half(), w.half()
        t_tc = tt.do_bench(lambda: x16 @ w16b.t())
        print(f"M={M:>3} K={K:>5} N={N:>5}  err {err:.1e}  native bf16 {t_nat * 1e3:6.1f} us  small-M {t_s * 1e3:6.1f} us  fp16 cublas {t_tc * 1e3:6.1f} us")
