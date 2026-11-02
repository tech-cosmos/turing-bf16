"""How does torch SDPA do in bf16 vs fp16 on this GPU? (B, H, S, D)"""
import torch, torch.nn.functional as F, triton.testing as tt

for B, H, S, D in [(1, 24, 4096, 128), (1, 12, 8192, 128), (1, 24, 16384, 64), (4, 32, 2048, 128)]:
    row = [f"{str((B, H, S, D)):>20}"]
    for dt in (torch.float16, torch.bfloat16):
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=dt) for _ in range(3))
        torch.cuda.reset_peak_memory_stats()
        try:
            ms = tt.do_bench(lambda: F.scaled_dot_product_attention(q, k, v), rep=200)
            mem = torch.cuda.max_memory_allocated() / 2**30
            row.append(f"{str(dt)[6:]}: {4 * B * H * S * S * D / ms / 1e9:6.2f} TFLOPS {ms:8.2f} ms {mem:5.2f} GB")
        except torch.OutOfMemoryError:
            row.append(f"{str(dt)[6:]}: OOM")
        del q, k, v
        torch.cuda.empty_cache()
    print("   ".join(row))
for dt in (torch.float16, torch.bfloat16):
    q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=dt)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q)
            print(dt, be.name, "ok")
        except RuntimeError as e:
            print(dt, be.name, "unsupported")
