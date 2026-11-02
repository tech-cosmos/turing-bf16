"""Phase 0: how fast are plain torch matmuls on this GPU in each dtype?"""
import torch, triton.testing as tt

SHAPES = [  # (M, K, N): M = tokens, K = in features, N = out features
    (4096, 4096, 4096),
    (8192, 3072, 3072),
    (8192, 3072, 12288),
    (8192, 12288, 3072),
    (16384, 5120, 5120),
    (1, 4096, 4096),
    (16, 4096, 11008),
]

def tflops(ms, M, K, N):
    return 2 * M * K * N / ms / 1e9

def main():
    print(torch.cuda.get_device_name(), torch.__version__)
    print(f"{'shape':>22} " + " ".join(f"{d:>9}" for d in ("fp32", "tf32off", "fp16", "bf16")))
    for M, K, N in SHAPES:
        row = []
        for dt in (torch.float32, None, torch.float16, torch.bfloat16):
            a = torch.randn(M, K, device="cuda", dtype=dt or torch.float32)
            b = torch.randn(K, N, device="cuda", dtype=dt or torch.float32)
            ms = tt.do_bench(lambda: a @ b, warmup=50, rep=300)
            row.append(tflops(ms, M, K, N))
        print(f"{str((M, K, N)):>22} " + " ".join(f"{x:9.2f}" for x in row))

if __name__ == "__main__":
    main()
