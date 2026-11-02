"""Compare each method against an fp64 reference computed from the same bf16 inputs."""
import torch
from tbf16 import prepare_weight, scaled_linear


def rel_err(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def case(name, x, w):
    ref = x.double() @ w.double().t()
    out = {
        "torch bf16": x @ w.t(),
        "naive fp16 cast": (x.half() @ w.half().t()),
        "tbf16 v0": scaled_linear(x, *prepare_weight(w)),
    }
    print(f"{name:<34}" + "".join(
        f"{k}: {'NaN/inf' if not torch.isfinite(v).all() else f'{rel_err(v, ref):.2e}'}   " for k, v in out.items()))


def main():
    torch.manual_seed(0)
    M, K, N = 4096, 3072, 3072
    x = torch.randn(M, K, device="cuda").bfloat16()
    w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
    case("gaussian", x, w)
    xb = x.clone(); xb[::97] *= 3e4  # a few "massive activation" tokens, common in DiTs/LLMs
    case("outlier tokens (x3e4)", xb, w)
    case("everything x1e6", x * 1e6, w)
    case("tiny (x1e-8)", x * 1e-8, w)
    xc = x.clone(); xc[:, :8] *= 1e3  # outlier channels
    case("outlier channels (x1e3)", xc, w)


if __name__ == "__main__":
    main()
