"""JIT-builds the CUDA kernels on first use (cached under ~/.cache/torch_extensions)."""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent
_mod = None
_failed = None
_gemm = None
_gemm_failed = None


def _cuda_home():
    if os.environ.get("CUDA_HOME"):
        return
    try:  # nvcc from the nvidia-cuda-nvcc wheel, matching torch's CUDA major version
        import nvidia
        for root in nvidia.__path__:
            for d in sorted(pathlib.Path(root).glob("cu*")):
                if (d / "bin" / "nvcc").exists():
                    os.environ["CUDA_HOME"] = str(d)
                    return
    except ImportError:
        pass


def _ldflags():
    """pip CUDA wheels ship libcudart.so.13 but no libcudart.so for the linker; add a link."""
    home = os.environ.get("CUDA_HOME")
    if not home or (pathlib.Path(home) / "lib" / "libcudart.so").exists() or (pathlib.Path(home) / "lib64").exists():
        return []
    real = sorted((pathlib.Path(home) / "lib").glob("libcudart.so.*"))
    if not real:
        return []
    d = pathlib.Path.home() / ".cache" / "tbf16" / "lib"
    d.mkdir(parents=True, exist_ok=True)
    link = d / "libcudart.so"
    if not link.exists():
        link.symlink_to(real[0])
    return [f"-L{d}"]


def _load(name, src):
    _cuda_home()
    bindir = os.path.dirname(sys.executable)  # ninja from the venv, even when it is not activated
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    from torch.utils.cpp_extension import load
    return load(name=name, sources=[str(_HERE / "csrc" / src)], extra_ldflags=_ldflags(),
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_75,code=sm_75", "--use_fast_math"],
                verbose=bool(os.environ.get("TBF16_VERBOSE")))


def ext():
    global _mod
    if _mod is None:
        _mod = _load("tbf16_ext", "small_kernels.cu")
    return _mod


def available() -> bool:
    """True if the CUDA kernels build/load here. On failure, warns once and tbf16 falls back to
    its Triton/torch paths (correct, slower to launch)."""
    global _failed
    if _mod is not None:
        return True
    if _failed is not None:
        return False
    try:
        ext()
        return True
    except Exception as e:  # missing nvcc/ninja, compiler mismatch, ...
        _failed = e
        import warnings
        warnings.warn(f"tbf16: CUDA kernels unavailable, using Triton/torch fallbacks ({type(e).__name__}: "
                      f"{str(e).splitlines()[0][:200]}). See README 'Install'.")
        return False


def gemm_ext():
    """Fused bf16 x fp16 GEMM (csrc/gemm_sm75.cu)."""
    global _gemm
    if _gemm is None:
        _gemm = _load("tbf16_gemm", "gemm_sm75.cu")
    return _gemm


def gemm_available() -> bool:
    """True if the fused GEMM builds/loads; otherwise scaled_linear keeps the cuBLAS path."""
    global _gemm_failed
    if _gemm is not None:
        return True
    if _gemm_failed is not None or not available():
        return False
    try:
        gemm_ext()
        return True
    except Exception as e:
        _gemm_failed = e
        return False
