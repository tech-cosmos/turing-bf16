"""Run any repo command on a Modal GPU (T4 = Turing sm_75; A10G/L4/... for no-op checks).

    modal run bench/modal_app.py --gpu T4 --cmd "python -m tbf16"
    modal run bench/modal_app.py --gpu T4:3 --cmd "python -m bench.e2e_llm --ref"

The repo is copied into the image; Hugging Face downloads, compiled kernels and anything
written to /cache/results persist in the Modal volume `tbf16-cache`
(`modal volume get tbf16-cache results/<dir> .` to fetch).
"""
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    # torch's default PyPI build is CUDA 13.2; the CUDA 13 pip wheels include nvcc (CUDA 12's do not)
    .pip_install("torch==2.15.*", "numpy", "nvidia-cuda-nvcc==13.2.*", "nvidia-nvvm==13.2.*",
                 "nvidia-cuda-crt==13.2.*")
    .pip_install("ninja", "transformers>=5", "accelerate", "peft", "diffusers",
                 "ftfy", "sentencepiece", "protobuf", "matplotlib", "imageio", "imageio-ffmpeg")
    .env({"HF_HOME": "/cache/hf", "TORCH_EXTENSIONS_DIR": "/cache/torch_ext", "TRITON_CACHE_DIR": "/cache/triton",
          "TRITON_DEFAULT_FP_FUSION": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT, "/root/turing-bf16", ignore=[".venv", "results", "**/__pycache__", "docs/images"])
)
cache = modal.Volume.from_name("tbf16-cache", create_if_missing=True)
app = modal.App("turing-bf16", image=image)


@app.function(gpu="T4", timeout=3 * 3600, volumes={"/cache": cache})
def run(cmd: str) -> int:
    subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit "
                   "--format=csv,noheader", shell=True)
    pathlib.Path("/cache/results").mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.run(cmd, shell=True, cwd="/root/turing-bf16").returncode
    finally:
        cache.commit()


@app.local_entrypoint()
def main(cmd: str = "python -m tbf16", gpu: str = "T4"):
    code = run.with_options(gpu=gpu).remote(cmd)
    print(f"[modal] exit code {code}")
