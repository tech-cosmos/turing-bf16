"""HF + tbf16 vs vLLM vs SGLang on one Tesla T4 (Modal), same prompts, same token counts.
Prefix caching is off in both engines and every timed prompt is different, so prefill is real.

    modal run bench/serving_compare.py --engines tbf16,vllm,sglang --models qwen,gemma

Per engine and model: time to first token (one request, 2048 and 4096 prompt tokens), decode
speed over 128 tokens, and the start of a greedy completion as a sanity check. Each engine runs
in its own image (they pin different torch versions). Results go to the `tbf16-cache` volume.
"""
import json
import time

import modal

MODELS = {"qwen": "Qwen/Qwen3-1.7B", "gemma": "unsloth/gemma-3-4b-it"}
LENGTHS = (2048, 4096)
DECODE = 128
FILLER = " ".join(f"The number {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(3000))
QUESTION = "\n\nQuestion: what is the capital of France? Answer:"

cache = modal.Volume.from_name("tbf16-cache", create_if_missing=True)
ENV = {"HF_HOME": "/cache/hf", "TORCH_EXTENSIONS_DIR": "/cache/torch_ext", "TRITON_DEFAULT_FP_FUSION": "1",
       "PYTHONUNBUFFERED": "1"}
app = modal.App("tbf16-serving-compare")

tbf16_image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential")
               .pip_install("turing-bf16==0.1.0", "transformers>=5", "accelerate", "nvidia-cuda-nvcc==13.2.*",
                            "nvidia-nvvm==13.2.*", "nvidia-cuda-crt==13.2.*").env(ENV))
vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm").env(ENV)
# SGLang's FlashInfer attention JIT-compiles kernels for sm_75 at startup, so it needs a full CUDA
# toolkit (nvcc) matching its torch build (CUDA 13.0).
sglang_image = (modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
                .apt_install("libnuma1").pip_install("sglang[all]")
                .env({**ENV, "CUDA_HOME": "/usr/local/cuda"}))


def prompt_ids(tok, n, variant=0):
    """Exactly n token ids: filler text, then a short question at the end. Each variant differs from
    the first token on, so no engine can reuse a cached prefix between timed runs."""
    q = tok(QUESTION, add_special_tokens=False).input_ids
    f = tok(f"Run {variant}. " + FILLER, add_special_tokens=False).input_ids
    return f[: n - len(q)] + q


def median(xs):
    return sorted(xs)[len(xs) // 2]


def report(engine, model, rows, note=""):
    out = {"engine": engine, "model": model, "note": note, "rows": rows}
    print(json.dumps(out), flush=True)
    with open("/cache/results/serving_compare.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    cache.commit()


@app.function(image=tbf16_image, gpu="T4", timeout=3600, volumes={"/cache": cache})
def run_tbf16(model_key: str):
    import os
    import torch
    import tbf16
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs("/cache/results", exist_ok=True)
    name = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tbf16.convert(model)
    rows = {}
    with torch.no_grad():
        for n in LENGTHS:
            ts = []
            for i in range(4):
                ids = torch.tensor([prompt_ids(tok, n, i)], device="cuda")
                torch.cuda.synchronize(); t = time.perf_counter()
                model(ids, logits_to_keep=1)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            rows[f"ttft_{n}_ms"] = median(ts[1:]) * 1e3
        ids = torch.tensor([prompt_ids(tok, 2048, 99)], device="cuda")
        model.generate(ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model.generate(ids, max_new_tokens=DECODE, min_new_tokens=DECODE, do_sample=False)
        torch.cuda.synchronize(); total = time.perf_counter() - t
        rows["decode_tok_s"] = (DECODE - 1) / (total - rows["ttft_2048_ms"] / 1e3)
        rows["sample"] = tok.decode(out[0, ids.shape[1]:ids.shape[1] + 24], skip_special_tokens=True)
    report("hf+tbf16", name, rows, "bf16 weights, eager HF transformers, tbf16.convert")


@app.function(image=vllm_image, gpu="T4", timeout=3600, volumes={"/cache": cache})
def run_vllm(model_key: str):
    import os
    os.makedirs("/cache/results", exist_ok=True)
    name = MODELS[model_key]
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        llm = LLM(name, dtype="auto", max_model_len=4096 + DECODE + 64, gpu_memory_utilization=0.92,
                  enable_prefix_caching=False)
        dtype = str(llm.llm_engine.model_config.dtype)
        tok = llm.get_tokenizer()
        rows = {"dtype": dtype}
        for n in LENGTHS:
            ts = []
            for i in range(4):
                p = TokensPrompt(prompt_token_ids=prompt_ids(tok, n, i))
                t = time.perf_counter()
                llm.generate([p], SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)
                ts.append(time.perf_counter() - t)
            rows[f"ttft_{n}_ms"] = median(ts[1:]) * 1e3
        llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(tok, 2048, 98))],
                     SamplingParams(max_tokens=16, temperature=0), use_tqdm=False)  # decode warm-up
        p = TokensPrompt(prompt_token_ids=prompt_ids(tok, 2048, 99))
        t = time.perf_counter()
        o = llm.generate([p], SamplingParams(max_tokens=DECODE, min_tokens=DECODE, temperature=0), use_tqdm=False)
        total = time.perf_counter() - t
        rows["decode_tok_s"] = (DECODE - 1) / (total - rows["ttft_2048_ms"] / 1e3)
        rows["sample"] = o[0].outputs[0].text[:120]
        report("vllm", name, rows, f"dtype=auto -> {dtype}")
    except Exception as e:  # e.g. does not fit, or unsupported on sm_75
        report("vllm", name, {}, f"failed: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")


@app.function(image=sglang_image, gpu="T4", timeout=3600, volumes={"/cache": cache})
def run_sglang(model_key: str):
    import os
    os.makedirs("/cache/results", exist_ok=True)
    name = MODELS[model_key]
    try:
        import sglang as sgl
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        llm = sgl.Engine(model_path=name, dtype="auto", context_length=4096 + DECODE + 64, mem_fraction_static=0.85,
                         disable_radix_cache=True)
        mc = getattr(getattr(llm, "tokenizer_manager", None), "model_config", None)
        rows = {"dtype": str(getattr(mc, "dtype", getattr(llm.server_args, "dtype", "auto")))}
        for n in LENGTHS:
            ts = []
            for i in range(4):
                ids = prompt_ids(tok, n, i)
                t = time.perf_counter()
                llm.generate(input_ids=[ids], sampling_params={"max_new_tokens": 1, "temperature": 0})
                ts.append(time.perf_counter() - t)
            rows[f"ttft_{n}_ms"] = median(ts[1:]) * 1e3
        llm.generate(input_ids=[prompt_ids(tok, 2048, 98)],
                     sampling_params={"max_new_tokens": 16, "temperature": 0})  # decode warm-up
        ids = prompt_ids(tok, 2048, 99)
        t = time.perf_counter()
        o = llm.generate(input_ids=[ids], sampling_params={"max_new_tokens": DECODE, "min_new_tokens": DECODE,
                                                           "temperature": 0})
        total = time.perf_counter() - t
        rows["decode_tok_s"] = (DECODE - 1) / (total - rows["ttft_2048_ms"] / 1e3)
        rows["sample"] = o[0]["text"][:120]
        out_ids = o[0].get("output_ids") or []
        rows["n_out"] = len(out_ids)
        rows["first_ids"] = list(out_ids[:12])
        llm.shutdown()
        report("sglang", name, rows, "dtype=auto")
    except Exception as e:
        report("sglang", name, {}, f"failed: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")


@app.local_entrypoint()
def main(engines: str = "tbf16,vllm,sglang", models: str = "qwen,gemma"):
    fns = {"tbf16": run_tbf16, "vllm": run_vllm, "sglang": run_sglang}
    calls = [fns[e].spawn(m) for e in engines.split(",") for m in models.split(",")]
    for c in calls:
        try:
            c.get()
        except Exception as e:
            print(f"[call failed] {type(e).__name__}: {e}")
