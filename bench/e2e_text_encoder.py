"""T5-family text encoders (as used by Wan2.1, Flux, SD3, CogVideoX) in fp32 / bf16 / tbf16 / fp16.

T5 activations are known to exceed fp16's range; Hugging Face's T5 code clamps them when it
runs in fp16. This compares the embeddings each variant produces against an fp32 run and
reports the largest activation seen inside the fp32 model.

    python -m bench.e2e_text_encoder --model wan-umt5        # needs 2+ GPUs for the fp32 reference
"""
import argparse
import time

import torch

import tbf16

MODELS = {
    "wan-umt5": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "text_encoder", "tokenizer", "UMT5EncoderModel"),
    "flan-t5-xl": ("google/flan-t5-xl", None, None, "T5EncoderModel"),
}
PROMPTS = [
    "A cat walks on the grass, realistic, golden hour, shallow depth of field.",
    "An astronaut rides a horse across the surface of Mars, cinematic wide shot, dust in the air.",
    "Close-up of raindrops on a window at night, neon city lights blurred in the background.",
    "A paper boat drifts down a stream in autumn, leaves floating around it, slow camera pan.",
    "Time-lapse of clouds over a mountain lake, reflections on the water, 4k.",
    "A chef flips vegetables in a wok, flames rising, steam, warm kitchen lighting.",
    "Two robots play chess in a library, dramatic lighting, dust particles in sunbeams.",
    "Aerial shot of waves crashing on black volcanic sand, foam patterns, overcast sky.",
]


def load(a, dtype, **kw):
    import transformers
    repo, sub, tok_sub, cls = MODELS[a.model]
    kwargs = {"subfolder": sub} if sub else {}
    model = getattr(transformers, cls).from_pretrained(repo, dtype=dtype, **kwargs, **kw).eval()
    tok = transformers.AutoTokenizer.from_pretrained(repo, **({"subfolder": tok_sub} if tok_sub else {}))
    return model, tok


def encode(model, ids, mask):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).last_hidden_state


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="wan-umt5")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--ref-device", default="auto", help="'cpu' for the fp32 reference, or 'auto' (other GPUs)")
    ap.add_argument("--suffix", default="", help="appended to every prompt (to vary the inputs)")
    ap.add_argument("--force", action="store_true", help="convert on non-Turing GPUs too (correctness checks)")
    a = ap.parse_args()
    print(a.model, torch.cuda.get_device_name(), flush=True)

    model, tok = load(a, torch.bfloat16)
    model.cuda()
    t = tok([p + a.suffix for p in PROMPTS], padding="max_length", max_length=a.max_len, truncation=True, return_tensors="pt")
    ids, mask = t.input_ids.cuda(), t.attention_mask.cuda()
    outs, times = {}, {}
    outs["bf16"], times["bf16"] = timed(lambda: encode(model, ids, mask))
    tbf16.convert(model, force=a.force)
    outs["tbf16"], times["tbf16"] = timed(lambda: encode(model, ids, mask))
    del model; torch.cuda.empty_cache(); tbf16.uninstall_sdpa()

    model, _ = load(a, torch.float16)
    model.cuda()
    outs["fp16"], times["fp16"] = timed(lambda: encode(model, ids, mask))
    del model; torch.cuda.empty_cache()

    n = torch.cuda.device_count()
    if a.ref_device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model, _ = load(a, torch.float32)
    else:
        mem = {0: "0GiB", **{i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.9 / 2**30)}GiB"
                             for i in range(1, n)}} if n > 1 else None
        model, _ = load(a, torch.float32, device_map="auto" if mem else None, max_memory=mem)
        if not mem:
            model.cuda()
    amax = {"value": 0.0, "where": ""}

    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o) and o.is_floating_point():
            m = o.abs().max().item()
            if m > amax["value"]:
                amax["value"], amax["where"] = m, type(mod).__name__
    hooks = [m.register_forward_hook(hook) for m in model.modules()]
    dev = next(model.parameters()).device
    ref = encode(model, ids.to(dev), mask.to(dev)).to("cuda:0").double()
    for h in hooks:
        h.remove()

    m = mask.bool()[..., None]
    print(f"largest activation inside the fp32 model: {amax['value']:.3g} (in {amax['where']}); fp16 max is 65504")
    print(f"{'variant':<8} {'encode ms':>10} {'rel err vs fp32':>16} {'min cos':>9} {'finite':>7}")
    for k in ("bf16", "tbf16", "fp16"):
        o = outs[k].double()
        finite = bool(torch.isfinite(o).all())
        d = ((o - ref) * m).norm() / (ref * m).norm()
        cos = torch.nn.functional.cosine_similarity(o[m.expand_as(o)].view(-1, o.shape[-1]),
                                                    ref[m.expand_as(ref)].view(-1, ref.shape[-1]), dim=-1)
        print(f"{k:<8} {times[k] * 1e3:10.1f} {d.item():16.3e} {cos.min().item():9.4f} {str(finite):>7}")


if __name__ == "__main__":
    main()
