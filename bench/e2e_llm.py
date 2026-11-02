"""Unmodified HF bf16 LLM on one Turing GPU: native bf16 vs tbf16.patch() vs tbf16.convert().

--ref also measures accuracy against the same weights run in fp32 on a second GPU.
"""
import argparse, contextlib, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import tbf16

TEXT = open(__file__).read() + " ".join(f"The number {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(6000))
KEEP = 1024  # positions whose logits are compared


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / n


def compare(logits, ref):
    if not torch.isfinite(logits).all():
        return "NaN/inf in logits"
    lp, lr = F.log_softmax(logits.float(), -1), F.log_softmax(ref.float(), -1)
    kl = (lr.exp() * (lr - lp)).sum(-1).mean().item()
    top1 = (logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return f"KL vs fp32 {kl:.2e}   top-1 agree {top1 * 100:.2f}%"


def run_suite(name, model, ids_all, seqs, ctx=contextlib.nullcontext):
    res = {}
    with torch.no_grad(), ctx():
        for S in seqs:
            ids = ids_all[:, :S]
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                res[S] = timed(lambda: model(ids, logits_to_keep=1))[1]
                print(f"  {name:<12} prefill {S:>5}: {res[S] * 1e3:8.1f} ms  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB", flush=True)
            except torch.OutOfMemoryError:
                print(f"  {name:<12} prefill {S:>5}: OOM", flush=True)
        ids = ids_all[:, :256]
        g = dict(max_new_tokens=128, min_new_tokens=128, do_sample=False)
        out, t = timed(lambda: model.generate(ids, **g), n=1)
        print(f"  {name:<12} generate 128 tokens after 256: {t:.2f} s ({128 / t:.1f} tok/s)", flush=True)
    return res, out[:, 256:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--no-patch", action="store_true", help="skip patch() mode (it caches an fp16 copy of the weights)")
    ap.add_argument("--seqs", default="512,2048,4096,8192")
    a = ap.parse_args()
    seqs = list(map(int, a.seqs.split(",")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ids_all = tok(TEXT, return_tensors="pt").input_ids.cuda()
    print(a.model, torch.cuda.get_device_name(), f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GB")

    logits = {}
    ids = ids_all[:, :2048]
    if a.ref:
        with torch.no_grad():
            logits["native bf16"] = model(ids, logits_to_keep=KEEP).logits.cpu()
            if not a.no_patch:
                with tbf16.patch():
                    logits["tbf16 patch"] = model(ids, logits_to_keep=KEEP).logits.cpu()

    nat, gen_nat = run_suite("native bf16", model, ids_all, seqs)
    gens = {}
    if not a.no_patch:
        _, gens["patch"] = run_suite("tbf16 patch", model, ids_all, seqs, tbf16.patch)
    tbf16.convert(model)
    print(f"  converted: weights {torch.cuda.memory_allocated() / 2**30:.2f} GB")
    if a.ref:
        with torch.no_grad():
            logits["tbf16 convert"] = model(ids, logits_to_keep=KEEP).logits.cpu()
    con, gens["convert"] = run_suite("tbf16 convert", model, ids_all, seqs)
    for S in seqs:
        if S in con:
            print(f"  prefill {S:>5}: convert vs native " + (f"{nat[S] / con[S]:.2f}x" if S in nat else "(native OOM)"))
    for n, g in gens.items():
        diff = (g != gen_nat)[0].nonzero()
        print(f"  greedy output vs native ({n}): " + ("identical" if len(diff) == 0 else f"first differs at token {diff[0].item()} of 128"))

    if a.ref:
        # Fresh fp16 load: a .half()/.bfloat16() round trip would also round fp32 buffers (rotary
        # inv_freq) that load_state_dict does not restore.
        del model; torch.cuda.empty_cache(); tbf16.uninstall_sdpa()
        half = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        with torch.no_grad():
            logits["model.half()"] = half(ids, logits_to_keep=KEEP).logits.cpu()
        del half; torch.cuda.empty_cache()
        n = torch.cuda.device_count()
        # fp32 reference on the other GPUs, leaving ~10% of each free for activations
        mem = {0: "0GiB", **{i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.9 / 2**30)}GiB"
                             for i in range(1, n)}}
        ref_model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa",
                                                         device_map="auto", max_memory=mem).eval()
        with torch.no_grad():
            ref = ref_model(ids.to(ref_model.device), logits_to_keep=KEEP).logits.cpu()
        for n, l in logits.items():
            print(f"  {n:<14} {compare(l, ref)}")


if __name__ == "__main__":
    main()
