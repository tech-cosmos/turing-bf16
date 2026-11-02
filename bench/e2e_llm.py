"""Unmodified HF bf16 LLM: native bf16 vs tbf16.patch() on one 2080 Ti.

Accuracy is measured against the same weights run in fp32 on a second GPU.
"""
import argparse, contextlib, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import tbf16

TEXT = open(__file__).read() + " ".join(f"The number {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(4000))


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / n


def compare(logits, ref):
    lp, lr = F.log_softmax(logits.float(), -1), F.log_softmax(ref.float(), -1)
    kl = (lr.exp() * (lr - lp)).sum(-1).mean().item()
    top1 = (logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return f"KL vs fp32 {kl:.2e}  top-1 agree {top1 * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ref", action="store_true", help="fp32 reference on cuda:1")
    ap.add_argument("--seqs", default="512,2048,4096")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ids_all = tok(TEXT, return_tensors="pt").input_ids.cuda()
    print(a.model, torch.cuda.get_device_name())

    if a.ref:
        ids = ids_all[:, :2048]
        with torch.no_grad():
            nat = model(ids, logits_to_keep=1024).logits.cpu()
            mode = tbf16.patch()
            with mode:
                ours = model(ids, logits_to_keep=1024).logits.cpu()
            print("calls:", mode.stats)
            naive = model.half()(ids, logits_to_keep=1024).logits.cpu()
            model.bfloat16()
            ref_model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").to("cuda:1").eval()
            ref = ref_model(ids.to("cuda:1"), logits_to_keep=1024).logits.cpu()
            del ref_model
        for name, l in [("native bf16", nat), ("tbf16", ours), ("model.half()", naive)]:
            print(f"  {name:<13}", "NaN/inf in logits" if not torch.isfinite(l).all() else compare(l, ref))

    with torch.no_grad():
        for S in map(int, a.seqs.split(",")):
            ids = ids_all[:, :S]
            res = {}
            for name, ctx in [("native bf16", contextlib.nullcontext), ("tbf16", tbf16.patch)]:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    with ctx():
                        res[name] = timed(lambda: model(ids, logits_to_keep=1))[1]
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    print(f"  prefill {S:>5} tokens: {name:<12} {res[name] * 1e3:8.1f} ms  peak {peak:5.2f} GB")
                except torch.OutOfMemoryError:
                    print(f"  prefill {S:>5} tokens: {name:<12} OOM")
            if len(res) == 2:
                print(f"  prefill {S:>5} tokens: speedup {res['native bf16'] / res['tbf16']:.2f}x")

        ids = ids_all[:, :256]
        g = dict(max_new_tokens=64, do_sample=False, min_new_tokens=64)
        out_nat, t_nat = timed(lambda: model.generate(ids, **g), n=1)
        with tbf16.patch():
            out_us, t_us = timed(lambda: model.generate(ids, **g), n=1)
        same = (out_nat == out_us).float()[:, 256:].mean().item()
        print(f"  generate 64 tokens: native {t_nat:.2f} s   tbf16 {t_us:.2f} s   greedy tokens identical: {same * 100:.0f}%")


if __name__ == "__main__":
    main()
