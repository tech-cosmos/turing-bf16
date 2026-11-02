"""Training on one Turing GPU: precision recipes compared on identical init and batches.

  scratch: ~75M-param Qwen3-architecture byte-level LM trained from scratch on the Python
           standard library source (no download needed)
  lora:    LoRA (r=16, all linear layers) on an HF model

Recipes: fp32 | bf16 (fp32 master + bf16 autocast) | bf16+tbf16 (same, tbf16.convert(train=True))
         | fp16-amp (fp32 master + fp16 autocast + GradScaler, the usual Turing recipe)
For lora the base weights are stored in bf16 (fp16 for fp16-amp) and frozen.
"""
import argparse, json, glob, math, os, sys, sysconfig, time
import torch
import tbf16


def stdlib_bytes():
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**", "*.py"), recursive=True))
    return torch.frombuffer(bytearray(b"".join(open(f, "rb").read() for f in files)), dtype=torch.uint8)


def batches(data, n, batch, seq, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        idx = torch.randint(0, len(data) - seq - 1, (batch,), generator=g)
        yield torch.stack([data[i:i + seq + 1] for i in idx]).long()


def build(a):
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    if a.task == "scratch":
        cfg = Qwen3Config(vocab_size=256, hidden_size=768, intermediate_size=2048, num_hidden_layers=12,
                          num_attention_heads=12, num_key_value_heads=4, head_dim=64, max_position_embeddings=4096,
                          tie_word_embeddings=False, attn_implementation="sdpa")
        model = Qwen3ForCausalLM(cfg).cuda()
        if a.recipe.startswith("bf16-pure"):
            model.bfloat16()
        return model, None
    from peft import LoraConfig, get_peft_model
    base_dtype = torch.float16 if a.recipe == "fp16-amp" else torch.float32 if a.recipe == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=base_dtype, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0))
    tok = __import__("transformers").AutoTokenizer.from_pretrained(a.model)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["scratch", "lora"], default="scratch")
    ap.add_argument("--recipe", choices=["fp32", "bf16", "bf16+tbf16", "fp16-amp", "bf16-pure", "bf16-pure+tbf16"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--out", default="results/train.jsonl")
    a = ap.parse_args()
    model, tok = build(a)
    if a.recipe.endswith("+tbf16"):
        tbf16.convert(model, train=True)
    model.train()  # from_pretrained returns eval mode, which also disables gradient checkpointing
    data = stdlib_bytes()
    if tok is not None:  # token ids for lora; byte ids for scratch
        text = bytes(data[: 20_000_000].tolist()).decode("utf-8", "ignore")
        data = torch.tensor(tok(text).input_ids, dtype=torch.int32)
    split = int(len(data) * 0.95)
    train, held = data[:split], data[split:]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    amp_dtype = {"bf16": torch.bfloat16, "bf16+tbf16": torch.bfloat16, "fp16-amp": torch.float16}.get(a.recipe)
    scaler = torch.amp.GradScaler("cuda", enabled=a.recipe == "fp16-amp")

    def loss_of(x):
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(input_ids=x[:, :-1]).logits
        return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())

    log, times, skipped = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for step, x in enumerate(batches(train, a.steps, a.batch, a.seq, seed=1)):
        x = x.cuda(non_blocking=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss = loss_of(x)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        old_scale = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if old_scale is not None and scaler.get_scale() < old_scale:
            skipped += 1
        if step >= 5:
            times.append(dt)
        l = loss.item()
        log.append(l)
        if step % 25 == 0 or step == a.steps - 1:
            print(f"{a.recipe:<16} step {step:>4} loss {l:.4f} {dt * 1e3:7.1f} ms", flush=True)
        if not math.isfinite(l):
            print("non-finite loss, stopping"); break
    model.eval()
    with torch.no_grad():
        ev = [loss_of(x.cuda()).item() for x in batches(held, 20, a.batch, a.seq, seed=2)]
    times.sort()
    res = dict(task=a.task, recipe=a.recipe, model=a.model if a.task == "lora" else "qwen3-75m-bytes", steps=a.steps,
               batch=a.batch, seq=a.seq, median_step_ms=times[len(times) // 2] * 1e3,
               tokens_per_s=a.batch * a.seq / times[len(times) // 2], peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               final_train_loss=sum(log[-20:]) / len(log[-20:]), eval_loss=sum(ev) / len(ev), skipped_steps=skipped,
               loss_curve=log, gpu=torch.cuda.get_device_name())
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
