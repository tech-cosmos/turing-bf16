1/ [image: 0_lead.png]
PyTorch says my RTX 2080 Ti supports bf16:
torch.cuda.is_bf16_supported() → True

Then it runs bf16 slower than fp32.

Out of pure frustration, we built turing-bf16 🧵

---
2/ [image: 1_hook.png]
Turing tensor cores do fp16, not bf16. So PyTorch falls back to a slow path.

tbf16 runs bf16 on the fp16 tensor cores: 7–10× faster matmuls, bf16 accuracy.

---
3/ [image: 2_why.png]
The trick: fp16 has MORE precision than bf16, just less range.

Scale each row by 2^k (exact), convert to fp16 losslessly, run the tensor cores, undo 2^k.

No calibration, no clamping.

---
4/ [image: 3_models.png]
Unmodified models on a T4, one line: tbf16.convert(model)

Qwen3 7.4× · Wan2.1 6.1× · Gemma 3 6.0× · Sana 5.6×

Same accuracy as native bf16. Training works too.

---
5/ [image: 4_video.png]
Wan2.1 on a 16 GB T4: 27 s → 4.5 s per step.

The full 81-frame video runs out of memory in native bf16. With tbf16 it finishes in ~14 min.

---
6/
If your model is fp16-safe, .half() is still fine. tbf16 is for when it isn't, or when you don't know.

Credit to WanGP and fp16-fixer, who used similar tricks for specific models.

pip install turing-bf16
github.com/tech-cosmos/turing-bf16
