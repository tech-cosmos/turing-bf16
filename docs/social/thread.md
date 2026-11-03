1/ [images: 0_lead.png, 1_hook.png]
PyTorch says my RTX 2080 Ti supports bf16. Then it runs bf16 slower than fp32.

So we built turing-bf16: run bf16 models on the fp16 tensor cores of Turing GPUs (RTX 20xx, T4), with bf16 accuracy.

One line: tbf16.convert(model) 🧵

---
2/ [images: 2_why.png, 3_models.png, 4_video.png]
The trick: fp16 has more precision than bf16, just less range. Scale each row by a power of two (exact), run fp16 tensor cores, undo the scale.

vs plain PyTorch bf16 on a T4: Qwen3 7.4×, Wan2.1 6.1×, Gemma 3 6.0× (where .half() gives NaN). Training works too.

---
3/
Honest scope: a PyTorch library, not a serving engine. If your model works in fp16, use fp16. tbf16 is for bf16-only models, or when you're unsure.

Credit to WanGP and fp16-fixer for similar per-model tricks.

github.com/tech-cosmos/turing-bf16
pypi.org/project/turing-bf16
