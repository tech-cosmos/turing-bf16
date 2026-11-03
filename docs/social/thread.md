1/ [images: 3_models.png, 4_video.png]
have an RTX 20-series, T4 or Quadro RTX? your bf16 models run slower than fp32 on these cards.

we made a pip package that fixes it: up to 7x faster on Qwen3, Wan2.1 video, Gemma 3 and Sana, same accuracy, no changes to your model 🧵

---
2/ [images: 1_hook.png, 2_why.png]
why: these GPUs have fast fp16 tensor cores but none for bf16, so pytorch falls back to a slow path. casting to fp16 is fast but breaks models like Gemma 3 (NaN).

turing-bf16 runs bf16 on the fp16 cores without that problem. works for inference and fine-tuning.

---
3/
it's a pytorch library, not a serving engine. if your model already works in fp16, just use fp16.

credit to WanGP and fp16-fixer for similar per-model tricks.

github.com/tech-cosmos/turing-bf16
pypi.org/project/turing-bf16
