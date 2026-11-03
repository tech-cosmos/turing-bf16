1/ [image: 0_lead.png]
My GPU "supports" bf16.

torch.cuda.is_bf16_supported() → True on an RTX 2080 Ti or a T4.

Then bf16 runs slower than fp32.

Our lab server is full of 2080 Tis and every new model ships in bf16. So, out of pure frustration, we built turing-bf16 🧵

---
2/ [image: 1_hook.png]
Turing tensor cores do fp16, not bf16. PyTorch quietly falls back to a slow path:

2080 Ti: bf16 7.6 TFLOPS, fp32 13.3
T4: bf16 2.9 TFLOPS, fp32 4.2

Meanwhile the fp16 tensor cores sit right there at 57 / 31 TFLOPS.

---
3/ [image: 2_why.png]
The usual workarounds all hurt:

• bf16 as-is: slow, and attention runs out of memory at long context
• .half(): overflows past 65,504. Gemma 3 4B gives NaN
• fp32: 2× the memory

The way out: fp16 has MORE mantissa bits than bf16 (10 vs 7). It only lacks range.

---
4/
So: scale each row by a power of two (exact, it only touches the exponent), convert to fp16 losslessly, run the fp16 tensor cores with fp32 accumulation, undo the scale.

Same trick for attention, convs and the backward pass. No calibration, no clamping.

---
5/ [image: 3_models.png]
On unmodified Hugging Face / diffusers models (T4):

Qwen3-1.7B prefill: 7.4×
Wan2.1 video: 6.1×
Gemma 3 4B: 6.0× (where .half() gives NaN)
Sana image: 5.6×
LoRA fine-tuning: 2.6× (2080 Ti)

One line: tbf16.convert(model). Output matches native bf16.

---
6/ [image: 4_video.png]
Wan2.1 1.3B on a 16 GB T4: 27.2 s → 4.45 s per denoising step.

The full 81-frame 480p video runs out of memory in native bf16. With tbf16 it finishes in about 14 minutes.

---
7/
Honest notes:

• If your model is fp16-safe, .half() is as fast or a bit faster. tbf16 is for when it isn't, or when you don't know.
• Token-by-token decoding is memory bound, so it's the same speed.
• Ampere and newer have real bf16; tbf16 does nothing there.

---
8/
Works on RTX 20xx, T4 (Colab, Kaggle), Quadro RTX and TITAN RTX. Inference and training, transformers and diffusers.

`python -m tbf16` checks your GPU in about a minute.

Credit to WanGP, who used the same trick for convs.

Repo: [link]
