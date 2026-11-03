1/ [images: 0_lead.png, 1_hook.png]
torch.cuda.is_bf16_supported() on a 2080 Ti says True.

then bf16 runs at 7.6 TFLOPS. fp32 does 13. the fp16 tensor cores do 57 and sit there idle.

our lab is full of these cards and every new model ships in bf16, so we got annoyed and fixed it. ~7x faster, same accuracy 🧵
---
2/ [images: 2_why.png, 3_models.png, 4_video.png]
the trick: fp16 has more precision than bf16, just less range. scale each row by a power of two (exact), run the fp16 tensor cores, undo the scale.

on a T4 vs plain pytorch bf16: qwen3 7.4x, wan2.1 6.1x, gemma 3 6x (.half() gives NaN there). training works too
---
3/
to be clear, it's a pytorch library, not a serving engine. if your model already works in fp16, use fp16. this is for bf16-only models, or when you're not sure.

credit to WanGP and fp16-fixer for similar per-model tricks.

github.com/tech-cosmos/turing-bf16
pypi.org/project/turing-bf16
