# Social images

1600x900 cards for posts. Sources are plain HTML; to regenerate, serve this folder
(`python3 -m http.server`) and screenshot each page at a 1600x900 viewport.

| File | Message |
|---|---|
| `0_lead.png` | thread lead: `is_bf16_supported()` says True, then every option fails |
| `1_hook.png` | bf16 on Turing is slower than fp32; tbf16 runs at fp16 speed |
| `2_why.png` | the three bad options, and the exact power-of-two trick |
| `3_models.png` | end-to-end speedups on real models (T4) |
| `4_video.png` | Wan2.1: same frame, 6x faster |

`thread.md` is the matching post text (each part under 280 characters).
