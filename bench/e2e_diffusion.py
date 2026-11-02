"""Unmodified diffusers bf16 pipelines on Turing: native bf16 vs tbf16.convert() vs fp16 vs fp32.

    python -m bench.e2e_diffusion --model wan --sizes 17x320x512,33x480x832
    python -m bench.e2e_diffusion --model sana --sizes 1024x1024,2048x2048

Same prompt, seed and initial noise (CPU generator) for every variant. The prompt is encoded
once (cached on disk) so the text encoder never shares a GPU with the transformer.

  fp32   transformer + VAE in fp32 on the second visible GPU (reference)
  bf16   native bf16 transformer + VAE
  tbf16  the same bf16 weights after tbf16.convert(transformer) and tbf16.convert(vae)
  fp16   pipeline loaded with torch_dtype=float16 (diffusers keeps its fp32 modules in fp32)

Per variant and size: s/step (median after the first step), denoise time, VAE decode time,
peak memory per stage, OOM; PSNR/SSIM of the decoded frames vs bf16 and vs fp32; and the
relative error of one transformer forward vs fp32 on identical inputs (captured from the
fp32 run), which is not amplified by the sampler. If the VAE decode fails (OOM, or no cuDNN
engine for a bf16 conv), the latents are
decoded with an fp32 VAE so the denoiser can still be compared (marked "*").
"""
import argparse, gc, hashlib, inspect, json, math, os, pathlib, time
import torch
import torch.nn.functional as F

import tbf16

CACHE = pathlib.Path.home() / ".cache" / "tbf16_diffusion"
WAN_NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
           "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
           "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
           "picture, messy background, three legs, many people in the background, walking backwards")


# ---------------------------------------------------------------------------------------- models
class Wan:
    repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    prompt = "A cat walks on the grass, realistic, sunny afternoon, shallow depth of field"
    guidance = 5.0

    def embeds(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        from diffusers import WanPipeline
        # UMT5-XXL (5.7B) does not fit an 11 GB card next to anything else: encode on the CPU in fp32.
        tok = AutoTokenizer.from_pretrained(self.repo, subfolder="tokenizer")
        te = UMT5EncoderModel.from_pretrained(self.repo, subfolder="text_encoder", dtype=torch.float32)
        pipe = WanPipeline(tokenizer=tok, text_encoder=te, transformer=None, vae=None, scheduler=None)
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(self.prompt, WAN_NEG, do_classifier_free_guidance=True, device="cpu",
                                        dtype=torch.float32)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne)

    def load(self, dtype, device):
        from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
        tr = WanTransformer3DModel.from_pretrained(self.repo, subfolder="transformer", torch_dtype=dtype)
        vae = AutoencoderKLWan.from_pretrained(self.repo, subfolder="vae", torch_dtype=dtype)
        return WanPipeline.from_pretrained(self.repo, transformer=tr, vae=vae, text_encoder=None, tokenizer=None,
                                           torch_dtype=dtype).to(device)

    def call_kwargs(self, size, steps):
        f, h, w = size
        return dict(num_frames=f, height=h, width=w, num_inference_steps=steps, guidance_scale=self.guidance)

    @staticmethod
    def decode(vae, latents):
        c = vae.config
        latents = latents.to(vae.dtype)
        mean = torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1).to(latents)
        std = torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1).to(latents)
        video = vae.decode(latents * std + mean, return_dict=False)[0]  # (1, 3, T, H, W) in [-1, 1]
        return video[0].permute(1, 2, 3, 0)  # (T, H, W, 3)

    @staticmethod
    def parse_size(s):
        return tuple(map(int, s.split("x")))  # frames x height x width

    @staticmethod
    def tokens(size):
        f, h, w = size
        return ((f - 1) // 4 + 1) * (h // 16) * (w // 16)


class Sana:
    repo = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    prompt = ("a cyberpunk cat with a neon sign that says \"Sana\", rainy street at night, reflections, "
              "highly detailed")
    guidance = 4.5
    load_kwargs = dict(variant="bf16")

    def embeds(self):
        from diffusers import SanaPipeline
        pipe = SanaPipeline.from_pretrained(self.repo, transformer=None, vae=None, torch_dtype=torch.bfloat16,
                                            **self.load_kwargs).to("cuda:0")
        chi = inspect.signature(pipe.__call__).parameters["complex_human_instruction"].default
        with torch.no_grad():
            pe, pm, ne, nm = pipe.encode_prompt(self.prompt, do_classifier_free_guidance=True, negative_prompt="",
                                                device="cuda:0", complex_human_instruction=chi)
        return dict(prompt_embeds=pe.float().cpu(), prompt_attention_mask=pm.cpu(),
                    negative_prompt_embeds=ne.float().cpu(), negative_prompt_attention_mask=nm.cpu())

    def load(self, dtype, device):
        from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
        tr = SanaTransformer2DModel.from_pretrained(self.repo, subfolder="transformer", torch_dtype=dtype,
                                                    **self.load_kwargs)
        vae = AutoencoderDC.from_pretrained(self.repo, subfolder="vae", torch_dtype=dtype, **self.load_kwargs)
        return SanaPipeline.from_pretrained(self.repo, transformer=tr, vae=vae, text_encoder=None, tokenizer=None,
                                            torch_dtype=dtype).to(device)

    def call_kwargs(self, size, steps):
        h, w = size
        return dict(height=h, width=w, num_inference_steps=steps, guidance_scale=self.guidance,
                    use_resolution_binning=False, negative_prompt=None)

    @staticmethod
    def decode(vae, latents):
        img = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        return img.permute(0, 2, 3, 1)  # (1, H, W, 3)

    @staticmethod
    def parse_size(s):
        return tuple(map(int, s.split("x")))  # height x width

    @staticmethod
    def tokens(size):
        return (size[0] // 32) * (size[1] // 32)


MODELS = {"wan": Wan, "sana": Sana}


# --------------------------------------------------------------------------------------- metrics
def to_uint8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def psnr(a, b):
    mse = ((a.float() - b.float()) / 255).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1 / mse)


def ssim(a, b, device="cuda:0"):
    """Mean SSIM (gaussian 11x11, sigma 1.5) over frames and RGB channels."""
    g = torch.exp(-(torch.arange(11, device=device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum()).view(1, 1, 1, 11).expand(3, 1, 1, 11)

    def blur(x):
        return F.conv2d(F.conv2d(x, g, groups=3), g.transpose(-1, -2), groups=3)

    vals = []
    for i in range(0, a.shape[0], 8):
        x = a[i:i + 8].to(device).permute(0, 3, 1, 2).float() / 255
        y = b[i:i + 8].to(device).permute(0, 3, 1, 2).float() / 255
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(s.mean(dim=(1, 2, 3)).cpu())
    return torch.cat(vals).mean().item()


def rel_err(y, ref):
    if not torch.isfinite(y).all():
        return float("nan")
    return ((y.double() - ref.double()).norm() / ref.norm()).item()


# ----------------------------------------------------------------------------------------- runs
def cast_inputs(inp, dtype, device):
    """Captured fp32 transformer inputs -> what the pipeline passes for this dtype (latents and text
    embeddings in the transformer dtype; timesteps and masks unchanged)."""
    return {k: (v.to(device, dtype) if v.is_floating_point() and v.dim() > 1 else v.to(device))
            if torch.is_tensor(v) else v for k, v in inp.items()}


def failure(e):
    if isinstance(e, torch.OutOfMemoryError):
        return "OOM"
    if "unable to find an engine" in str(e):
        return "no-cuDNN-engine"
    raise e


def run_variant(name, m, pipe, embeds, sizes, a, results, outputs, capture=None, fp32_vae=None):
    dev = pipe._execution_device
    tdtype = pipe.transformer.dtype
    emb = {k: v.to(dev, tdtype) if v.is_floating_point() else v.to(dev) for k, v in embeds.items()}
    for size in sizes:
        key = "x".join(map(str, size))
        r = results.setdefault(key, {}).setdefault(name, {})
        gc.collect(); torch.cuda.empty_cache()
        hook = None
        if capture is not None and key == capture["size"]:
            calls = []

            def pre(mod, args, kwargs):
                if len(calls) == capture["call"]:
                    capture["inputs"] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in kwargs.items()}
                    for i, x in enumerate(args):
                        capture["inputs"]["hidden_states" if i == 0 else f"arg{i}"] = x.detach().cpu()
                calls.append(1)
            hook = pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            with torch.no_grad():
                # Warm-up: one step at this size (JIT, cuDNN heuristics, allocator).
                pipe(**emb, **m.call_kwargs(size, 1), generator=torch.Generator("cpu").manual_seed(a.seed),
                     output_type="latent")
                if hook is not None:
                    calls.clear()
                torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
                stamps = []

                def cb(p, i, t, kw):
                    torch.cuda.synchronize(dev); stamps.append(time.perf_counter()); return kw
                t0 = time.perf_counter()
                lat = pipe(**emb, **m.call_kwargs(size, a.steps), generator=torch.Generator("cpu").manual_seed(a.seed),
                           output_type="latent", callback_on_step_end=cb)
                lat = (lat.frames if hasattr(lat, "frames") else lat.images)
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter()
            steps = [b - c for b, c in zip(stamps[1:], stamps[:-1])]
            r.update(step_s=sorted(steps)[len(steps) // 2], denoise_s=t1 - t0,
                     peak_denoise_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                     latents_finite=bool(torch.isfinite(lat).all()))
        except (torch.OutOfMemoryError, RuntimeError) as e:
            r["denoise"] = failure(e)
            print(f"  {name:<6} {key:>12}: denoise {r['denoise']}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            continue
        finally:
            if hook is not None:
                hook.remove()
        lat_cpu = lat.float().cpu()
        outputs.setdefault(key, {})[name + ":latents"] = lat_cpu
        del lat
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        try:
            with torch.no_grad():
                m.decode(pipe.vae, lat_cpu.to(dev))  # warm-up (first call at a shape is slower)
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.perf_counter()
                frames = m.decode(pipe.vae, lat_cpu.to(dev))
                torch.cuda.synchronize(dev)
                r.update(decode_s=time.perf_counter() - t0, peak_decode_gb=torch.cuda.max_memory_allocated(dev) / 2**30)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            r["decode"] = failure(e)
            gc.collect(); torch.cuda.empty_cache()
            if fp32_vae is None:
                print(f"  {name:<6} {key:>12}: decode {r['decode']}", flush=True)
                continue
            vae = fp32_vae().to(dev)
            with torch.no_grad():
                frames = m.decode(vae, lat_cpu.to(dev))
            del vae
            r["decoded_with"] = "fp32 VAE"
        frames = to_uint8(frames)
        r["output_finite_nonblack"] = bool(frames.float().std() > 1.0)
        outputs[key][name] = frames
        del frames
        gc.collect(); torch.cuda.empty_cache()
        print(f"  {name:<6} {key:>12}: {json.dumps(r)}", flush=True)


def one_forward(pipe, capture, dtype):
    dev = pipe._execution_device
    inp = cast_inputs(capture["inputs"], dtype, dev)
    hs = inp.pop("hidden_states")
    inp.pop("return_dict", None)
    with torch.no_grad():
        out = pipe.transformer(hs, **inp, return_dict=False)[0]
    return out.float().cpu()


def strip(outputs, key, names, path, tile=256):
    """Middle frame of each variant side by side with a label, as a small PNG."""
    from PIL import Image, ImageDraw
    tiles = []
    for n in names:
        if n not in outputs.get(key, {}):
            continue
        fr = outputs[key][n]
        img = Image.fromarray(fr[fr.shape[0] // 2].numpy())
        img = img.resize((tile, round(tile * img.height / img.width)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (img.width, img.height + 18), "white")
        canvas.paste(img, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), n, fill="black")
        tiles.append(canvas)
    if not tiles:
        return
    out = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width + 4
    out.save(path, optimize=True)
    if os.path.getsize(path) > 200_000:
        out.quantize(256).save(path, optimize=True)
    print(f"  wrote {path} ({os.path.getsize(path) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="wan")
    ap.add_argument("--sizes", required=True, help="comma-separated; wan: FxHxW, sana: HxW")
    ap.add_argument("--ref-sizes", default=None, help="sizes for the fp32 reference (default: all)")
    ap.add_argument("--variants", default="fp32,bf16,tbf16,fp16")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", default=None, help="size whose middle frame goes into the comparison PNG")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    m = MODELS[a.model]()
    sizes = [m.parse_size(s) for s in a.sizes.split(",")]
    ref_sizes = sizes if a.ref_sizes is None else [m.parse_size(s) for s in a.ref_sizes.split(",")]
    variants = a.variants.split(",")
    out_dir = pathlib.Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(a.model, m.repo, torch.cuda.get_device_name(0), {"x".join(map(str, s)): m.tokens(s) for s in sizes}, "tokens")

    CACHE.mkdir(parents=True, exist_ok=True)
    ef = CACHE / f"{a.model}-{hashlib.sha1(m.prompt.encode()).hexdigest()[:10]}.pt"
    if not ef.exists():
        t = time.perf_counter()
        torch.save(m.embeds(), ef)
        print(f"  encoded prompt in {time.perf_counter() - t:.0f} s -> {ef}")
        gc.collect(); torch.cuda.empty_cache()
    embeds = torch.load(ef)

    results, outputs, fwd = {}, {}, {}
    capture = {"size": "x".join(map(str, ref_sizes[0])), "call": (a.steps // 2) * (2 if a.model == "wan" else 1)}

    def fp32_vae():  # fallback decoder when a variant's own VAE OOMs; loaded only then
        from diffusers import AutoencoderDC, AutoencoderKLWan
        cls = AutoencoderKLWan if a.model == "wan" else AutoencoderDC
        return cls.from_pretrained(m.repo, subfolder="vae", torch_dtype=torch.float32, **getattr(m, "load_kwargs", {}))

    def free(pipe):
        for c in pipe.components.values():
            if isinstance(c, torch.nn.Module):
                c.to("meta")
        gc.collect(); torch.cuda.empty_cache()

    if "fp32" in variants:
        ref_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        pipe = m.load(torch.float32, ref_dev)
        run_variant("fp32", m, pipe, embeds, ref_sizes, a, results, outputs, capture=capture)
        if "inputs" in capture:
            fwd["fp32"] = one_forward(pipe, capture, torch.float32)
        free(pipe)
    if "bf16" in variants or "tbf16" in variants:
        pipe = m.load(torch.bfloat16, "cuda:0")
        if "bf16" in variants:
            run_variant("bf16", m, pipe, embeds, sizes, a, results, outputs, fp32_vae=fp32_vae)
            if "inputs" in capture:
                fwd["bf16"] = one_forward(pipe, capture, torch.bfloat16)
        if "tbf16" in variants:
            tbf16.convert(pipe.transformer)
            tbf16.convert(pipe.vae)
            run_variant("tbf16", m, pipe, embeds, sizes, a, results, outputs, fp32_vae=fp32_vae)
            if "inputs" in capture:
                fwd["tbf16"] = one_forward(pipe, capture, torch.bfloat16)
            tbf16.uninstall_sdpa()
        free(pipe)
    if "fp16" in variants:
        pipe = m.load(torch.float16, "cuda:0")
        run_variant("fp16", m, pipe, embeds, sizes, a, results, outputs, fp32_vae=fp32_vae)
        if "inputs" in capture:
            fwd["fp16"] = one_forward(pipe, capture, torch.float16)
        free(pipe)

    print("\n== similarity of decoded outputs (PSNR dB / SSIM)")
    for key, outs in outputs.items():
        for name in variants:
            if name not in outs:
                continue
            r = results[key][name]
            for ref in ("fp32", "bf16"):
                if ref in outs and ref != name:
                    r[f"psnr_vs_{ref}"] = psnr(outs[name], outs[ref])
                    r[f"ssim_vs_{ref}"] = ssim(outs[name], outs[ref])
                    la, lr = outs[name + ":latents"], outs[ref + ":latents"]
                    r[f"latent_rel_err_vs_{ref}"] = rel_err(la, lr)
            print(f"  {key:>12} {name:<6}" + "".join(
                f"  vs {ref}: {r[f'psnr_vs_{ref}']:6.2f} dB / {r[f'ssim_vs_{ref}']:.4f}"
                for ref in ("fp32", "bf16") if f"psnr_vs_{ref}" in r))
    if "fp32" in fwd:
        print(f"\n== one transformer forward vs fp32 on identical inputs ({capture['size']}, call {capture['call']})")
        for n, y in fwd.items():
            if n != "fp32":
                e = rel_err(y, fwd["fp32"])
                results.setdefault("one_forward_rel_err_vs_fp32", {})[n] = e
                print(f"  {n:<6} {e:.3e}")

    print("\n== summary")
    print(f"  {'size':>12} {'variant':<7} {'s/step':>8} {'denoise':>9} {'decode':>8} {'peak GB':>8}")
    for key, rs in results.items():
        if key.startswith("one_forward"):
            continue
        for n, r in rs.items():
            st = r["denoise"] if "denoise" in r else f"{r['step_s']:.3f}"
            dn = "" if "denoise_s" not in r else f"{r['denoise_s']:.1f}"
            dc = r["decode"] + ("*" if r.get("decoded_with") else "") if "decode" in r else (
                f"{r['decode_s']:.1f}" if "decode_s" in r else "")
            pk = max(r.get("peak_denoise_gb", 0), r.get("peak_decode_gb", 0))
            print(f"  {key:>12} {n:<7} {st:>8} {dn:>9} {dc:>8} {pk:8.2f}")
    path = out_dir / f"{a.model}.json"
    json.dump(results, open(path, "w"), indent=1)
    print(f"  wrote {path}")
    if a.png:
        strip(outputs, a.png, [v for v in ("fp32", "bf16", "tbf16", "fp16") if v in variants],
              out_dir / f"{a.model}_{a.png}.png")
        for n, fr in outputs.get(a.png, {}).items():
            if ":" not in n:
                from PIL import Image
                Image.fromarray(fr[fr.shape[0] // 2].numpy()).save(out_dir / f"{a.model}_{a.png}_{n}_full.png")


if __name__ == "__main__":
    main()
