#!/usr/bin/env python3
"""
RunPod Serverless Worker — Juggernaut XL v9
Model: RunDiffusion/Juggernaut-XL-v9
VAE: madebyollin/sdxl-vae-fp16-fix
"""
import io, base64, time, traceback, runpod, torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKL

_model_id = "RunDiffusion/Juggernaut-XL-v9"
_vae_id = "madebyollin/sdxl-vae-fp16-fix"
_pipe = None

DEFAULT_NEGATIVE = (
    "ugly, deformed, blurry, low quality, bad anatomy, bad hands, "
    "missing fingers, extra fingers, fused fingers, mutated hands, "
    "poorly drawn face, bad proportions, gross proportions, "
    "malformed limbs, mutated, disfigured, watermark, text, signature, "
    "username, artist name, jpeg artifacts, compression artifacts, "
    "cartoon, anime, 3D render, CGI, plastic skin, flat texture"
)

def load_model():
    global _pipe
    print(f"[WORKER] Loading {_model_id}...", flush=True)
    t0 = time.time()
    try:
        vae = AutoencoderKL.from_pretrained(_vae_id, torch_dtype=torch.float16)
        _pipe = StableDiffusionXLPipeline.from_pretrained(
            _model_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
        )
        _pipe.to("cuda")
        print(f"[WORKER] Loaded in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"[WORKER] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise

def generate_image(job):
    global _pipe
    if _pipe is None:
        return {"error": "model not loaded"}
    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    negative = inp.get("negative_prompt", DEFAULT_NEGATIVE)
    width = inp.get("width", 768)
    height = inp.get("height", 1024)
    steps = inp.get("num_inference_steps", 25)
    guidance = inp.get("guidance_scale", 6.0)
    seed = inp.get("seed")
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed else None

    print(f"[WORKER] {prompt[:80]}... ({width}x{height}, {steps}s, CFG={guidance})", flush=True)
    t0 = time.time()
    result = _pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width, height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    gen_ms = int((time.time() - t0) * 1000)
    print(f"[WORKER] Done {gen_ms}ms", flush=True)
    return {"image": img_b64, "format": "png", "width": width, "height": height, "gen_time_ms": gen_ms}

if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": generate_image})
