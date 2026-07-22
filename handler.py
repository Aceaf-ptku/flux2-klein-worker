#!/usr/bin/env python3
"""
RunPod Serverless Worker — Realistic Vision V5.1 (Uncensored)
Model: SG161222/Realistic_Vision_V5.1_noVAE

Menerima prompt → generate gambar → return base64 + metadata.
"""
import io
import base64
import time
import runpod

_pipe = None
_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

def load_model():
    global _pipe
    print(f"[WORKER] Loading {_model_id}...", flush=True)
    t0 = time.time()

    from diffusers import StableDiffusionPipeline
    import torch

    _pipe = StableDiffusionPipeline.from_pretrained(
        _model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
    )
    _pipe.to("cuda")
    _pipe.enable_attention_slicing()

    print(f"[WORKER] Loaded in {time.time()-t0:.1f}s", flush=True)

def generate_image(job):
    global _pipe
    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    width = inp.get("width", 512)
    height = inp.get("height", 512)
    steps = inp.get("num_inference_steps", 25)
    guidance = inp.get("guidance_scale", 7.5)
    seed = inp.get("seed")

    if not prompt:
        return {"error": "prompt is required"}

    import torch
    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed else None

    print(f"[WORKER] {prompt[:50]}... {width}x{height}", flush=True)
    t0 = time.time()

    result = _pipe(prompt=prompt, width=width, height=height,
                   num_inference_steps=steps, guidance_scale=guidance,
                   generator=generator)
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
