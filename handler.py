#!/usr/bin/env python3
"""
RunPod Serverless Worker — Flux2 Klein 9B Uncensored
Model: ponpoke/flux2-klein-9b-uncensored-text-encoder

Menerima prompt → generate gambar → return base64 + metadata.
"""

import io
import base64
import time
import traceback
import runpod

# Global model — loaded once, reused across requests
_pipe = None
_model_id = "ponpoke/flux2-klein-9b-uncensored-text-encoder"


def load_model():
    """Load Flux2 Klein pipeline. Dipanggil sekali saat container start."""
    global _pipe

    print(f"[WORKER] Loading {_model_id}...", flush=True)
    t0 = time.time()

    from diffusers import FluxPipeline
    import torch

    _pipe = FluxPipeline.from_pretrained(
        _model_id,
        torch_dtype=torch.bfloat16,          # efficient on modern GPUs
        use_safetensors=True,
    )
    _pipe.to("cuda")
    # Optional: enable memory-efficient attention
    try:
        _pipe.enable_attention_slicing()
        print("[WORKER] Attention slicing enabled", flush=True)
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"[WORKER] Model loaded in {elapsed:.1f}s", flush=True)


def generate_image(job):
    """
    Generate gambar dari prompt.

    Input job (dari RunPod):
        {
            "input": {
                "prompt": "...",
                "width": 1024,        # optional, default 1024
                "height": 1024,       # optional, default 1024
                "num_inference_steps": 28,  # optional, default 28
                "guidance_scale": 3.5,      # optional, default 3.5
                "seed": null                  # optional, null = random
            }
        }

    Returns:
        {
            "image": "<base64>",
            "format": "png",
            "width": 1024,
            "height": 1024,
            "gen_time_ms": 1234
        }
    """
    global _pipe

    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    steps = inp.get("num_inference_steps", 28)
    guidance = inp.get("guidance_scale", 3.5)
    seed = inp.get("seed")

    if not prompt:
        return {"error": "prompt is required"}

    # Set seed if provided
    import torch
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    else:
        generator = None

    print(f"[WORKER] Generating: {width}x{height}, {steps} steps, seed={seed}", flush=True)
    t0 = time.time()

    result = _pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    image = result.images[0]

    gen_ms = int((time.time() - t0) * 1000)

    # Encode to PNG base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    print(f"[WORKER] Done in {gen_ms}ms, {len(img_b64)//1024}KB base64", flush=True)

    return {
        "image": img_b64,
        "format": "png",
        "width": width,
        "height": height,
        "gen_time_ms": gen_ms,
    }


# ─── RunPod handler ──────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": generate_image})
