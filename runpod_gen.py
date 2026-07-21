#!/usr/bin/env python3
"""
AURA → RunPod Flux2 (Uncensored) Generator
Panggil RunPod Serverless Endpoint untuk generate gambar tanpa sensor.

Usage:
    python3 runpod_gen.py "prompt" --output /path/output.png
    python3 runpod_gen.py "prompt" --width 768 --height 512 --steps 20

Env vars:
    RUNPOD_API_KEY   — RunPod API key
    RUNPOD_ENDPOINT  — Serverless endpoint ID (auto-set after deploy)
"""

import os
import sys
import json
import time
import base64
import argparse
import requests

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5
TIMEOUT = 120  # detik


def generate(prompt, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
             steps=DEFAULT_STEPS, guidance=DEFAULT_GUIDANCE, seed=None,
             output=None):
    """
    Generate gambar via RunPod Serverless.

    Returns dict: {image_b64, width, height, gen_time_ms}
    Simpan ke file jika --output diberikan.
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint = os.environ.get("RUNPOD_ENDPOINT")

    if not api_key:
        print("❌ RUNPOD_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not endpoint:
        print("❌ RUNPOD_ENDPOINT not set", file=sys.stderr)
        sys.exit(1)

    url = f"https://api.runpod.ai/v2/{endpoint}/runsync"

    payload = {
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
    }
    if seed is not None:
        payload["input"]["seed"] = int(seed)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"🎨 Generating: {width}x{height}, {steps} steps...", file=sys.stderr)
    t0 = time.time()

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after {TIMEOUT}s", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    # RunPod returns status "COMPLETED" or "FAILED"
    status = data.get("status")
    if status != "COMPLETED":
        error_msg = data.get("error", data.get("output", "Unknown error"))
        print(f"❌ RunPod error [{status}]: {error_msg}", file=sys.stderr)
        sys.exit(1)

    output_data = data.get("output", {})
    gen_ms = int((time.time() - t0) * 1000)

    # Save image
    img_b64 = output_data.get("image")
    if not img_b64:
        print("❌ No image in response", file=sys.stderr)
        sys.exit(1)

    img_bytes = base64.b64decode(img_b64)

    if output:
        out_path = output
    else:
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        out_path = f"flux2_{slug}.png"

    with open(out_path, "wb") as f:
        f.write(img_bytes)

    size_kb = len(img_bytes) // 1024
    server_ms = output_data.get("gen_time_ms", "?")
    print(f"✅ {out_path}  ({size_kb}KB, gen {server_ms}ms, total {gen_ms}ms)")

    return {
        "path": out_path,
        "size_kb": size_kb,
        "gen_time_ms": server_ms,
        "total_ms": gen_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA RunPod Flux2 Generator")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    generate(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        output=args.output,
    )
