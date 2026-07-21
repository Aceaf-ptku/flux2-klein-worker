# RunPod Serverless Worker — Flux2 Klein 9B Uncensored
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Python deps (torch CUDA — versi kecil dari index)
RUN pip install --no-cache-dir --default-timeout=120 \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --default-timeout=120 \
    diffusers>=0.31.0 transformers>=4.45.0 \
    accelerate>=0.34.0 safetensors>=0.4.0 \
    pillow>=10.0.0 huggingface_hub>=0.25.0 \
    sentencepiece>=0.2.0 runpod>=1.7.0

# Copy worker
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
