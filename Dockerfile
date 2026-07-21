# RunPod Serverless Worker — Flux2 Klein 9B Uncensored
# Model: ponpoke/flux2-klein-9b-uncensored-text-encoder

FROM runpod/base:0.6.0-cuda12.2.0

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (torch/accelerate/safetensors already in base image)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --default-timeout=120 -r /tmp/requirements.txt

# Copy worker
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
