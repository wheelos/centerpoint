# Use a configurable base image. Override at build time if you need a different mirror or CUDA variant.
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

FROM ${BASE_IMAGE}

# Build context: project root. This Dockerfile is kept in /deploy to separate
# build/runtime concerns from application source code.
WORKDIR /app

# Replace default Ubuntu apt sources with the project-provided mirror list.
COPY deploy/sources.list /etc/apt/sources.list

# System deps required by inference runtime
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    curl \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps (inference_requirements.txt is in inference_service/)
COPY inference_service/inference_requirements.txt /tmp/inference_requirements.txt
# Install python deps; fail build if requirements are incomplete so issues are visible.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/inference_requirements.txt

# Copy inference service plus the repo assets it depends on at runtime.
COPY inference_service /app/inference_service
COPY apollo_centerpoint_trt /app/apollo_centerpoint_trt
COPY reference /app/reference
COPY mmdet3d_example_configs /app/mmdet3d_example_configs

WORKDIR /app/inference_service

EXPOSE 8010

# Default worker concurrency can be overridden via environment variable
ENV WEB_CONCURRENCY=1 \
    REPO_ROOT=/app

# Keep startup simple for now: run FastAPI directly under uvicorn.
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
