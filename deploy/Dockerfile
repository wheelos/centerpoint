# Use a configurable base image so we can build from a nearby mirror in China
# Default is set to an Alibaba Cloud mirror for faster pulls in CN regions.
# You can override at build time: `docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime .`
ARG BASE_IMAGE=registry.cn-hangzhou.aliyuncs.com/pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

# Build context: project root. This Dockerfile is kept in /deploy to separate
# build/runtime concerns from application source code.
WORKDIR /app

# System deps required by inference runtime
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps (inference_requirements.txt is in inference_service/)
COPY inference_service/inference_requirements.txt /tmp/inference_requirements.txt
# Install python deps; fail build if requirements are incomplete so issues are visible.
RUN pip install --no-cache-dir -r /tmp/inference_requirements.txt

# Copy app source into image (code lives in inference_service/)
COPY inference_service /app

EXPOSE 8010

# Default worker concurrency can be overridden via environment variable
ENV WEB_CONCURRENCY=1

# Use Gunicorn with Uvicorn workers for production-grade serving
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8010", "--timeout", "120"]
