CenterPoint deployment layout and responsibilities

Overview:
- `inference_service/` : application source code (FastAPI app, model loading, handlers). No build files here.
- `deploy/` : deployment artifacts (Dockerfile, docker-compose.yml, healthchecks). Responsible for building and running the service image.

How to build & run (local machine with NVIDIA drivers):

1) Build & run with docker compose (recommended):

```bash
cd /home/wfh/01code/centerpoint/deploy
docker compose up -d --build
```

2) Health check:

```bash
curl -f http://localhost:8010/v1/health/ready
```

Development notes:
- To make code changes visible without rebuilding, for development you may mount the source into the container by uncommenting the volume line in `docker-compose.yml`.
- Keep runtime/config in `deploy/`; keep all code changes in `inference_service/`.
Notes:
- The container exposes the service on port `8010` (Gunicorn). When running `main.py` locally for development, the built-in server defaults to `PORT=8000` unless overridden.
- If you want GPU-accelerated ONNX Runtime, install `onnxruntime-gpu` (or build an image that includes it) and set the environment variable `ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider` in the compose file or at runtime.
- For local docker-compose GPU access, `device_requests` is used in `docker-compose.yml`. If your environment requires a different approach, adjust accordingly.
