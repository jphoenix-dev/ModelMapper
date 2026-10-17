# Worker image for the distributed neuron map (src/dist). Slim: base runtime image + a few pip packages + src/.
# No weights, data or credentials inside: the pinned model downloads at boot; keys arrive as env vars.
# Build:  docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t <registry>/model-mapper:<tag> .
# Local test: docker run --gpus all --env-file .env -e DIST_STORE=s3://<bucket>/<run> <image> --no-terminate
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0 MODEL_MAPPER_MODELS=/nonexistent
WORKDIR /app
COPY requirements-dist.txt .
RUN pip install --no-cache-dir -r requirements-dist.txt
COPY src/config.py src/neurons.py src/
COPY src/dist/ src/dist/
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}
ENTRYPOINT ["python", "src/dist/worker.py"]
