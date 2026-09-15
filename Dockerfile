# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    CAUSAL_LM_MODEL=/models/qwen \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu

# Bake the public Qwen checkpoint into the image so Cloud Run never downloads
# model files during a request. The token is provided as a BuildKit secret and
# is not written into an image layer.
RUN --mount=type=secret,id=HF_TOKEN \
    sh -c 'HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"Qwen/Qwen3-0.6B-Base\", local_dir=\"/models/qwen\", token=True)"'

COPY app.py functions.py baseline_full.json basicness_detector_model.pkl ./

CMD exec gunicorn --bind :${PORT} --workers 1 --threads 4 --timeout 600 app:app
