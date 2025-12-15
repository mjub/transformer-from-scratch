ARG RUNPOD_VERSION="2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
FROM runpod/pytorch:${RUNPOD_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends nano tmux && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip uv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache-dir --system -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
