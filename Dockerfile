# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    RERUN_DISABLE_TELEMETRY=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates pkg-config build-essential \
    cmake ninja-build \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

ARG REPO_URL="https://github.com/Dhuldhoyavarun/mast3r-slam-rerun.git@cu126"
ARG REPO_REF="cu126"
WORKDIR /app
RUN git clone --depth=1 "${REPO_URL}" /app \
 && git submodule update --init --recursive || true \
 && git lfs pull || true

RUN pixi install --locked || pixi install

RUN pixi task list || true

EXPOSE 7860

CMD ["pixi", "run", "app"]
