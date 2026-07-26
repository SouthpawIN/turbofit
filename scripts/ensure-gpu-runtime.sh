#!/usr/bin/env bash
# Ensure Docker can expose NVIDIA GPUs. Safe to run repeatedly.
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required before enabling GPU containers." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU detected; local GPU containers disabled (API-only mode)."
  exit 0
fi

if docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L >/dev/null 2>&1; then
  echo "Docker NVIDIA GPU runtime: ready"
  exit 0
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required to install NVIDIA Container Toolkit." >&2
  exit 3
fi

sudo install -d -m 0755 /usr/share/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | python3 -c 'import sys; print("".join(line.replace("deb https://", "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://") for line in sys.stdin if line.strip() and not line.startswith("#")), end="")' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi -L
echo "Docker NVIDIA GPU runtime: configured"
