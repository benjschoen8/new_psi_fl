# Runtime for new_psi_fl on NVIDIA GPUs: RTX 5090 (Blackwell, x86-64) and GX10 / DGX Spark (GB10, arm64).
# Only the libraries live in the image; the code, datasets and runs are mounted from the host
# (docker/run.sh), so a `git pull` on the host is what the container runs.
#
#   docker build -f docker/Dockerfile -t new_psi_fl:cuda .            # from the repo root
#   docker build -f docker/Dockerfile --build-arg BASE=nvcr.io/nvidia/pytorch:<newer tag> -t new_psi_fl:cuda .
# Both CPU types at once, pushed to GitHub's registry (private by default):
#   docker buildx create --use --name multi   # once
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile \
#       -t ghcr.io/benjschoen8/new_psi_fl:cuda --push .
#   then on a GPU machine: docker pull ghcr.io/benjschoen8/new_psi_fl:cuda
#                          IMAGE=ghcr.io/benjschoen8/new_psi_fl:cuda docker/run.sh
#
# The NVIDIA PyTorch image ships CUDA, cuDNN and a PyTorch built for current GPUs, for both x86-64 and
# arm64. Blackwell needs CUDA >= 12.8 (a 25.xx tag or newer; the GX10's GB10 wants a recent one):
# pick the newest tag from catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.
ARG BASE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE}

LABEL org.opencontainers.image.source=https://github.com/benjschoen8/new_psi_fl \
      org.opencontainers.image.description="new_psi_fl runtime (NVIDIA PyTorch + deps); code is mounted at /workspace"

ARG WITH_ENCODER=0          # 1: also sentence-transformers (only for tests/fuzzy_threshold.py calibration)

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/workspace/data/hf

# keep the image's own torch / torchvision / numpy: never let pip replace them
RUN pip freeze | grep -iE '^(torch|torchvision|numpy)[= @]' > /tmp/keep.txt; \
    pip install -c /tmp/keep.txt omegaconf scikit-learn matplotlib tqdm cryptography pandas seaborn && \
    if [ "$WITH_ENCODER" = "1" ]; then pip install -c /tmp/keep.txt sentence-transformers; fi && \
    python -c "import torch, torchvision, omegaconf, sklearn, matplotlib, cryptography; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /workspace
CMD ["bash"]
