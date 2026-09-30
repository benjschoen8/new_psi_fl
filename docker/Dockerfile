# new_psi_fl with its code, for NVIDIA GPUs: RTX 5090 / 3060 (x86-64) and GX10 / DGX Spark (GB10, arm64).
# The code is in the image (/workspace); datasets and results stay on the machine (./data, ./runs,
# mounted by docker/run.sh). Build and push from the repo root (both CPU types; after a code change only
# the small code layer is rebuilt and uploaded):
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile \
#       --annotation "index:org.opencontainers.image.source=https://github.com/benjschoen8/new_psi_fl" \
#       -t ghcr.io/benjschoen8/new_psi_fl:cuda --push .
#
# On a GPU machine, no clone needed: see docker/run.sh.
#
# The NVIDIA PyTorch image ships CUDA, cuDNN and a PyTorch built for current GPUs, for both x86-64 and
# arm64 (host driver >= 580 for 25.09; older driver: --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3).
ARG BASE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE}

LABEL org.opencontainers.image.source=https://github.com/benjschoen8/new_psi_fl \
      org.opencontainers.image.description="new_psi_fl: code in /workspace; ./data and ./runs mounted by docker/run.sh"

# 1: also sentence-transformers (only for tests/fuzzy_threshold.py calibration)
ARG WITH_ENCODER=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/workspace/data/hf

# keep the image's own torch / torchvision / numpy: never let pip replace them
RUN pip freeze | grep -iE '^(torch|torchvision|numpy)[= @]' > /tmp/keep.txt; \
    pip install -c /tmp/keep.txt omegaconf scikit-learn matplotlib tqdm cryptography pandas seaborn && \
    if [ "$WITH_ENCODER" = "1" ]; then pip install -c /tmp/keep.txt sentence-transformers; fi && \
    python -c "import torch, torchvision, omegaconf, sklearn, matplotlib, cryptography; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /workspace
# last, so a code change rebuilds (and uploads) only this layer; readable by any --user
COPY --chmod=755 . /workspace
CMD ["bash"]
