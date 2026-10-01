# new_psi_fl environment for NVIDIA GPUs: RTX 5090 / 3060 (x86-64) and GX10 / DGX Spark (GB10, arm64).
# Datasets and results stay on the machine (./data, ./runs). Usual way: docker/run.sh in a checkout builds the env stage on the machine
# itself (NVIDIA's part straight from NVIDIA, no registry of ours) and mounts the checkout. Optional, a
# pushed image with the code inside, for machines without a checkout (both CPU types):
#
#   docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile \
#       --annotation "index:org.opencontainers.image.source=https://github.com/benjschoen8/new_psi_fl" \
#       -t ghcr.io/benjschoen8/new_psi_fl:cuda --push .
#
# The NVIDIA PyTorch image ships CUDA, cuDNN and a PyTorch built for current GPUs, for both x86-64 and
# arm64 (host driver >= 580 for 25.09; older driver: --build-arg BASE=nvcr.io/nvidia/pytorch:25.06-py3).
ARG BASE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE} AS env

LABEL org.opencontainers.image.source=https://github.com/benjschoen8/new_psi_fl \
      org.opencontainers.image.description="new_psi_fl environment; docker/run.sh mounts the checkout at /workspace"

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
CMD ["bash"]

# Only for a pushed image (machines without a checkout): the code baked in. docker/run.sh in a checkout
# builds --target env and mounts the checkout instead, so code changes and old results never enter a
# layer. Readable by any --user (chmod in a RUN, not COPY --chmod: works without BuildKit too).
FROM env AS code
COPY . /workspace
RUN chmod -R a+rX /workspace
