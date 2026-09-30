#!/usr/bin/env bash
# Run new_psi_fl in its container. From a git checkout (the usual way) it first builds the image from this
# checkout: NVIDIA's PyTorch image comes straight from NVIDIA (no login, ~10 GB the first time), your code
# is copied in; later builds take seconds unless the code or the Dockerfile changed. Datasets and results
# stay in the checkout: ./data and ./runs.
#
#   git clone https://github.com/benjschoen8/new_psi_fl.git && cd new_psi_fl    # or: git pull
#   docker/run.sh python -m tests.check_speedups
#   docker/run.sh bash run_experiments.sh
#   docker/run.sh env GPUS="0 1" ROUNDS=45 bash run_experiments.sh    # settings go after env (inside)
#   docker/run.sh                          # a shell inside (exit removes the container)
#   GPUS_VISIBLE=1 docker/run.sh ...       # only GPU 1 (default: all)
#   BASE=nvcr.io/nvidia/pytorch:25.06-py3 docker/run.sh ...    # older NVIDIA driver (< 580)
# Long runs: inside tmux (tmux new -s exp; docker/run.sh ...; detach Ctrl-b d, back: tmux attach -t exp).
#
# Without a checkout (a copy of this script alone), it runs a pushed image instead, with ./data and
# ./runs in the current folder: IMAGE=ghcr.io/benjschoen8/new_psi_fl:cuda ./run.sh ... (default image;
# docker login ghcr.io first if it is private; PULL=never to skip the check for a newer one).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [[ -z ${IMAGE:-} && -f $here/Dockerfile ]]; then         # in a checkout: build from it
    cd "$here/.."
    IMAGE=new_psi_fl:cuda
    build=(docker build -f docker/Dockerfile -t "$IMAGE" ${BASE:+--build-arg BASE="$BASE"} .)
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        "${build[@]}" -q >/dev/null                         # cached: only a changed code layer is redone
    else
        echo "first build: downloads NVIDIA's PyTorch image (~10 GB), then installs the packages" >&2
        "${build[@]}"
    fi
fi
IMAGE=${IMAGE:-ghcr.io/benjschoen8/new_psi_fl:cuda}
[[ $IMAGE == */* ]] && pull=always || pull=never             # a local image has no registry to ask
NAME=${NAME:-new_psi_fl_$(date +%H%M%S)_$$}                  # unique: no "name already in use"
tty=(); [[ -t 0 ]] && tty=(-it)
gpus=all; [[ -n ${GPUS_VISIBLE:-} ]] && gpus="\"device=$GPUS_VISIBLE\""   # e.g. GPUS_VISIBLE=0,1
mkdir -p data runs
exec docker run --rm ${tty[@]+"${tty[@]}"} --pull "${PULL:-$pull}" --name "$NAME" \
    --gpus "$gpus" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD/data":/workspace/data -v "$PWD/runs":/workspace/runs -w /workspace \
    "$IMAGE" "${@:-bash}"
