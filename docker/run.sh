#!/usr/bin/env bash
# Run new_psi_fl in its container. From a git checkout (the usual way) it first builds the environment
# (NVIDIA's PyTorch image straight from NVIDIA, no login, ~10 GB the first time, + pip packages; later:
# cached, seconds) and mounts the checkout at /workspace: the code you run is the code in the folder (no
# rebuild after git pull or an edit), datasets and results stay in ./data and ./runs.
#
#   git clone https://github.com/benjschoen8/new_psi_fl.git && cd new_psi_fl    # or: git pull
#   docker/run.sh python -m tests.check_speedups
#   docker/run.sh bash run_experiments.sh
#   docker/run.sh env GPUS="0 1" ROUNDS=45 bash run_experiments.sh    # settings go after env (inside)
#   docker/run.sh                          # a shell inside (exit removes the container)
#   GPUS_VISIBLE=1 docker/run.sh ...       # only GPU 1 (default: all)
#   BASE=nvcr.io/nvidia/pytorch:25.06-py3 docker/run.sh ...    # older NVIDIA driver (< 580)
#   MPS=1 docker/run.sh ...                # NVIDIA MPS: the client processes' kernels share the GPU at the
#                                          # same time instead of taking turns (same results, less VRAM each)
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
    build=(docker build -f docker/Dockerfile --target env -t "$IMAGE" ${BASE:+--build-arg BASE="$BASE"} .)
    docker image inspect "$IMAGE" >/dev/null 2>&1 ||
        echo "first build: downloads NVIDIA's PyTorch image (~10 GB), then installs the packages" >&2
    echo "building $IMAGE (cached: a few seconds)..." >&2
    mounts=(-v "$PWD":/workspace)                            # the whole checkout (data/ and runs/ in it)
    "${build[@]}" >&2 || exit                                # shown: a slow step must not look like a hang
    echo "starting the container..." >&2
fi
IMAGE=${IMAGE:-ghcr.io/benjschoen8/new_psi_fl:cuda}
[[ $IMAGE == */* ]] && pull=always || pull=never             # a local image has no registry to ask
NAME=${NAME:-new_psi_fl_$(date +%H%M%S)_$$}                  # unique: no "name already in use"
tty=(); [[ -t 0 ]] && tty=(-it)
gpus=all; [[ -n ${GPUS_VISIBLE:-} ]] && gpus="\"device=$GPUS_VISIBLE\""   # e.g. GPUS_VISIBLE=0,1
cmd=("${@:-bash}") env=()
if [[ ${MPS:-0} == 1 ]]; then                                # MPS daemon inside the container: no host setup
    env=(-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps)
    cmd=(bash -c 'mkdir -p /tmp/mps && nvidia-cuda-mps-control -d && echo "MPS on" >&2 ||
                  echo "MPS could not start: running without it" >&2; exec "$@"' mps "${cmd[@]}")
fi
# files written in data/ and runs/ must be yours: normal Docker runs as your uid; rootless Docker already
# maps the container's root to you (your uid there would map to a subuid that cannot write your folders)
user=(--user "$(id -u):$(id -g)")
[[ $(docker info -f '{{.SecurityOptions}}' 2>/dev/null) == *rootless* ]] && user=()
mkdir -p data runs
[[ -n ${mounts+x} ]] || mounts=(-v "$PWD/data":/workspace/data -v "$PWD/runs":/workspace/runs)   # pushed image
exec docker run --rm ${tty[@]+"${tty[@]}"} --pull "${PULL:-$pull}" --name "$NAME" \
    --gpus "$gpus" \
    --ipc=host \
    ${user[@]+"${user[@]}"} -e HOME=/tmp ${env[@]+"${env[@]}"} \
    "${mounts[@]}" -w /workspace \
    "$IMAGE" "${cmd[@]}"
