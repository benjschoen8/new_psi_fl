#!/usr/bin/env bash
# Start the container with the GPUs and this repo mounted at /workspace (code, data/, runs/ stay on the host).
#
#   docker/run.sh                          # interactive shell in /workspace
#   docker/run.sh python -m tests.check_speedups
#   docker/run.sh bash run_experiments.sh  # e.g. with PY="python -u" GPUS="0 1" ... in front, see below
#   GPUS_VISIBLE=1 docker/run.sh ...       # only GPU 1 (default: all)
#
# Environment variables for run_experiments.sh can be passed with -e, or just set inside the shell.
# Run it inside tmux on the host (tmux new -s exp; docker/run.sh; ...): detach with Ctrl-b d, the run goes on.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-new_psi_fl:cuda}
NAME=${NAME:-new_psi_fl_$(date +%H%M%S)}           # unique: no "name already in use"
tty=(); [[ -t 0 ]] && tty=(-it)
gpus=all; [[ -n ${GPUS_VISIBLE:-} ]] && gpus="\"device=$GPUS_VISIBLE\""   # e.g. GPUS_VISIBLE=0,1
exec docker run --rm ${tty[@]+"${tty[@]}"} --name "$NAME" \
    --gpus "$gpus" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/workspace -w /workspace \
    "$IMAGE" "${@:-bash}"
