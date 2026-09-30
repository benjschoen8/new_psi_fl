#!/usr/bin/env bash
# Run new_psi_fl from its image (the code is inside). Pulls the image (newest version) and keeps datasets and
# results in the folder you run it from: ./data and ./runs (created if missing). No clone needed:
#
#   docker login ghcr.io -u benjschoen8                  # once per machine (token with read:packages)
#   docker run --rm ghcr.io/benjschoen8/new_psi_fl:cuda cat docker/run.sh > psi.sh && chmod +x psi.sh
#   ./psi.sh python -m tests.check_speedups
#   ./psi.sh bash run_experiments.sh                     # results in ./runs
#   ./psi.sh env GPUS="0 1" ROUNDS=45 bash run_experiments.sh    # settings go after env (inside the container)
#   ./psi.sh bash emnist10_run_experiments.sh
#   ./psi.sh                                             # a shell inside (exit removes the container)
#
#   GPUS_VISIBLE=1 ./psi.sh ...     only GPU 1 (default: all)
#   PULL=never ./psi.sh ...         offline: use the image already here (default: check for a newer one)
#   DEV=1 docker/run.sh ...         from a git checkout: run the checkout's code instead of the image's
#   IMAGE=new_psi_fl:cuda ...       a locally built image (docker build -f docker/Dockerfile -t new_psi_fl:cuda .)
# Long runs: inside tmux (tmux new -s exp; ./psi.sh ...; detach Ctrl-b d, back with tmux attach -t exp).
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/benjschoen8/new_psi_fl:cuda}
[[ $IMAGE == */* ]] && pull=always || pull=never             # a local image name has no registry to ask
NAME=${NAME:-new_psi_fl_$(date +%H%M%S)_$$}                  # unique: no "name already in use"
tty=(); [[ -t 0 ]] && tty=(-it)
gpus=all; [[ -n ${GPUS_VISIBLE:-} ]] && gpus="\"device=$GPUS_VISIBLE\""   # e.g. GPUS_VISIBLE=0,1
if [[ -n ${DEV:-} ]]; then
    mounts=(-v "$PWD":/workspace)                           # code, data and runs from this checkout
else
    mkdir -p data runs
    mounts=(-v "$PWD/data":/workspace/data -v "$PWD/runs":/workspace/runs)
fi
exec docker run --rm ${tty[@]+"${tty[@]}"} --pull "${PULL:-$pull}" --name "$NAME" \
    --gpus "$gpus" \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    "${mounts[@]}" -w /workspace \
    "$IMAGE" "${@:-bash}"
