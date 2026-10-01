#!/usr/bin/env bash
# EMNIST-only experiment: 10 clients over the 62 EMNIST classes; a client knows only its own classes.
#   CLASSES=even (default)  every class held by exactly 2 clients; every client gets 12-13 classes and
#                           about the same number of images (~70k), so no client holds up the round
#   CLASSES=8,20            every client draws 8..20 classes at random (least covered first, 2-3 holders)
# Each class's images are split evenly among its holders. Same phases, settings and resume rules as
# run_experiments.sh.
#   GEN=perlabel (default)  one whole DCGAN generator per label (as perlabel_run_experiments.sh);
#                           output runs/emnist10_<CLASSES>_perlabel
#   GEN=cbn                 shared trunk + CBN rows; output runs/emnist10_<CLASSES>
#                           (earlier 8,20 runs: CLASSES=8,20 GEN=cbn OUT=runs/emnist10)
#
#   GPUS="0 1" WARMUP=5 ROUNDS=50 PHASES="1 4" EXTRA="--keep-frac 1.0" bash emnist10_run_experiments.sh
cd "$(dirname "$0")"
CLASSES=${CLASSES:-even}
GEN=${GEN:-perlabel}
if [[ $GEN == perlabel ]]; then
    export OUT=${OUT:-runs/emnist10_${CLASSES/,/-}_perlabel}
    export EXTRA="--per-label-gen --gen-widths ${GEN_WIDTHS:-64,32,16} ${EXTRA:-}"
else
    export OUT=${OUT:-runs/emnist10_${CLASSES/,/-}}
fi
export DATA="--num-train-mnist 0 --num-train-cifar10 0 --num-train-emnist 10 --class-subsets $CLASSES"
exec bash run_experiments.sh
