#!/usr/bin/env bash
# EMNIST-only experiment: 10 clients, each holding 8..20 of the 62 EMNIST classes (drawn at random, least
# covered classes first, so every class has 2-3 holders; each class's images split evenly among them).
# A client knows only its own classes. Same phases, settings and resume rules as run_experiments.sh;
# output runs/emnist10_<YYYYMMDD-HHMM>.
#
#   GPUS="0 1" WARMUP=5 ROUNDS=50 PHASES="1 4" EXTRA="--keep-frac 1.0" bash emnist10_run_experiments.sh
#   CLASSES=8,20 (default)
cd "$(dirname "$0")"
export OUT=${OUT:-runs/emnist10}
export DATA="--num-train-mnist 0 --num-train-cifar10 0 --num-train-emnist 10 --class-subsets ${CLASSES:-8,20}"
exec bash run_experiments.sh
