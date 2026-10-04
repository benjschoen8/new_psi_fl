#!/usr/bin/env bash
# Heter version of run_experiments.sh: every client trains its own heterogeneous classifier on its real
# data (10 architectures by client id: MLP, CNN, ResNet8/18, MobileNetV2/V3, LeNet, AlexNet, ShuffleNetV2,
# SqueezeNet), freezes it, and its generator is trained with GAN loss + GUIDE_WEIGHT x classifier CE.
# Same phases, settings and resume rules as run_experiments.sh; output runs/heter_<YYYYMMDD-HHMM>.
#
#   GPUS="0 1" WARMUP=5 ROUNDS=50 PHASES="1 4" bash heter_run_experiments.sh
#   GUIDE_EPOCHS=20 GUIDE_WEIGHT=0.5 (defaults; --heter is also on by default in every run now)
cd "$(dirname "$0")"
export OUT=${OUT:-runs/heter}
export EXTRA="--heter --guide-epochs ${GUIDE_EPOCHS:-20} --guide-weight ${GUIDE_WEIGHT:-0.5} ${EXTRA:-}"
exec bash run_experiments.sh
