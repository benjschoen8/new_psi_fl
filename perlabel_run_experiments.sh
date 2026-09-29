#!/usr/bin/env bash
# Per-label version of run_experiments.sh: one whole DCGAN generator per label (no shared trunk, no CBN).
# Row k = every weight of label k's generator: aggregated over k's holders (SecAgg), sent by KEM to them.
# Same phases, settings and resume rules as run_experiments.sh; output runs/perlabel_<YYYYMMDD-HHMM>.
#
#   GPUS="0 1" WARMUP=5 ROUNDS=50 PHASES="1 4" EXTRA="--keep-frac 1.0" bash perlabel_run_experiments.sh
#   GEN_WIDTHS=64,32,16 (default; each label ~173k params, upload ~10x the CBN version)
cd "$(dirname "$0")"
export OUT=${OUT:-runs/perlabel}
export EXTRA="--per-label-gen --gen-widths ${GEN_WIDTHS:-64,32,16} ${EXTRA:-}"
exec bash run_experiments.sh
