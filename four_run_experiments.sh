#!/usr/bin/env bash
# Four clients, one dataset each: MNIST, Fashion-MNIST, SVHN, CIFAR-10 (no EMNIST). MNIST and SVHN both
# have the digits 0-9; every other label has one holder (--min-holders 1).
# Same phases, settings and resume rules as run_experiments.sh. Output runs/four_<date>.
#   WARMUP=20 PHASES="1 4" EXTRA="--keep-frac 1.0" bash four_run_experiments.sh
#   SWEEP=1 bash four_run_experiments.sh        # instead: plain_sweep.py over the three candidate settings
#                                               # (JOBS, WORKERS, ROUNDS, OUT, GRID override; output runs/four_sweep)
cd "$(dirname "$0")"
export DATA="--num-train-mnist 1 --num-train-fashionmnist 1 --num-train-svhn 1 --num-train-cifar10 1 --num-train-emnist 0 --min-holders 1"
if [[ ${SWEEP:-0} == 1 ]]; then
    GRID=${GRID:-'[
 {"gen_label_batch":32, "--gen-widths":"64,32,16",  "--warmup-epochs":20, "global_samples_per_class":1024},
 {"gen_label_batch":64, "--gen-widths":"128,64,32", "--warmup-epochs":20, "global_samples_per_class":1024},
 {"gen_label_batch":32, "--gen-widths":"64,32,16",  "--warmup-epochs":60, "global_samples_per_class":1024}]'}
    exec ${PY:-python} plain_sweep.py --data="$DATA" --jobs "${JOBS:-3}" --workers "${WORKERS:-4}" \
        --rounds "${ROUNDS:-2}" --out "${OUT:-runs/four_sweep}" --grid "$GRID" ${SWEEP_ARGS:-}
fi
export OUT=${OUT:-runs/four}
exec bash run_experiments.sh
