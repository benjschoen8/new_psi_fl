#!/usr/bin/env bash
# Scale experiments, five datasets (MNIST, EMNIST, CIFAR-10, Fashion-MNIST, STL-10), methods plain and
# similar (= fuzzy) only. Output: <OUT>_<YYYYMMDD-HHMM>/{setup,train,comm}.
#
#   setup  setup only (no training), clients 7 10 30 50 (totals, 2 special included). Default (SETUP_MPC=model): grouping as its ideal
#          functionality, MPC traffic / rounds / compute time from the fitted cost model of the same circuits
#          (label_union/mpc_model.py), no MP-SPDZ needed. SETUP_MPC=real: real MP-SPDZ, measured. Both: union /
#          SecAgg bytes, modelled communication time (NET_MBPS, NET_RTT_MS), pair MCC -> setup/setup.{json,csv}
#   train  training with accuracy, 30 clients (28 + 2 special) over 5 datasets, NO MPC (grouping = ideal functionality):
#          run_experiments.sh phase 1 + 4 with METHODS="plain similar"  -> train/<run>/metrics.jsonl
#   comm   per-round communication without training (--comm-only): COMM_ROUNDS simulated rounds,
#          averaged over rounds 2.. (round 1 has no downlink); 3 clients / 3 datasets and 30 clients /
#          5 datasets, plain and similar, NO MPC  -> comm/<cfg>_<method>/comm_summary.json + comm/summary.txt
#
#   bash scale_experiments.sh                       # all three parts
#   PARTS="comm" bash scale_experiments.sh          # one part
#   DEVICE=cuda GPUS="0 1" SEQ=1 WORKERS=8 PARTS=train bash scale_experiments.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
PARTS=${PARTS:-setup train comm}
OUT=${OUT:-runs/scale}_$(date +%Y%m%d-%H%M)
DEVICE=${DEVICE:-cuda}
FIVE=MNIST,EMNIST,CIFAR10,FashionMNIST,STL10
# main split, 30 clients: 28 normal (6 MNIST, 6 EMNIST, 6 Fashion-MNIST, 5 CIFAR-10, 5 STL-10), each with 5-6
# classes of one dataset and ALL their images (labels split, data not), plus 2 special clients holding
# CIFAR-10 + STL-10 merged per class name. Most labels have one holder (--min-holders 1)
DATA5="--num-train-mnist 6 --num-train-emnist 6 --num-train-fashionmnist 6 --num-train-cifar10 5 --num-train-stl10 5 \
--class-subsets ${CLASS_SUBSETS:-5,6} --class-share full --num-train-cifar10stl10 ${MIXED:-2} --min-holders 1"
DATA3="--num-train-mnist 1 --num-train-emnist 1 --num-train-cifar10 1 --min-holders 1"
mkdir -p "$OUT"
echo "output folder: $OUT"
has() { [[ " $PARTS " == *" $1 "* ]]; }
fails=0

if has setup; then                                   # the only part with MPC (real or modelled)
    echo "[$(date '+%F %T')] setup: clients ${SETUP_CLIENTS:-7 10 30 50}"
    # shellcheck disable=SC2086
    model=(--mpc-model); [[ ${SETUP_MPC:-model} == real ]] && model=()
    $PY -m setup_smoke_hybrid "${model[@]}" --datasets "$FIVE" --clients ${SETUP_CLIENTS:-7 10 30 50} \
        --class-subsets "${CLASS_SUBSETS:-5,6}" --class-share full --num-train-cifar10stl10 "${MIXED:-2}" \
        --methods plain similar --pair-protocol hegc --pca-dim "${PCA_DIM:-48}" --gc-protocol "${GC_PROTOCOL:-semi-bin}" \
        --group-protocol atlas --group-version 2 --pad-max mpc --pair-workers "${PAIR_WORKERS:-8}" \
        --mpc-timeout "${MPC_TIMEOUT:-7200}" --net-mbps "${NET_MBPS:-100}" --net-rtt-ms "${NET_RTT_MS:-20}" \
        --out "$OUT/setup" 2>&1 | tee "$OUT/setup.log"
    [[ ${PIPESTATUS[0]} == 0 ]] || fails=1
fi

if has train; then                                   # no MPC: ideal functionality
    echo "[$(date '+%F %T')] train: 30 clients (28 + ${MIXED:-2} special), 5 datasets (label split), plain + similar"
    env -u MPSPDZ MPC=0 OUT="$OUT/train/run" DATA="$DATA5" METHODS="plain similar" PHASES="1 4" \
        DEVICE="$DEVICE" bash run_experiments.sh || fails=1
fi

if has comm; then                                    # no MPC, no training
    rounds=$(( ${COMM_ROUNDS:-5} + 1 ))
    cx=${EXTRA:-}; [[ -n ${GEN_WIDTHS:-} ]] && cx="--gen-widths $GEN_WIDTHS $cx"   # same model as training
    mkdir -p "$OUT/comm"
    for cfg in "3c3d:$DATA3" "30c5d:$DATA5"; do
        for method in plain similar; do
            flags=(--union similar); [[ $method == plain ]] && flags=(--agg plain)
            dir=$OUT/comm/${cfg%%:*}_$method
            echo "[$(date '+%F %T')] comm: ${cfg%%:*} $method ($rounds rounds)"
            # shellcheck disable=SC2086
            env -u MPSPDZ $PY -m secure_code_no_cluster ${cfg#*:} "${flags[@]}" $cx --comm-only --rounds "$rounds" \
                --device "$DEVICE" --net-mbps "${NET_MBPS:-100}" --no-progress --output "$dir" \
                > "$dir.log" 2>&1 || { echo "  FAIL, see $dir.log"; fails=1; }
        done
    done
    $PY - "$OUT/comm" <<'EOF' | tee "$OUT/comm/summary.txt"
import json, sys
from pathlib import Path
print(f"{'run':<16}{'clients':>8}{'up/client':>12}{'down/client':>13}{'compute s':>11}{'transfer s':>12}")
for f in sorted(Path(sys.argv[1]).glob('*/comm_summary.json')):
    s = json.loads(f.read_text())
    print(f"{f.parent.name:<16}{s['clients']:>8}{s['upload_bytes_per_client'] / 1e3:>9.1f} kB"
          f"{s['download_bytes_per_client'] / 1e3:>10.1f} kB{s['compute_seconds']:>11.3f}"
          f"{s['transfer_seconds_per_client']:>12.3f}")
print("(averages over rounds 2..; compute = downlink + aggregation on this host; "
      "transfer = (up + down) per client at --net-mbps)")
EOF
fi
exit $fails
