#!/usr/bin/env bash
# Whole no-cluster paper experiment in one command (resumable: RESUME=latest after a crash).
#
#   phase 1  accuracy: Plain-GeFL, Ours (exact PSI) and Ours (fuzzy PSI), ROUNDS rounds each, all at once
#   phase 2  cost: plain / ours-uncompressed / ours, TIME_ROUNDS rounds each, one at a time (clean timings)
#   phase 3  ablations (ABLATIONS=1): min-holders 1, keep-frac 0.5 / 0.2 / 0.05, two at a time
#   phase 4  figures (tests/plot_paper.py) + a summary table
#
#   bash run_experiments.sh                                   # everything with defaults
#   GEN=cbn bash run_experiments.sh                           # shared trunk + CBN rows instead of the
#                                                             # default: one whole generator per label
#   DEVICE=cuda WORKERS=8 ABLATIONS=1 bash run_experiments.sh
#   tmux new -s exp 'bash run_experiments.sh'                 # keeps running after you disconnect
#   SEQ=1 bash run_experiments.sh                             # phase 1/3 runs one at a time (low GPU memory)
#
# Output: <OUT>_<YYYYMMDD-HHMM>/ per start (link: latest). With RESUME=<folder> a finished run (DONE file)
# is skipped and an unfinished one resumes from its
# checkpoint; a failing one is retried RETRIES times. Logs: $OUT/logs/<run>.log, progress: $OUT/progress.log
# PHASES="2 3 4" runs only those phases (e.g. split the work between two machines; copy the finished
# run folders into one $OUT before the final figures). GPUS="0 1": the two runs of a pair each get their
# own GPU, single runs use the first (unset: every run uses DEVICE).
# In a terminal the screen shows live progress bars + ETA per run (tests/progress.py), refreshed every
# REFRESH seconds; MONITOR=0 turns that off (plain event lines instead), MONITOR=1 forces it on.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-8}              # per run in phases 1 and 3 (two runs share the machine)
TIME_WORKERS=${TIME_WORKERS:-16}   # phase 2 runs alone
ROUNDS=${ROUNDS:-45}
ABL_ROUNDS=${ABL_ROUNDS:-$ROUNDS}  # ablations only need the trend, e.g. ABL_ROUNDS=15
WARMUP=${WARMUP:-0}                # local generator epochs per client before round 1 (every run)
TIME_ROUNDS=${TIME_ROUNDS:-3}
ABLATIONS=${ABLATIONS:-0}
RETRIES=${RETRIES:-3}
# every start gets its own folder <OUT>_<YYYYMMDD-HHMM> (+ link <dir of OUT>/latest);
# RESUME=<that folder> (or RESUME=latest) continues / adds phases to an existing one instead
if [[ -n ${RESUME:-} ]]; then
    [[ $RESUME == latest ]] && RESUME=$(dirname "${OUT:-runs/base}")/latest
    [[ -d $RESUME ]] || { echo "RESUME=$RESUME: no such folder" >&2; exit 1; }
    OUT=$(cd "$RESUME" && pwd -P)
else
    OUT=${OUT:-runs/base}_$(date +%Y%m%d-%H%M)          # OUT is optional: just a name prefix
fi
EXTRA=${EXTRA:-}                   # extra CLI flags for every run, e.g. EXTRA="--smoke" for a dry run
GEN=${GEN:-perlabel}               # perlabel (default): one whole DCGAN per label; cbn: trunk + CBN rows
[[ $GEN == cbn ]] && EXTRA="--no-per-label-gen $EXTRA"
[[ -n ${GEN_WIDTHS:-} ]] && EXTRA="--gen-widths $GEN_WIDTHS $EXTRA"   # per-label DCGAN widths (default 128,64,32)
[[ -n ${GUIDE_EPOCHS:-} ]] && EXTRA="--guide-epochs $GUIDE_EPOCHS $EXTRA"
[[ -n ${GUIDE_WEIGHT:-} ]] && EXTRA="--guide-weight $GUIDE_WEIGHT $EXTRA"
# clients: one per dataset (MNIST, EMNIST, CIFAR-10); most labels then have one holder, so a row is
# updated from a single client (--min-holders 1). emnist10_run_experiments.sh sets its own DATA.
DATA=${DATA:---num-train-mnist 1 --num-train-emnist 1 --num-train-cifar10 1 --min-holders 1}
PHASES=${PHASES:-1 3 4}            # phase 2 only on request: phase 1 records aggregation / distribution
                                    # time and size per round (comm_summary.json, phase 4 table)
METHODS=${METHODS:-plain similar}   # phase 1 runs; also: similar (= fuzzy, run name ours_similar)
# MPC=1 (default): circuit-PSI grouping as a real MPC in MP-SPDZ, downloaded once to third_party/ by
# get_mpspdz.sh; measured cost in the union stats. MPC=0, or no x86-64 Linux: ideal functionality + estimate.
if [[ ${MPC:-1} == 1 && -z ${MPSPDZ:-} ]]; then
    if MPSPDZ=$(bash get_mpspdz.sh); then export MPSPDZ; else echo "MP-SPDZ unavailable: grouping runs as its ideal functionality" >&2; fi
fi
read -r -a GPU_LIST <<< "${GPUS:-}"
SEQ=${SEQ:-0}                       # 1: runs of phases 1 and 3 one at a time, each on all GPUS (clients
                                    # round-robin; use WORKERS >= number of clients to train them in parallel)
MONITOR=${MONITOR:-auto}
REFRESH=${REFRESH:-30}

mkdir -p "$OUT/logs" "$OUT/figs"
ln -sfn "$(cd "$OUT" && pwd -P)" "$(dirname "$OUT")/latest" 2>/dev/null || true
echo "output folder: $OUT"
quiet=
if [[ $MONITOR == 1 || ( $MONITOR == auto && -t 1 ) ]]; then
    $PY -m tests.progress --out "$OUT" --watch "$REFRESH" &     # live bars; events go to progress.log
    monitor=$!
    trap 'kill $monitor 2>/dev/null' EXIT
    quiet=1
fi
stop_monitor() { [[ -n $quiet ]] && kill "$monitor" 2>/dev/null && wait "$monitor" 2>/dev/null; quiet=; }
say() {
    local line="[$(date '+%F %T')] $*"
    echo "$line" >> "$OUT/progress.log"
    [[ -z $quiet ]] && echo "$line"
    return 0
}

run() {  # run <name> <rounds> <workers> [cli flags...]
    local name=$1 rounds=$2 workers=$3; shift 3
    local dir=$OUT/$name log=$OUT/logs/$name.log
    if [[ -f $dir/DONE ]]; then say "skip  $name (already done)"; return 0; fi
    mkdir -p "$dir" && echo "$rounds" > "$dir/target_rounds"     # lets tests/progress.py show it at once
    for ((try = 1; try <= RETRIES; try++)); do
        local resume=()
        [[ -f $dir/checkpoint_last.pt ]] && resume=(--resume "$dir/checkpoint_last.pt")
        say "start $name (try $try${resume[0]:+, resuming})"
        local t0=$SECONDS
        # shellcheck disable=SC2086
        local gpu=()
        [[ -n ${RUN_GPU:-} ]] && gpu=(env "CUDA_VISIBLE_DEVICES=$RUN_GPU")
        local warm=() devs=()
        (( WARMUP > 0 )) && warm=(--warmup-epochs "$WARMUP")
        [[ -n ${RUN_DEVICES:-} ]] && devs=(--devices "$RUN_DEVICES")
        if "${gpu[@]}" $PY -m secure_code_no_cluster $DATA "$@" "${warm[@]}" $EXTRA --rounds "$rounds" --device "$DEVICE" "${devs[@]}" --workers "$workers" \
               --no-progress --output "$dir" "${resume[@]}" >> "$log" 2>&1; then
            touch "$dir/DONE"
            say "done  $name in $(( (SECONDS - t0) / 60 )) min"
            return 0
        fi
        say "FAIL  $name (try $try), last lines of $log:"
        tail -n 5 "$log" | sed 's/^/        /' >> "$OUT/progress.log"
        [[ -z $quiet ]] && tail -n 5 "$log" | sed 's/^/        /'
        sleep ${RETRY_WAIT:-10}
    done
    return 1
}

group() {  # run several runs at once: group "<run args>" "<run args>" ...; GPUs round-robin
    if [[ $SEQ == 1 ]]; then                               # SEQ=1: one after another, all GPUs each
        local rc=0 spec
        local all devs=''                                  # each run gets every GPU in GPUS: clients are
        all=$(IFS=,; echo "${GPU_LIST[*]}")                # spread over them round-robin (--devices)
        (( ${#GPU_LIST[@]} > 1 )) && devs=$(seq -s, -f 'cuda:%g' 0 $(( ${#GPU_LIST[@]} - 1 )))
        for spec in "$@"; do RUN_GPU=$all RUN_DEVICES=$devs eval "run $spec" || rc=1; done
        return $rc
    fi
    local pids=() i=0 prev='' prevpid=''
    for spec in "$@"; do
        local n=${#GPU_LIST[@]} g=''
        (( n > 0 )) && g=${GPU_LIST[$(( i % n ))]}
        # each run starts once the previous one has loaded the data (args.json written), so datasets
        # are downloaded and splits written by one process only
        if [[ -n $prev ]]; then
            while kill -0 "$prevpid" 2>/dev/null && [[ ! -f $prev/args.json && ! -f $prev/DONE ]]; do sleep 5; done
        fi
        RUN_GPU=$g eval "run $spec" & prevpid=$!
        pids+=("$prevpid"); prev=$OUT/${spec%% *}; i=$(( i + 1 ))
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}
pair() { group "$@"; }

fails=0
has() { [[ " $PHASES " == *" $1 "* ]]; }
RUN_GPU=${GPU_LIST[0]:-}                                   # single runs: first GPU
if has 1; then
say "phase 1: accuracy ($ROUNDS rounds, $METHODS $([[ $SEQ == 1 ]] && echo "one at a time" || echo "in parallel"))"
# GPUS="0 1" puts plain + ours_fuzzy on GPU 0, ours on GPU 1
specs=()
for m in $METHODS; do
    case $m in
        plain) specs+=("plain $ROUNDS $WORKERS --agg plain") ;;
        ours) specs+=("ours $ROUNDS $WORKERS") ;;
        ours_fuzzy) specs+=("ours_fuzzy $ROUNDS $WORKERS --union fuzzy") ;;
        similar) specs+=("ours_similar $ROUNDS $WORKERS --union similar") ;;
        *) echo "METHODS: unknown $m" >&2; exit 1 ;;
    esac
done
group "${specs[@]}" || fails=1
fi

if has 2; then
say "phase 2: cost ($TIME_ROUNDS rounds each, one at a time)"
run time_plain    "$TIME_ROUNDS" "$TIME_WORKERS" --agg plain   || fails=1
run time_ours_noq "$TIME_ROUNDS" "$TIME_WORKERS" --no-quantize || fails=1
run time_ours     "$TIME_ROUNDS" "$TIME_WORKERS"               || fails=1
run time_ours_fuzzy "$TIME_ROUNDS" "$TIME_WORKERS" --union fuzzy || fails=1
fi

if [[ $ABLATIONS == 1 ]] && has 3; then
    say "phase 3: ablations ($ABL_ROUNDS rounds, two at a time)"
    pair "ours_t1 $ABL_ROUNDS $WORKERS --min-holders 1" "ours_k0.5 $ABL_ROUNDS $WORKERS --keep-frac 0.5" || fails=1
    pair "ours_k0.2 $ABL_ROUNDS $WORKERS --keep-frac 0.2" "ours_k0.05 $ABL_ROUNDS $WORKERS --keep-frac 0.05" || fails=1
fi

if has 4; then
say "phase 4: figures"
plot() {  # plot <out name> <run:label>...
    local name=$1; shift
    local runs=() labels=()
    for rl in "$@"; do
        if [[ ! -f $OUT/${rl%%:*}/DONE ]]; then
            local need="phase 1"; [[ ${rl%%:*} == time_* ]] && need="phase 2"
            say "      $name figure skipped: ${rl%%:*} not finished (add it: RESUME=$OUT PHASES=\"${need#phase } 4\" bash run_experiments.sh)"
            return
        fi
        runs+=("$OUT/${rl%%:*}"); labels+=("${rl#*:}")
    done
    $PY -m tests.plot_paper --runs "${runs[@]}" --labels "${labels[@]}" --out "$OUT/figs/$name" \
        >> "$OUT/logs/plots.log" 2>&1 && say "      $OUT/figs/$name/paper.{png,pdf,csv}"
}
[[ " $METHODS " == *" ours "* ]] && plot accuracy "plain:Plain-GeFL" "ours:Ours (exact PSI)" "ours_fuzzy:Ours (fuzzy PSI)"
[[ " $METHODS " == *" similar "* ]] && plot accuracy_similar "plain:Plain-GeFL" "ours_similar:Ours (similar)"
[[ -d $OUT/time_plain ]] && plot cost "time_plain:Plain-GeFL" "time_ours_noq:Ours (uncompressed)" "time_ours:Ours (exact PSI)" "time_ours_fuzzy:Ours (fuzzy PSI)"
if [[ $ABLATIONS == 1 ]]; then
    plot keep_frac "ours_k0.5:keep 0.5" "ours_k0.2:keep 0.2" "ours:keep 0.1" "ours_k0.05:keep 0.05"
    plot min_holders "ours_t1:t = 1" "ours:t = 2"
fi

stop_monitor
$PY -m tests.progress --out "$OUT"
$PY - "$OUT" <<'EOF' | tee -a "$OUT/progress.log"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(f"\n{'run':<16}{'rounds':>7}{'final acc':>11}{'best acc':>10}{'upload/client':>15}{'s/round':>9}")
for d in sorted(p for p in out.iterdir() if (p / 'metrics.jsonl').exists()):
    rows = []
    for line in (d / 'metrics.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        continue
    up = rows[-1]['bytes'].get('upload_per_client', rows[-1]['bytes'].get('upload', 0))
    sec = sum(sum(r['seconds'].values()) for r in rows) / len(rows)
    print(f"{d.name:<16}{rows[-1]['round']:>7}{rows[-1]['accuracy']:>11.4f}{max(r['accuracy'] for r in rows):>10.4f}"
          f"{up / 1e3:>12.1f} kB{sec:>9.0f}")
    tail = [r for r in rows[-10:] if 'by_dataset' in r.get('evaluation', {})]   # per test set, last 10 rounds
    if tail:
        names = tail[-1]['evaluation']['by_dataset']
        print(' ' * 16 + '  '.join(f"{k} {sum(r['evaluation']['by_dataset'][k]['accuracy'] for r in tail) / len(tail):.4f}"
                                   for k in names) + f"  (mean of last {len(tail)} rounds)")
EOF
# per-round communication of the phase 1 runs (rounds 2..): aggregation = after local training until the
# aggregator holds the aggregate (upload), distribution = aggregate to the row holders (download);
# s = measured compute (all clients in one process) + one client's bytes at NET_MBPS
NET_MBPS=${NET_MBPS:-100} $PY - "$OUT" <<'EOF' | tee -a "$OUT/progress.log"
import json, os, sys
from pathlib import Path
bps = float(os.environ['NET_MBPS']) * 1e6 / 8
print(f"\nper round (rounds 2.., {bps * 8 / 1e6:g} Mbit/s)   aggregation: kB/client  compute s  total s"
      f"   distribution: kB/client  compute s  total s")
for d in sorted(p for p in Path(sys.argv[1]).iterdir() if (p / 'metrics.jsonl').exists()):
    rows = [json.loads(l) for l in (d / 'metrics.jsonl').read_text().splitlines() if l.strip().startswith('{')]
    rows = rows[1:] or rows
    if not rows:
        continue
    m = lambda f: sum(map(f, rows)) / len(rows)
    up, down = m(lambda r: r['bytes'].get('upload_per_client', r['bytes']['upload'])), m(lambda r: r['bytes']['download_per_client'])
    ca, cd = m(lambda r: r['seconds']['aggregation']), m(lambda r: r['seconds']['downlink'])
    print(f"{d.name:<35}{up / 1e3:>11.1f}{ca:>11.2f}{ca + up / bps:>9.2f}{down / 1e3:>25.1f}{cd:>11.2f}{cd + down / bps:>9.2f}")
EOF

fi

if (( fails )); then say "finished WITH FAILURES: see $OUT/progress.log; retry with RESUME=$OUT bash run_experiments.sh"; exit 1; fi
say "all done"
