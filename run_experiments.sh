#!/usr/bin/env bash
# Whole no-cluster paper experiment in one command (resumable: just run it again after a crash).
#
#   phase 1  accuracy: Plain-GeFL and Ours, ROUNDS rounds each, both at once (one GPU is enough)
#   phase 2  cost: plain / ours-uncompressed / ours, TIME_ROUNDS rounds each, one at a time (clean timings)
#   phase 3  ablations (ABLATIONS=1): min-holders 1, keep-frac 0.5 / 0.2 / 0.05, two at a time
#   phase 4  figures (tests/plot_paper.py) + a summary table
#
#   bash run_experiments.sh                                   # everything with defaults
#   DEVICE=cuda WORKERS=8 ABLATIONS=1 bash run_experiments.sh
#   tmux new -s exp 'bash run_experiments.sh'                 # keeps running after you disconnect
#
# A finished run gets a DONE file and is skipped next time; an unfinished one resumes from its
# checkpoint; a failing one is retried RETRIES times. Logs: $OUT/logs/<run>.log, progress: $OUT/progress.log
# In a terminal the screen shows live progress bars + ETA per run (tests/progress.py), refreshed every
# REFRESH seconds; MONITOR=0 turns that off (plain event lines instead), MONITOR=1 forces it on.
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
WORKERS=${WORKERS:-8}              # per run in phases 1 and 3 (two runs share the machine)
TIME_WORKERS=${TIME_WORKERS:-16}   # phase 2 runs alone
ROUNDS=${ROUNDS:-45}
TIME_ROUNDS=${TIME_ROUNDS:-3}
ABLATIONS=${ABLATIONS:-0}
RETRIES=${RETRIES:-3}
OUT=${OUT:-runs/paper}
EXTRA=${EXTRA:-}                   # extra CLI flags for every run, e.g. EXTRA="--smoke" for a dry run
MONITOR=${MONITOR:-auto}
REFRESH=${REFRESH:-30}

mkdir -p "$OUT/logs" "$OUT/figs"
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
        if $PY -m secure_code_no_cluster "$@" $EXTRA --rounds "$rounds" --device "$DEVICE" --workers "$workers" \
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

pair() {  # run two runs at once: pair "<run args>" "<run args>"
    eval "run $1" & local a=$!
    # the second starts once the first has loaded the data (args.json written), so datasets are
    # downloaded and splits written by one process only
    local first=$OUT/${1%% *}
    while kill -0 $a 2>/dev/null && [[ ! -f $first/args.json && ! -f $first/DONE ]]; do sleep 5; done
    eval "run $2" & local b=$!
    wait $a; local sa=$?
    wait $b; local sb=$?
    return $(( sa || sb ))
}

fails=0
say "phase 1: accuracy ($ROUNDS rounds, plain + ours in parallel)"
pair "plain $ROUNDS $WORKERS --agg plain" "ours $ROUNDS $WORKERS" || fails=1

say "phase 2: cost ($TIME_ROUNDS rounds each, one at a time)"
run time_plain    "$TIME_ROUNDS" "$TIME_WORKERS" --agg plain   || fails=1
run time_ours_noq "$TIME_ROUNDS" "$TIME_WORKERS" --no-quantize || fails=1
run time_ours     "$TIME_ROUNDS" "$TIME_WORKERS"               || fails=1

if [[ $ABLATIONS == 1 ]]; then
    say "phase 3: ablations ($ROUNDS rounds, two at a time)"
    pair "ours_t1 $ROUNDS $WORKERS --min-holders 1" "ours_k0.5 $ROUNDS $WORKERS --keep-frac 0.5" || fails=1
    pair "ours_k0.2 $ROUNDS $WORKERS --keep-frac 0.2" "ours_k0.05 $ROUNDS $WORKERS --keep-frac 0.05" || fails=1
fi

say "phase 4: figures"
plot() {  # plot <out name> <run:label>...
    local name=$1; shift
    local runs=() labels=()
    for rl in "$@"; do
        [[ -f $OUT/${rl%%:*}/DONE ]] || { say "      $name: ${rl%%:*} missing, figure skipped"; return; }
        runs+=("$OUT/${rl%%:*}"); labels+=("${rl#*:}")
    done
    $PY -m tests.plot_paper --runs "${runs[@]}" --labels "${labels[@]}" --out "$OUT/figs/$name" \
        >> "$OUT/logs/plots.log" 2>&1 && say "      $OUT/figs/$name/paper.{png,pdf,csv}"
}
plot accuracy "plain:Plain-GeFL" "ours:Ours"
plot cost "time_plain:Plain-GeFL" "time_ours_noq:Ours (uncompressed)" "time_ours:Ours"
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
EOF

if (( fails )); then say "finished WITH FAILURES: see $OUT/progress.log, rerun this script to retry"; exit 1; fi
say "all done"
