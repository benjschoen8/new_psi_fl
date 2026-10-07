#!/usr/bin/env bash
# Download MP-SPDZ (binary release, x86-64 Linux) into third_party/ once; prints its path.
# Used by run_experiments.sh (MPC=1) so the circuit-PSI grouping runs as a real MPC.
#   export MPSPDZ=$(bash get_mpspdz.sh)
set -euo pipefail
cd "$(dirname "$0")"
VER=${MPSPDZ_VERSION:-0.4.2}
DIR=$PWD/third_party/mp-spdz-$VER
if [[ ! -x $DIR/shamir-party.x ]]; then
    [[ $(uname -s)-$(uname -m) == Linux-x86_64 ]] || { echo "get_mpspdz.sh: binary release is x86-64 Linux only" >&2; exit 1; }
    mkdir -p third_party
    echo "downloading MP-SPDZ $VER (~140 MB) ..." >&2
    curl -fL --retry 3 -o third_party/mp-spdz.tar.xz \
        "https://github.com/data61/MP-SPDZ/releases/download/v$VER/mp-spdz-$VER.tar.xz"
    tar xf third_party/mp-spdz.tar.xz -C third_party && rm third_party/mp-spdz.tar.xz
    (cd "$DIR" && Scripts/tldr.sh >/dev/null && Scripts/setup-ssl.sh 16 >/dev/null 2>&1)
fi
echo "$DIR"
