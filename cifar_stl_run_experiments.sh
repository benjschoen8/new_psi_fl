#!/usr/bin/env bash
# CIFAR-10 + STL-10, each split over 3 clients (6 clients). Every client holds a random subset of its
# dataset's classes (CLASSES=4,7: 4 to 7 of 10, every class covered); 9 of 10 classes are shared by the
# two datasets (STL 'car' is named 'automobile'; 'frog' is CIFAR-only, 'monkey' STL-only), so most labels
# have several holders across both datasets. STL-10 images (96x96) are resized to 32x32.
#   SHARE=split (default)   a class's images are split among its holders in that dataset; full: all of them
# Same phases, settings and resume rules as run_experiments.sh; output runs/cifar_stl_<CLASSES>_<SHARE>_<date>.
#   WARMUP=20 PHASES="1 4" EXTRA="--keep-frac 1.0" bash cifar_stl_run_experiments.sh
#   SWEEP=1 bash cifar_stl_run_experiments.sh   # instead: plain_sweep.py (union check first, then the
#       candidate settings, 10 rounds, with --diagnostics; global_augment: shifted / flipped / noisy synthetic
#       images for the server classifier, training.augment; gan_diffaug / gan_spectral_norm: DiffAugment and
#       spectral normalisation in the clients' GAN training, secfl/cbn_gan.py; global_resample: fresh synthetic
#       images every classifier epoch); gen_label_batch 32, widths 64,32,16 (best so far); --heter is the default
#       (local classifiers guide the generators): the list compares guide epochs and weights; JOBS, WORKERS, ROUNDS, OUT, GRID, SWEEP_ARGS override;
#       GPUS="0 1" JOBS=2: one run per GPU at a time.
#       Results: python plain_sweep.py --summary runs/cifar_stl_guide;  per run: --report <run folder>
cd "$(dirname "$0")"
CLASSES=${CLASSES:-4,7}
SHARE=${SHARE:-split}
export DATA="--num-train-cifar10 3 --num-train-stl10 3 --num-train-mnist 0 --num-train-emnist 0 --class-subsets $CLASSES --class-share $SHARE --min-holders 1"
if [[ ${SWEEP:-0} == 1 ]]; then
    GRID=${GRID:-'[
 {"--warmup-epochs":60, "global_augment":true, "gan_diffaug":true, "gan_spectral_norm":true, "global_resample":true, "--guide-epochs":5,  "--guide-weight":0.5},
 {"--warmup-epochs":60, "global_augment":true, "gan_diffaug":true, "gan_spectral_norm":true, "global_resample":true, "--guide-epochs":20, "--guide-weight":0.5},
 {"--warmup-epochs":60, "global_augment":true, "gan_diffaug":true, "gan_spectral_norm":true, "global_resample":true, "--guide-epochs":20, "--guide-weight":1.0},
 {"--warmup-epochs":60, "global_augment":true, "gan_diffaug":true, "gan_spectral_norm":true, "global_resample":true, "--guide-epochs":50, "--guide-weight":0.5}]'}
    exec ${PY:-python} plain_sweep.py --data="$DATA" --jobs "${JOBS:-3}" --workers "${WORKERS:-1}" --gpus "${GPUS:-}" \
        --rounds "${ROUNDS:-10}" --out "${OUT:-runs/cifar_stl_guide}" --grid "$GRID" \
        --extra="--keep-frac 1.0 --diagnostics --gen-widths 64,32,16" ${SWEEP_ARGS:-}
fi
export OUT=${OUT:-runs/cifar_stl_${CLASSES/,/-}_$SHARE}
exec bash run_experiments.sh
