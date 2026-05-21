#!/bin/zsh

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <path_to_train_script> <kernel> <n_experiments> <timesteps_per_experiment>"
  exit 1
fi

SCRIPT=$1
KERNEL=$2
N=$3
TIMESTEPS=$4

for i in $(seq 0 $((N - 1))); do
  uv run python scripts/$SCRIPT \
    --train clean_cache/$KERNEL/A100.json clean_cache/$KERNEL/A4000.json clean_cache/$KERNEL/MI250X.json clean_cache/$KERNEL/W6600.json \
    --holdout clean_cache/$KERNEL/A6000.json clean_cache/$KERNEL/W7800.json \
    --seed $i --total-timesteps $TIMESTEPS
done
