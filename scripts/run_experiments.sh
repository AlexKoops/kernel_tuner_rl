#!/bin/zsh

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <kernel> <n_experiments>"
  exit 1
fi

KERNEL=$1
N=$2

for i in $(seq 0 $((N - 1))); do
  uv run python scripts/train.py \
    --train clean_cache/$KERNEL/A100.json.gz clean_cache/$KERNEL/A4000.json.gz clean_cache/$KERNEL/MI250X.json.gz clean_cache/$KERNEL/W6600.json.gz \
    --holdout clean_cache/$KERNEL/A6000.json.gz clean_cache/$KERNEL/W7800.json.gz \
    --seed $i
done
