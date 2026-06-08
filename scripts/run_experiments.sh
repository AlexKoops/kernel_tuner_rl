#!/bin/zsh
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <script> <kernel> <n_seeds> <timesteps> [agent] [tag]"
  echo ""
  echo "  <script>   training script filename (e.g. train.py)"
  echo "  <kernel>   kernel name matching folder in clean_cache/"
  echo "  <n_seeds>  number of seeds to train"
  echo "  <timesteps> timestep budget per run"
  echo "  [agent]    optional agent type: 'hamming' (default) or 'perparameter'"
  echo "  [tag]      optional run label"
  exit 1
fi

SCRIPT=$1
KERNEL=$2
N=$3
TIMESTEPS=$4
AGENT=${5:-hamming}  # default to hamming if not provided
TAG=$6

for i in $(seq 0 $((N - 1))); do
  ARGS=(
    --agent $AGENT
    --train clean_cache/$KERNEL/A100.json clean_cache/$KERNEL/A4000.json clean_cache/$KERNEL/MI250X.json clean_cache/$KERNEL/W6600.json
    --seed $i --total-timesteps $TIMESTEPS
  )

  if [[ -n "$TAG" ]]; then
    ARGS+=(--tag $TAG)
  fi

  uv run python scripts/$SCRIPT "${ARGS[@]}"
done
