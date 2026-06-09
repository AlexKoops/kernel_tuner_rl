# Kernel Tuner RL

Kernel Tuner RL is a reinforcement learning environment designed around the GPU kernel auto-tuning problem. The environment exposes two action space variants: a Hamming-1 action space, where a single parameter can be incremented or decremented by one step, and a per-parameter action space, where any value within a single parameter can be selected directly. Both action spaces are discrete, making the environment compatible with any RL algorithm supporting discrete actions. Training and evaluation scripts for PPO-based agents are included in `scripts/`.

## Requirements

Python >= 3.9 and pip (or UV)

## Installation

Clone the repo:

```bash
git clone https://github.com/AlexKoops/kernel_tuner_rl.git
cd kernel-tuner-rl
```

### UV

```bash
uv sync
```

### Pip

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Project Structure

- `kernel_tuner_rl/` — main package containing environment logic, backends, and utilities
  - `envs/` — base environment, Hamming-1 environment, and per-parameter environment
- `scripts/` — training script, evaluation script, and bash script for multi-seed training
- `clean_cache/` — pre-computed brute-forced cache files from the AutoTuningAssociation benchmark hub

## Usage

Unpack the .gz files from the clean_cache/ directory

### Training an agent

```bash
python scripts/train.py --agent hamming --train clean_cache/<kernel_name>/*.json --seed 0 --total-timesteps 600000 --tag example
```

#### Multi seed training

The following command trains an hamming-1 agent on 8 different seeds for 600000 timesteps each

```bash
bash scripts/train_seeds.sh train.py convolution_milo 8 600000 hamming
```

### Evaluating a trained agent

```bash
python scripts/evaluate.py --test clean_cache/convolution_milo/*.json --models models/convolution_milo/ppo_seed0 --agent hamming --episodes 10
```

## Supported Kernels

- convolution_milo
- dedispersion_milo
- gemm_milo
- hotspot_milo

## Cache Files

Pre-computed brute-forced search spaces from the [AutoTuningAssociation benchmark hub](https://github.com/AutoTuningAssociation/benchmark_hub), available for the following GPU architectures:

#### NVIDIA

- A100.json
- A4000.json
- A6000.json

#### AMD

- MI250X.json
- W6600.json
- W7800.json

## Citation / Reference

This work builds on [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner) by Ben van Werkhoven.
