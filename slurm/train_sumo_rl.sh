#!/bin/bash
#SBATCH -J sumo-rl
#SBATCH -p gpu-dev
#SBATCH -c 16
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out

module load enroot
export NVIDIA_DRIVER_CAPABILITIES=all

LOG_DIR="$HOME/sumo_runs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

enroot start --root --rw sumo-rl-1.22 << EOF
  ray start --head --include-dashboard=false --num-gpus=1 --num-cpus=16
  python scripts/train_marl.py --num-iterations 2 --results-dir $LOG_DIR
EOF
