# SUMO-RL PettingZoo MARL Template

This repository contains a **minimal, runnable example** of a multi‑agent
reinforcement‑learning workflow that couples **SUMO**, **PettingZoo** and
**Ray/RLlib (PPO)**. It is pre‑configured for execution on the **CEDIA HPC** via
*enroot* containers, but you can also run it on any CUDA‑enabled workstation.

## 1. Quick start (local GPU)

```bash
# Build image locally (CUDA 12.4 required)
docker build -t jp/sumo-rl:1.0 -f docker/Dockerfile .

# Run training test
docker run --gpus all -it --rm \
       -v $(pwd):/workspace jp/sumo-rl:1.0 \
       python scripts/train_marl.py --num-iterations 2
```

The script will create a small network with ~2 500 vehicles and run **2 PPO
iterations**, printing the mean episode reward.

## 2. Push to Docker Hub

```bash
docker tag jp/sumo-rl:1.0 your_dockerhub/sumo-rl:1.0
docker push your_dockerhub/sumo-rl:1.0
```

## 3. Deploy on CEDIA HPC

```bash
# On login node
module load enroot
enroot import docker://your_dockerhub/sumo-rl:1.0
enroot create --name sumo-rl-1.22 sumo-rl_1.0.sqsh
```

Submit the provided Slurm script:

```bash
sbatch slurm/train_sumo_rl.sh
```

Logs, TensorBoard data and checkpoints will be in `~/sumo_runs/<JOB_ID>/`.

## 4. Clean up

```bash
docker rmi jp/sumo-rl:1.0
enroot remove sumo-rl-1.22
```

> **Tip:** Adjust `num-iterations`, network files or RL hyper‑parameters in
`scripts/train_marl.py` as needed for full‑scale experiments.
