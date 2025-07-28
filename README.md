# SUMO-RL 🚦 – Gap‑Actuated, Fuzzy & Deep‑RL Traffic‑Light Control
High‑performance research framework for traffic‑signal optimisation with **SUMO**, **TRACI** and **PyTorch** (A2C & REINFORCE‑baseline).  
Includes **gap‑out + fuzzy** hybrids, ready‑to‑train scripts, Docker images (CPU/GPU) and batch files for on‑prem or HPC clusters.

---

## 📁 Repository layout

```text
.
├── README.md                ← you are here
├── docker/                  ← ready‑to‑build images
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   └── Dockerfile.gpu.slim
├── envs/
│   └── sumo_env/scenario/
│       ├── osm.net.xml
│       └── routes_medium.rou.xml
├── models/                  ← pretrained checkpoints
│   ├── policy_a2c_iter0270.pt
│   └── policy_reinf_iter0115.pt
├── outputs/                 ← tensorboard logs, CSVs, gifs, …
├── requirements.txt         ← pinned Python deps
├── scripts/                 ← training & inference runners
│   ├── train_a2c.py
│   ├── train_int.py
│   ├── train_int_policy.py
│   └── train_renforce_bl.py
└── slurm/                   ← batch helpers (adapt to your cluster)
    ├── train_int.sbatch
    └── train_sumo_rl.sh
```

---

## ⚡ Quick start (local)

```bash
git clone https://github.com/juanpablo-perez/sumo-marl-project.git
cd sumo-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **SUMO ≥ 1.16** is required.  
> On Ubuntu: `sudo apt install sumo sumo-tools`

Run a demo simulation:

```bash
python scripts/train_int_policy.py \
  --net envs/sumo_env/scenario/osm.net.xml \
  --route envs/sumo_env/scenario/routes_medium.rou.xml
```

TensorBoard:

```bash
tensorboard --logdir outputs/
```

---

## 🐳 Docker workflow

```bash
# CPU‑only (local dev)
docker build -f docker/Dockerfile.cpu -t sumo-rl:cpu .

# CUDA 12 + PyTorch 2.3 (GPU)
docker build -f docker/Dockerfile.gpu -t sumo-rl:gpu .

# Slim runtime (inference only)
docker build -f docker/Dockerfile.gpu.slim -t sumo-rl:gpu-slim .
```

Run inside the image:

```bash
docker run --gpus all -v $PWD:/workspace sumo-rl:gpu \
  python scripts/train_a2c.py --help
```

---

## 🚀 Cluster / Enroot usage

If your cluster provides **Enroot**, create or pull the container image and launch an interactive session:

```bash
enroot start --root --rw \
  --mount $HOME/sumo_marl_project:/workspace \
  sumo_rl_gpu
```

Then:

```bash
python scripts/train_a2c.py         # or train_renforce_bl.py, etc.
```

Batch examples are in **slurm/** — adjust partitions, GPUs, memory and time limits to match your scheduler.

---

## 🧩 Controllers

| File | Strategy | When to use |
|------|----------|-------------|
| `train_int.py` | **Gap‑Actuated + Fuzzy** baseline | Classical heuristics |
| `train_int_policy.py` | **Gap + A2C** (pretrained) | Best performance |
| `train_renforce_bl.py` | **Gap + REINFORCE‑BL** | Lightweight alternative |

All controllers expose the TRACI hook:

```python
def get_phase_duration(tls_id: str) -> int: ...
```

so they can be plugged into any SUMO runner.

---

## 🏋️‍♀️ Training scripts

| Script | Algorithm | Notes |
|--------|-----------|-------|
| `train_a2c.py` | A2C | 4×512 MLP, LayerNorm, GELU, AMP |
| `train_renforce_bl.py` | REINFORCE + baseline | Monte‑Carlo, entropy‑reg |

Checkpoints are saved to **models/** as `policy_<algo>_iterXXXX.pt`.

---

## 📊 Key metrics

| Metric | Meaning |
|--------|---------|
| `system_mean_wait` | Mean waiting time of all vehicles (s) |
| `vehicle_throughput` | Vehicles reached destination |
| `reward` | −`system_mean_wait` (normalised) |
| `gap_events` | Number of times *gap‑out* triggered |

---

## 🤝 Contributing

1. Fork → clone → create feature branch.  
2. Follow **black** & **ruff** style (`pip install -r dev-requirements.txt`).  
3. Open a PR against **dev** (CI will lint & test).

---

## 📜 Citation

```bibtex
@misc{sumo-rl-2025,
  title  = {Gap‑Actuated, Fuzzy & Deep‑RL Traffic‑Light Control},
  author = {Pérez Vargas, Juan Pablo; Zhangallimbay, Jorge}
  year   = {2025},
  url    = {https://github.com/juanpablo-perez/sumo-marl-project}
}
```

---

## 📝 License

[MIT](LICENSE) © 2025 Juan Pablo Pérez Vargas
