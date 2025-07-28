#!/usr/bin/env python3
# ===========================================================================
# REINFORCE + baseline  •  SUMO-RL + PettingZoo  • 
# ===========================================================================

import os, random, logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import ray, traci
from pettingzoo.utils import ParallelEnv
from sumo_rl import parallel_env as sumo_parallel_env

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("train_rl")

# ───────────────────────── Config ───────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    net_file: str
    route_file: str
    num_seconds: int = 14_400
    delta_time: int  = 5
    obs_size: int    = 9
    action_size: int = 2
    gamma: float     = 0.98
    lr: float        = 2e-4
    num_iters: int   = 600
    num_workers: int = 16
    seed: int        = 12345
    entropy_start: float = 0.05
    entropy_end:   float = 0.02
    decay_iter:    int   = 300
    reward_clip:   float = 20.0
    max_stopped:   int   = 500
    ckpt_every:    int   = 5
    grad_clip:     float = 0.5
    value_coef:    float = 0.25

# ───────── Observación (9-D) ───────────────────────────────────────────────
def build_obs(tls: str) -> np.ndarray:
    lanes = traci.trafficlight.getControlledLanes(tls)
    phase = traci.trafficlight.getPhase(tls)
    min_g = traci.trafficlight.getPhaseDuration(tls) <= 0
    halt  = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
    spd   = [traci.lane.getLastStepMeanSpeed(l)     for l in lanes]

    return np.array([
        phase / 10.0,
        float(min_g),
        np.mean(halt) if halt else 0.0,
        np.max(halt)  if halt else 0.0,
        np.sum(halt)  if halt else 0.0,
        np.mean(spd)  if spd  else 0.0,
        np.min(spd)   if spd  else 0.0,
        np.max(spd)   if spd  else 0.0,
        len(lanes) / 10.0
    ], dtype=np.float32)

# ───────── Wrapper PettingZoo ──────────────────────────────────────────────
class SumoPZ(ParallelEnv):
    metadata = {"name": "sumo_tls_v0", "parallel": True}

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._env = sumo_parallel_env(
            net_file   = cfg.net_file,
            route_file = cfg.route_file,
            num_seconds= cfg.num_seconds,
            delta_time = cfg.delta_time,
            use_gui    = False
        )
        self.possible_agents = list(self._env.possible_agents)
        self.agents = self.possible_agents.copy()

    def reset(self, seed=None, options=None):
        obs_raw = self._env.reset()[0]
        self.agents = list(obs_raw)
        return {a: build_obs(a) for a in obs_raw}, {}

    def step(self, acts):
        out = self._env.step(acts)
        if len(out) == 5:
            obs_raw, _, term, trunc, infos = out
            dones = {a: term.get(a, False) or trunc.get(a, False)
                     for a in set(term) | set(trunc)}
        else:
            obs_raw, _, dones, infos = out

        rewards = {}
        for tls, d in infos.items():
            r = -d.get("system_mean_waiting_time", 0.0) \
                + 0.1 * d.get("system_mean_speed", 0.0)
            rewards[tls] = float(np.clip(r, -self.cfg.reward_clip,
                                            self.cfg.reward_clip))

        obs = {t: build_obs(t) for t in obs_raw}
        self.agents = [a for a, done in dones.items() if not done]
        return obs, rewards, dones, infos

    def close(self): self._env.close()

# ───────── Red Actor-Critic ───────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs: int, act: int, hid: int = 512, p: float = 0.1):
        super().__init__()
        layers, in_f = [], obs
        for _ in range(4):
            layers += [nn.Linear(in_f, hid),
                       nn.LayerNorm(hid),
                       nn.GELU(),
                       nn.Dropout(p)]
            in_f = hid
        self.shared = nn.Sequential(*layers)
        self.actor  = nn.Linear(hid, act)
        self.critic = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor):
        z = self.shared(x)
        return torch.softmax(self.actor(z), dim=-1), self.critic(z).squeeze(-1)

# ───────── Rollout remoto (Ray) ───────────────────────────────────────────
@ray.remote(num_cpus=1)
def rollout_worker(state_dict, cfg: Config, seed: int):
    os.environ.setdefault("SUMO_HOME", "/opt/sumo")
    os.environ["PATH"]       = "/opt/sumo/bin:"   + os.getenv("PATH", "")
    os.environ["PYTHONPATH"] = "/opt/sumo/tools:" + os.getenv("PYTHONPATH", "")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = ActorCritic(cfg.obs_size, cfg.action_size)
    net.load_state_dict(state_dict); net.eval()

    env = SumoPZ(cfg)
    obs, _ = env.reset()

    obs_b, act_b, rew_b, waits = [], [], [], []
    done = False
    while not done:
        acts = {}
        for tls, o in obs.items():
            with torch.no_grad():
                pi, _ = net(torch.tensor(o, dtype=torch.float32))
            a = torch.distributions.Categorical(pi).sample().item()
            acts[tls] = a
            obs_b.append(o); act_b.append(a)

        next_obs, rews, dns, infos = env.step(acts)

        for tls in acts:
            rew_b.append(rews[tls])
        waits.append(np.mean([d["system_mean_waiting_time"] for d in infos.values()]))

        if sum(d["system_total_stopped"] for d in infos.values()) > cfg.max_stopped:
            break

        obs, done = next_obs, all(dns.values())

    env.close()

    # retornos descontados
    G, returns = 0.0, []
    for r in reversed(rew_b):
        G = r + cfg.gamma * G
        returns.insert(0, G)
    return obs_b, act_b, returns, float(np.median(waits))

# ───────── Trainer principal ─────────────────────────────────────────────
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = ActorCritic(cfg.obs_size, cfg.action_size)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.sched = LambdaLR(self.opt, lr_lambda=lambda it:
                              0.5 if it >= cfg.decay_iter else 1.0)
        self.tb = SummaryWriter("runs/reinforce_bl_rev")
        self.start_iter = self._load_ckpt() + 1
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # — carga sólo los PESOS, no el estado de Adam —
    def _load_ckpt(self) -> int:
        ckpts = sorted(Path("models").glob("policy_reinf_iter0050.pt"))
        if not ckpts: return 0
        d = torch.load(ckpts[-1], map_location="cpu")
        self.net.load_state_dict(d["model"])
        log.info(f"Resumed weights from {ckpts[-1].name}")
        return d["iter"]

    # — coeficiente de entropía variable —
    def _entropy_coef(self, it):
        if it <= self.cfg.decay_iter: return self.cfg.entropy_start
        frac = (it - self.cfg.decay_iter) / (self.cfg.num_iters - self.cfg.decay_iter)
        return (self.cfg.entropy_start -
                frac * (self.cfg.entropy_start - self.cfg.entropy_end))

    def train(self):
        ray.init(ignore_reinit_error=True, num_cpus=self.cfg.num_workers)
        for it in range(self.start_iter, self.cfg.num_iters + 1):
            state_cpu = {k: v.cpu() for k, v in self.net.state_dict().items()}
            seeds = [self.cfg.seed + it*100 + w for w in range(self.cfg.num_workers)]
            futs  = [rollout_worker.remote(state_cpu, self.cfg, s) for s in seeds]
            res   = ray.get(futs)

            obs, act, ret, waits = [], [], [], []
            for o, a, r, w in res:
                if len(o) == 0: continue
                obs += o; act += a; ret += r; waits.append(w)
            if len(obs) == 0:
                log.warning("Iteración sin transiciones; saltada"); continue

            obs_t = torch.tensor(obs, dtype=torch.float32)
            act_t = torch.tensor(act, dtype=torch.int64)
            ret_t = torch.tensor(ret, dtype=torch.float32)

            pi, vals = self.net(obs_t)
            logp = torch.log(pi.gather(1, act_t[:, None]).squeeze() + 1e-8)
            adv = ret_t - vals.detach()
            # — normalización de ventaja —
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            actor_loss  = -(logp * adv).mean()
            critic_loss = F.mse_loss(vals, ret_t / 10.0)
            ent = -(pi * torch.log(pi + 1e-8)).sum(1).mean()

            loss = (actor_loss +
                    self.cfg.value_coef * critic_loss -
                    self._entropy_coef(it) * ent)

            self.opt.zero_grad(); loss.backward()
            clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
            self.opt.step(); self.sched.step()

            wm = float(np.median(waits)); lr = self.opt.param_groups[0]["lr"]
            log.info(f"[{it:04d}] wait_med={wm:.2f}s | "
                     f"a_loss={actor_loss:.2f} | c_loss={critic_loss:.2f} | "
                     f"ent={ent:.3f} | lr={lr:.5f}")
            self.tb.add_scalars("Train", {
                "Wait/median": wm,
                "ActorLoss": actor_loss.item(),
                "CriticLoss": critic_loss.item(),
                "Entropy": ent.item(),
                "LR": lr
            }, it)

            if it % self.cfg.ckpt_every == 0 or it == self.cfg.num_iters:
                Path("models").mkdir(exist_ok=True)
                p = f"models/policy_reinf_iter{it:04d}.pt"
                torch.save({"iter": it,
                            "model": self.net.state_dict()}, p)
                log.info(f"✔ saved {p}")

        self.tb.close(); ray.shutdown()

# ───────── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("SUMO_HOME", "/opt/sumo")
    os.environ["PATH"]       = "/opt/sumo/bin:"   + os.getenv("PATH", "")
    os.environ["PYTHONPATH"] = "/opt/sumo/tools:" + os.getenv("PYTHONPATH", "")

    cfg = Config(
        net_file   = "envs/sumo_env/scenario/osm.net.xml",
        route_file = "envs/sumo_env/scenario/routes_medium.rou.xml"
    )
    Trainer(cfg).train()
