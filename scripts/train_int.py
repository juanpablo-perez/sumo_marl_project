#!/usr/bin/env python3
# ======================================================================
#  RL training – SUMO-RL + PettingZoo (resumible, GPU-ready, robust)
#  * Reanuda del último checkpoint
#  * Big-MLP 4×512 (LayerNorm+GELU) con AMP
#  * Reward-clip ±20   |  γ = 0.98
#  * Early-stop si system_total_stopped > 500
#  * Métrica de espera = mediana (menos sensible a outliers)
# ======================================================================

import os, random, logging
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import ray, traci
from pettingzoo.utils import ParallelEnv
from sumo_rl import parallel_env as sumo_parallel_env
from train_int_policy import Policy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("train_rl")

# ───────── Config ────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    net_file: str
    route_file: str
    num_seconds: int = 14_400
    delta_time: int  = 10
    obs_size: int    = 9
    action_size: int = 2
    gamma: float     = 0.98          # ← más miopía
    lr: float        = 5e-4
    num_iters: int   = 600
    num_workers: int = 8
    seed: int        = 12345
    entropy_start: float = 0.01
    entropy_end:   float = 0.005
    decay_iter:    int   = 300
    reward_clip:   float = 20.0      # ← clip estricto
    max_stopped:   int   = 500       # ← early-stop
    ckpt_every:    int   = 50
    grad_clip:     float = 0.5

# ───────── Obs builder 9-dim ─────────────────────────────────────────
def build_obs(tls: str) -> np.ndarray:
    lanes=traci.trafficlight.getControlledLanes(tls)
    phase=traci.trafficlight.getPhase(tls)
    min_g=traci.trafficlight.getPhaseDuration(tls)<=0
    halt=[traci.lane.getLastStepHaltingNumber(l) for l in lanes]
    spd =[traci.lane.getLastStepMeanSpeed(l)     for l in lanes]
    mean_h,max_h,sum_h=(np.mean(halt) if halt else 0,
                        np.max(halt) if halt else 0,
                        np.sum(halt) if halt else 0)
    mean_s,min_s,max_s=(np.mean(spd) if spd else 0,
                        np.min(spd)  if spd else 0,
                        np.max(spd)  if spd else 0)
    return np.array([phase/10,min_g,mean_h,max_h,sum_h,
                     mean_s,min_s,max_s,len(lanes)/10],dtype=np.float32)

# ───────── PZ wrapper ────────────────────────────────────────────────
class SumoPZ(ParallelEnv):
    metadata={"render.modes":[]}
    def __init__(self,cfg:Config):
        self._env=sumo_parallel_env(net_file=cfg.net_file,
                                    route_file=cfg.route_file,
                                    num_seconds=cfg.num_seconds,
                                    delta_time=cfg.delta_time,
                                    use_gui=False)
        self.possible_agents=list(self._env.possible_agents); self.agents=self.possible_agents
    def reset(self):
        o=self._env.reset(); o=o[0] if isinstance(o,tuple) else o
        self.agents=list(o); return {t:build_obs(t) for t in self.agents}
    def step(self,acts):
        out=self._env.step(acts)
        if len(out)==5:
            obs,_,ter,trc,inf=out
            done={a:ter.get(a,0) or trc.get(a,0) for a in ter.keys()|trc.keys()}
        else: obs,_,done,inf=out
        rew={}
        for t,d in inf.items():
            r=-d.get("system_mean_waiting_time",0)+0.1*d.get("system_mean_speed",0)
            rew[t]=np.clip(r,-cfg.reward_clip,cfg.reward_clip)
        obs={t:build_obs(t) for t in obs}
        self.agents=[a for a,d in done.items() if not d]
        return obs,rew,done,inf
    def close(self): self._env.close()

# ───────── Remote rollout (early-stop) ───────────────────────────────
@ray.remote(num_cpus=1)
def worker(state, cfg: Config, seed: int, device_str: str):
    device = torch.device("cpu")                     # workers solo CPU
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    pol = Policy(cfg.obs_size, cfg.action_size).to(device)
    pol.load_state_dict(state); pol.eval()

    env = SumoPZ(cfg)
    obs = env.reset()

    ob, ac, rew, waits = [], [], [], []              # ← LISTA 'rew'
    sum_r = 0.0
    done  = False

    while not done:
        acts = {}
        for tls, o in obs.items():
            with torch.no_grad():
                a = torch.distributions.Categorical(
                    pol(torch.tensor(o, device=device))
                ).sample().item()
            acts[tls] = a
            ob.append(o); ac.append(a)

        obs, rws, dns, inf = env.step(acts)

        # early-stop si atasco extremo
        total_stopped = sum(inf[t]["system_total_stopped"] for t in inf)
        if total_stopped > cfg.max_stopped:
            break

        for r in rws.values():
            r_clipped = float(np.clip(r, -cfg.reward_clip, cfg.reward_clip))
            rew.append(r_clipped)                    # ← usa la lista correcta
            sum_r += r_clipped

        waits.append(np.mean([inf[t]["system_mean_waiting_time"] for t in inf]))
        done = all(dns.values())

    env.close()
    return ob, ac, rew, sum_r, float(np.mean(waits))

# ───────── Trainer ───────────────────────────────────────────────────
class Trainer:
    def __init__(self,cfg:Config):
        self.cfg=cfg
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pol=Policy(cfg.obs_size,cfg.action_size).to(self.device)
        self.opt=optim.Adam(self.pol.parameters(),lr=cfg.lr)
        self.sched=LambdaLR(self.opt,lr_lambda=lambda it:0.5 if it>=cfg.decay_iter else 1.0)
        self.tb=SummaryWriter("runs/rl_wait_opt")
        self.start_iter=self._load_ckpt()+1
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    def _load_ckpt(self)->int:
        return 0
        ckpts=sorted(Path("models").glob("policy_wait_iter*.pt"))
        if not ckpts: return 0
        data=torch.load(ckpts[-1],map_location="cpu")
        self.pol.load_state_dict(data["model"])
        self.opt.load_state_dict(data["optim"])
        self.sched.load_state_dict(data["sched"])
        log.info(f"Resumed from {ckpts[-1].name}")
        return data["iter"]

    def _coef(self,it):   # entropía lineal
        if it<=self.cfg.decay_iter: return self.cfg.entropy_start
        frac=(it-self.cfg.decay_iter)/(self.cfg.num_iters-self.cfg.decay_iter)
        return self.cfg.entropy_start-frac*(self.cfg.entropy_start-self.cfg.entropy_end)

    def train(self):
        ray.init(ignore_reinit_error=True, num_cpus=self.cfg.num_workers)
        state_cpu={k:v.cpu() for k,v in self.pol.state_dict().items()}

        for it in range(self.start_iter,self.cfg.num_iters+1):
            seeds=[self.cfg.seed+it*100+w for w in range(self.cfg.num_workers)]
            fut=[worker.remote(state_cpu,self.cfg,s,"cpu") for s in seeds]
            res=ray.get(fut)

            obs,acts,rets=[],[],[]
            waits,step_r,ep_ret=[],[],[]
            for o,a,r,sum_r,w in res:
                obs+=o; acts+=a; waits.append(w); step_r.append(sum_r/len(r))
                R=0; disc=[]
                for x in reversed(r):
                    R=x+self.cfg.gamma*R; disc.insert(0,R)
                ep_ret.append(disc[0]); rets+=disc

            obs_t=torch.tensor(np.array(obs),dtype=torch.float32,device=self.device)
            act_t=torch.tensor(acts,dtype=torch.long,device=self.device)
            ret_t=torch.tensor(rets,dtype=torch.float32,device=self.device)
            adv=(ret_t-ret_t.mean())/(ret_t.std()+1e-8)

            probs=self.pol(obs_t)
            logp=torch.log(probs.gather(1,act_t[:,None]).squeeze())
            ent=-(probs*torch.log(probs+1e-8)).sum(1).mean()
            loss=-(logp*adv).mean()-self._coef(it)*ent

            self.opt.zero_grad(); loss.backward()
            clip_grad_norm_(self.pol.parameters(),self.cfg.grad_clip)
            self.opt.step(); self.sched.step()

            median_wait=float(np.median(waits))
            lr=self.opt.param_groups[0]["lr"]
            log.info(f"[{it:04d}] wait_med={median_wait:.2f}s | "
                     f"loss={loss.item():.3f} | ent={ent.item():.3f} | lr={lr:.5f}")
            self.tb.add_scalar("Wait/median",median_wait,it)
            self.tb.add_scalar("Loss/train",loss.item(),it)

            if it%self.cfg.ckpt_every==0 or it==self.cfg.num_iters:
                Path("models").mkdir(exist_ok=True)
                ckpt=f"models/policy_wait_iter{it:04d}.pt"
                torch.save({"iter":it,"model":self.pol.state_dict(),
                            "optim":self.opt.state_dict(),
                            "sched":self.sched.state_dict()},ckpt)
                log.info(f"✔ saved {ckpt}")

        self.tb.close(); ray.shutdown()

# ───────── Main ──────────────────────────────────────────────────────
if __name__=="__main__":
    os.environ["SUMO_HOME"]="/opt/sumo"
    os.environ["PATH"]   ="/opt/sumo/bin:"+os.getenv("PATH","")
    os.environ["PYTHONPATH"]="/opt/sumo/tools:"+os.getenv("PYTHONPATH","")
    cfg=Config(
        net_file="envs/sumo_env/scenario/osm.net.xml",
        route_file="envs/sumo_env/scenario/routes_medium.rou.xml",
    )
    Trainer(cfg).train()
