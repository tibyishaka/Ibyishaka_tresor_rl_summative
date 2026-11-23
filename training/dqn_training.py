"""DQN (value-based) training on SmallholderFarmEnv with TorchRL.

Features: experience replay (TensorDictReplayBuffer), target network (soft/hard update),
optional Double-DQN, linear epsilon-greedy annealing.

Usage:
    uv run python -m training.dqn_training --sweep          # 10 hyper-parameter runs
    uv run python -m training.dqn_training --run 0          # a single config
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (
    LOG_DIR, MODEL_DIR, N_ACTIONS, OBS_DIM, episodes_from_batch, evaluate_net, make_env, mlp,
    save_run, set_seed,
)

BASE = dict(
    lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64, eps_init=1.0, eps_end=0.05,
    eps_frac=0.3, target="soft", tau=0.01, double=False, hidden=[128, 128],
    total_frames=150_000, frames_per_batch=50, utd=2, warmup=2_000, seed=0,
)

# 10 hyper-parameter combinations (each changes 1-3 knobs from BASE)
SWEEP = [
    dict(),                                                        # 0 baseline
    dict(lr=5e-4),                                                 # 1 lower LR
    dict(lr=3e-3),                                                 # 2 higher LR
    dict(gamma=0.9),                                               # 3 short horizon
    dict(gamma=0.999),                                             # 4 long horizon
    dict(buffer_size=5_000, batch_size=32),                        # 5 small buffer/batch
    dict(batch_size=256, buffer_size=100_000),                     # 6 large batch/buffer
    dict(eps_end=0.01, eps_frac=0.1),                              # 7 fast greedy exploration
    dict(eps_end=0.2, eps_frac=0.6),                               # 8 heavy exploration
    dict(double=True, target="hard", tau=1.0, lr=5e-4),            # 9 Double-DQN + hard target
    dict(double=True, hidden=[256, 256], lr=5e-4, gamma=0.995),    # 10 Double-DQN, bigger net
]


def train(cfg: dict, run_id: int, save_model: bool = False, verbose: bool = True):
    set_seed(cfg["seed"])
    env = make_env(seed=cfg["seed"])
    qnet = mlp(OBS_DIM, N_ACTIONS, cfg["hidden"], act=torch.nn.ReLU)
    actor = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(
        env.action_spec, eps_init=cfg["eps_init"], eps_end=cfg["eps_end"],
        annealing_num_steps=int(cfg["eps_frac"] * cfg["total_frames"]),
    )
    policy = TensorDictSequential(actor, egreedy)

    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True,
                      double_dqn=cfg["double"], loss_function="smooth_l1")
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target"] == "soft":
        updater = SoftUpdate(loss_fn, tau=cfg["tau"])
    else:
        updater = HardUpdate(loss_fn, value_network_update_interval=500)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])
    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"], auto_register_policy_transforms=False, init_random_frames=cfg["warmup"])

    episodes, updates, frames, t0 = [], [], 0, time.time()
    for i, data in enumerate(collector):
        frames += data.numel()
        episodes_from_batch(data, frames, episodes)
        rb.extend(data.reshape(-1))
        egreedy.step(data.numel())
        if frames < cfg["warmup"]:
            continue
        for _ in range(cfg["utd"]):
            sample = rb.sample()
            loss = loss_fn(sample)["loss"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
            optim.step()
            updater.step()
        if i % 20 == 0:
            updates.append({"frames": frames, "loss": float(loss.item()), "epsilon": float(egreedy.eps.item()),
                            "q_mean": float(sample["chosen_action_value"].mean().item()) if "chosen_action_value" in sample.keys() else float("nan")})
        if verbose and i % 200 == 0 and episodes:
            recent = [e["return"] for e in episodes[-50:]]
            print(f"[DQN run {run_id}] frames {frames:6d} eps {egreedy.eps.item():.2f} "
                  f"loss {loss.item():.3f} avgR(50) {sum(recent)/len(recent):+.2f}")
    collector.shutdown()

    result = evaluate_net(qnet, n_episodes=100)
    result_unseen = evaluate_net(qnet, n_episodes=100, scenario="unseen")
    result["unseen_mean_return"] = result_unseen["mean_return"]
    result["unseen_std_return"] = result_unseen["std_return"]
    result["train_time_s"] = time.time() - t0
    result["episodes"] = len(episodes)
    save_run("dqn", run_id, cfg, episodes, updates, result)
    if save_model:
        (MODEL_DIR / "dqn").mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": qnet.state_dict(), "config": cfg}, MODEL_DIR / "dqn" / f"dqn_run{run_id:02d}.pt")
    print(f"[DQN run {run_id}] eval mean {result['mean_return']:+.2f} ± {result['std_return']:.2f} "
          f"unseen {result['unseen_mean_return']:+.2f} ({result['train_time_s']:.0f}s)")
    return result, qnet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--runs", type=str, default=None, help="comma-separated subset of sweep runs")
    args = p.parse_args()
    runs = range(len(SWEEP)) if args.sweep else [args.run]
    if args.runs:
        runs = [int(x) for x in args.runs.split(",")]
    summary = []
    for r in runs:
        cfg = {**BASE, **SWEEP[r]}
        if args.frames:
            cfg["total_frames"] = args.frames
        res, _ = train(cfg, r, save_model=True)
        summary.append({"run": r, **{k: v for k, v in cfg.items()}, **{k: v for k, v in res.items() if k != "returns"}})
    if args.sweep and not args.runs:
        import pandas as pd

        pd.DataFrame(summary).to_csv(LOG_DIR / "dqn_sweep.csv", index=False)
        best = max(summary, key=lambda s: s["mean_return"])
        with open(LOG_DIR / "dqn_best.json", "w") as f:
            json.dump({"run": best["run"]}, f)


if __name__ == "__main__":
    main()
