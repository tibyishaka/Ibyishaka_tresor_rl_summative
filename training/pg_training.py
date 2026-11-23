"""Policy-gradient training (REINFORCE, PPO, A2C) on SmallholderFarmEnv with TorchRL.

* REINFORCE - Monte-Carlo policy gradient on complete episodes collected with
  ``env.rollout``; baseline = learned V(s) (Monte-Carlo regression), running mean or none;
  optional return normalisation and entropy bonus.
* PPO       - ``ClipPPOLoss`` + ``GAE`` with mini-batch epochs over each collected batch.
* A2C       - ``A2CLoss`` + ``GAE``(n-step style), a single synchronous update per batch.

Usage:
    uv run python -m training.pg_training --algo ppo --sweep
    uv run python -m training.pg_training --algo reinforce --run 3
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from training.common import (
    LOG_DIR, MODEL_DIR, N_ACTIONS, OBS_DIM, episodes_from_batch, evaluate_net, make_env, mlp,
    save_run, set_seed,
)

# ----------------------------------------------------------------------------- configs
BASES = {
    "reinforce": dict(lr=1e-3, gamma=0.99, episodes_per_update=8, baseline="value", normalize=True,
                      entropy_coef=0.02, hidden=[64, 64], total_frames=150_000, seed=0),
    "ppo": dict(lr=3e-4, gamma=0.99, lmbda=0.95, clip_eps=0.2, epochs=4, frames_per_batch=1024,
                minibatch=256, entropy_coef=0.01, critic_coef=0.5, hidden=[64, 64],
                total_frames=150_000, seed=0),
    "a2c": dict(lr=7e-4, gamma=0.99, lmbda=1.0, frames_per_batch=64, entropy_coef=0.01,
                critic_coef=0.5, hidden=[64, 64], total_frames=150_000, seed=0),
}

SWEEPS = {
    "reinforce": [
        dict(),                                           # 0 baseline (learned V(s) baseline)
        dict(lr=3e-4),                                    # 1 low LR
        dict(lr=3e-3),                                    # 2 high LR
        dict(gamma=0.95),                                 # 3 shorter horizon
        dict(gamma=0.999),                                # 4 near-undiscounted
        dict(baseline="mean"),                            # 5 running-mean scalar baseline
        dict(baseline="none", normalize=False),           # 6 vanilla REINFORCE
        dict(episodes_per_update=2),                      # 7 small batch of episodes
        dict(episodes_per_update=32, lr=3e-3),            # 8 large batch of episodes
        dict(entropy_coef=0.0),                           # 9 no entropy bonus
        dict(entropy_coef=0.05, hidden=[128, 128]),       # 10 strong entropy + bigger net
    ],
    "ppo": [
        dict(),                                           # 0 baseline
        dict(lr=1e-4),                                    # 1 low LR
        dict(lr=1e-3),                                    # 2 high LR
        dict(clip_eps=0.1),                               # 3 tight clip
        dict(clip_eps=0.3),                               # 4 loose clip
        dict(epochs=10),                                  # 5 many epochs
        dict(frames_per_batch=256, minibatch=64),         # 6 small batches
        dict(frames_per_batch=2048, minibatch=512),       # 7 large batches
        dict(lmbda=0.8, gamma=0.95),                      # 8 short-horizon GAE
        dict(entropy_coef=0.0),                           # 9 no entropy
        dict(entropy_coef=0.05, hidden=[128, 128]),       # 10 high entropy + big net
    ],
    "a2c": [
        dict(),                                           # 0 baseline
        dict(lr=2.5e-4),                                  # 1 low LR
        dict(lr=2e-3),                                    # 2 high LR
        dict(frames_per_batch=16),                        # 3 short rollouts (TD-like)
        dict(frames_per_batch=256),                       # 4 long rollouts
        dict(lmbda=0.95),                                 # 5 GAE lambda 0.95
        dict(gamma=0.95),                                 # 6 short horizon
        dict(entropy_coef=0.0),                           # 7 no entropy
        dict(entropy_coef=0.05),                          # 8 high entropy
        dict(critic_coef=1.0, hidden=[128, 128]),         # 9 stronger critic, bigger net
        dict(lr=1e-3, frames_per_batch=128, lmbda=0.95),  # 10 mixed
    ],
}


# ----------------------------------------------------------------------------- modules
def build_actor(env, hidden):
    logits_net = mlp(OBS_DIM, N_ACTIONS, hidden)
    module = TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(module, spec=env.action_spec, in_keys=["logits"],
                               distribution_class=OneHotCategorical, return_log_prob=True)
    return actor, logits_net


def build_critic(hidden):
    return ValueOperator(mlp(OBS_DIM, 1, hidden), in_keys=["observation"])


def entropy_of(dist):
    try:
        return dist.entropy()
    except NotImplementedError:
        p = dist.probs
        return -(p * p.clamp_min(1e-8).log()).sum(-1)


# ----------------------------------------------------------------------------- REINFORCE
def train_reinforce(cfg, run_id, verbose=True):
    env = make_env(seed=cfg["seed"])
    actor, logits_net = build_actor(env, cfg["hidden"])
    optim = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    value_net = mlp(OBS_DIM, 1, cfg["hidden"]) if cfg["baseline"] == "value" else None
    v_optim = torch.optim.Adam(value_net.parameters(), lr=1e-3) if value_net is not None else None
    episodes, updates, frames, baseline = [], [], 0, 0.0
    n_updates = cfg["total_frames"] // (30 * cfg["episodes_per_update"])  # approx, for logging
    u = 0
    while frames < cfg["total_frames"]:
        logps, returns, ents, obs = [], [], [], []
        for _ in range(cfg["episodes_per_update"]):
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                td = env.rollout(max_steps=40, policy=actor, break_when_any_done=True)
            frames += td.shape[0]
            rewards = td["next", "reward"].squeeze(-1)
            G, rets = 0.0, []
            for r in reversed(rewards.tolist()):
                G = r + cfg["gamma"] * G
                rets.append(G)
            rets = torch.tensor(rets[::-1])
            episodes.append({"episode": len(episodes) + 1, "frames": frames, "return": float(rewards.sum())})
            # re-evaluate log-probs with gradient
            dist = actor.get_dist(td.select("observation"))
            logps.append(dist.log_prob(td["action"]))
            ents.append(entropy_of(dist))
            returns.append(rets)
            obs.append(td["observation"])
        logp = torch.cat(logps)
        ret = torch.cat(returns)
        ent = torch.cat(ents)
        adv = ret.clone()
        if cfg["baseline"] == "mean":
            baseline = 0.9 * baseline + 0.1 * float(ret.mean()) if u else float(ret.mean())
            adv = adv - baseline
        elif cfg["baseline"] == "value":
            o = torch.cat(obs)
            v = value_net(o).squeeze(-1)
            adv = adv - v.detach()
            v_loss = torch.nn.functional.mse_loss(v, ret)  # Monte-Carlo regression, no bootstrapping
            v_optim.zero_grad()
            v_loss.backward()
            v_optim.step()
        if cfg["normalize"] and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(logp * adv).mean() - cfg["entropy_coef"] * ent.mean()
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optim.step()
        updates.append({"frames": frames, "episode": len(episodes), "loss": float(loss.item()),
                        "entropy": float(ent.mean().item())})
        if verbose and u % max(1, n_updates // 8) == 0:
            recent = [e["return"] for e in episodes[-50:]]
            print(f"[REINFORCE run {run_id}] frames {frames:6d} ep {len(episodes):5d} H {ent.mean():.2f} avgR(50) {sum(recent)/len(recent):+.2f}")
        u += 1
    return logits_net, value_net, episodes, updates


# ----------------------------------------------------------------------------- PPO / A2C
def train_ac(cfg, run_id, algo, verbose=True):
    env = make_env(seed=cfg["seed"])
    actor, logits_net = build_actor(env, cfg["hidden"])
    critic = build_critic(cfg["hidden"])
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=False)
    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"], entropy_coeff=cfg["entropy_coef"],
                              critic_coeff=cfg["critic_coef"], loss_critic_type="smooth_l1",
                              normalize_advantage=True)
    else:
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"],
                          loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    collector = Collector(env, actor, auto_register_policy_transforms=False, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"])
    rb = None
    if algo == "ppo":
        rb = ReplayBuffer(storage=LazyTensorStorage(cfg["frames_per_batch"]),
                          sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch"])
    episodes, updates, frames, it = [], [], 0, 0
    n_iters = cfg["total_frames"] // cfg["frames_per_batch"]
    for data in collector:
        frames += data.numel()
        episodes_from_batch(data, frames, episodes)
        with torch.no_grad():
            gae(data)
        if algo == "ppo":
            for _ in range(cfg["epochs"]):
                rb.extend(data.reshape(-1))
                for batch in rb:
                    out = loss_fn(batch)
                    loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", torch.zeros(()))
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
                    optim.step()
        else:
            out = loss_fn(data.reshape(-1))
            loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", torch.zeros(()))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
            optim.step()
        collector.update_policy_weights_()
        it += 1
        if "entropy" in out.keys():
            ent = float(out["entropy"].mean())
        else:  # entropy bonus disabled -> compute policy entropy for logging only
            with torch.no_grad():
                ent = float(entropy_of(actor.get_dist(data.reshape(-1).select("observation"))).mean())
        if it % max(1, n_iters // 100) == 0 or it == 1:
            updates.append({"frames": frames, "episode": len(episodes), "loss": float(loss.item()),
                            "loss_objective": float(out["loss_objective"].item()),
                            "loss_critic": float(out["loss_critic"].item()),
                            "entropy": ent})
        if verbose and it % max(1, n_iters // 8) == 0 and episodes:
            recent = [e["return"] for e in episodes[-50:]]
            print(f"[{algo.upper()} run {run_id}] frames {frames:6d} H {ent:.2f} "
                  f"avgR(50) {sum(recent)/len(recent):+.2f}")
    collector.shutdown()
    return logits_net, critic, episodes, updates


def train(algo, cfg, run_id, save_model=True, verbose=True):
    set_seed(cfg["seed"])
    t0 = time.time()
    if algo == "reinforce":
        logits_net, critic, episodes, updates = train_reinforce(cfg, run_id, verbose)
    else:
        logits_net, critic, episodes, updates = train_ac(cfg, run_id, algo, verbose)
    result = evaluate_net(logits_net, n_episodes=100)
    unseen = evaluate_net(logits_net, n_episodes=100, scenario="unseen")
    result["unseen_mean_return"] = unseen["mean_return"]
    result["unseen_std_return"] = unseen["std_return"]
    result["train_time_s"] = time.time() - t0
    result["episodes"] = len(episodes)
    save_run(algo, run_id, cfg, episodes, updates, result)
    if save_model:
        (MODEL_DIR / "pg").mkdir(parents=True, exist_ok=True)
        ck = {"actor": logits_net.state_dict(), "config": cfg, "algo": algo}
        if critic is not None:
            ck["critic"] = critic.state_dict()
        torch.save(ck, MODEL_DIR / "pg" / f"{algo}_run{run_id:02d}.pt")
    print(f"[{algo.upper()} run {run_id}] eval mean {result['mean_return']:+.2f} ± {result['std_return']:.2f} "
          f"unseen {result['unseen_mean_return']:+.2f} ({result['train_time_s']:.0f}s)")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(BASES), required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--runs", type=str, default=None, help="comma-separated subset of sweep runs")
    args = p.parse_args()
    runs = range(len(SWEEPS[args.algo])) if args.sweep else [args.run]
    if args.runs:
        runs = [int(x) for x in args.runs.split(",")]
    summary = []
    for r in runs:
        cfg = {**BASES[args.algo], **SWEEPS[args.algo][r]}
        res = train(args.algo, cfg, r)
        summary.append({"run": r, **cfg, **{k: v for k, v in res.items() if k != "returns"}})
    if args.sweep and not args.runs:
        import pandas as pd

        pd.DataFrame(summary).to_csv(LOG_DIR / f"{args.algo}_sweep.csv", index=False)
        best = max(summary, key=lambda s: s["mean_return"])
        with open(LOG_DIR / f"{args.algo}_best.json", "w") as f:
            json.dump({"run": best["run"]}, f)


if __name__ == "__main__":
    main()
