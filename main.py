"""Run the best trained agent in the 3D OpenGL farm simulation.

Examples
    uv run main.py                         # best overall model, GUI window
    uv run main.py --algo dqn --episodes 2 # a specific method
    uv run main.py --random                # random-action demo (no model)
    uv run main.py --record assets/agent.mp4 --headless
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import ACTIONS, SmallholderFarmEnv
from training.common import N_ACTIONS, OBS_DIM, mlp

ROOT = Path(__file__).resolve().parent
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def load_policy(algo: str):
    path = ROOT / "models" / ("dqn" if algo == "dqn" else "pg") / f"best_{algo}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train first (see README).")
    ck = torch.load(path)
    if algo == "dqn":
        net = mlp(OBS_DIM, N_ACTIONS, ck["config"]["hidden"], act=torch.nn.ReLU)
        net.load_state_dict(ck["state_dict"])
    else:
        net = mlp(OBS_DIM, N_ACTIONS, ck["config"]["hidden"])
        net.load_state_dict(ck["actor"])
    net.eval()
    return lambda obs: int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1).item())


def best_algo() -> str:
    summary = ROOT / "logs" / "results_summary.json"
    if summary.exists():
        gen = json.load(open(summary))["generalization"]
        return max(gen, key=lambda a: gen[a]["train_mean"])
    return "ppo"


def main():
    p = argparse.ArgumentParser(description="Smallholder farm advisory RL demo")
    p.add_argument("--algo", choices=ALGOS + ["best"], default="best")
    p.add_argument("--random", action="store_true", help="random actions (environment demo)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", choices=["train", "unseen"], default="train")
    p.add_argument("--record", type=str, default=None, help="save an .mp4/.gif of the run")
    p.add_argument("--headless", action="store_true", help="no window (use with --record)")
    p.add_argument("--fps", type=float, default=3.0)
    args = p.parse_args()

    algo = best_algo() if args.algo == "best" else args.algo
    policy = (lambda obs: int(np.random.randint(N_ACTIONS))) if args.random else load_policy(algo)
    name = "RANDOM" if args.random else algo.upper()
    print(f"Agent: {name} | scenario: {args.scenario}")

    env = SmallholderFarmEnv(render_mode="rgb_array" if args.headless else "human", scenario=args.scenario)
    if args.headless:
        env.render_mode = "rgb_array"
    frames = []
    np.random.seed(args.seed)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, R = False, 0.0
        while not done:
            a = policy(obs)
            obs, r, term, trunc, info = env.step(a)
            R += r
            done = term or trunc
            # the advisory as it would be serialised to the SMS/USSD gateway
            print(json.dumps({"week": info["week"], "advice": ACTIONS[a], "reward": round(r, 3),
                              "price": round(info["price"], 2), "profit": round(info["profit"], 2)}))
            if args.record or args.headless:
                frame = env.render() if env.render_mode == "rgb_array" else env._renderer.draw(env)
                frames.append(frame)
            if not args.headless:
                time.sleep(1.0 / args.fps)
        print(f"Episode {ep + 1}: return {R:+.2f} | profit {info['profit']:+.2f} | "
              f"sold {info['sold_tonnes']:.2f} t | lost {info['lost_tonnes']:.2f} t")
    env.close()

    if args.record and frames:
        import imageio

        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".gif":
            imageio.mimsave(out, frames, duration=1.0 / args.fps)
        else:
            imageio.mimsave(out, frames, fps=args.fps, macro_block_size=8)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
