"""Aggregate sweep results, pick best models and produce the report figures.

Usage:  uv run python -m training.analysis
Outputs: logs/<algo>_sweep.csv, models/*/best_*.pt, assets/*.png, logs/results_summary.json
"""

from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from environment.custom_env import SmallholderFarmEnv
from training.common import LOG_DIR, MODEL_DIR, N_ACTIONS, OBS_DIM, evaluate_net, mlp

ROOT = LOG_DIR.parent
ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "axes.titleweight": "bold", "axes.titlesize": 10, "lines.linewidth": 2,
})


def smooth(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    w = max(1, min(w, len(x) // 5 or 1))
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def load_sweep(algo):
    rows = []
    for f in sorted((LOG_DIR / algo).glob("run*_summary.json")):
        d = json.load(open(f))
        run = int(f.name[3:5])
        rows.append({"run": run, **d["config"], **d["result"]})
    df = pd.DataFrame(rows).sort_values("run")
    df.to_csv(LOG_DIR / f"{algo}_sweep.csv", index=False)
    return df


def load_net(algo, run):
    if algo == "dqn":
        ck = torch.load(MODEL_DIR / "dqn" / f"dqn_run{run:02d}.pt")
        net = mlp(OBS_DIM, N_ACTIONS, ck["config"]["hidden"], act=torch.nn.ReLU)
        net.load_state_dict(ck["state_dict"])
    else:
        ck = torch.load(MODEL_DIR / "pg" / f"{algo}_run{run:02d}.pt")
        net = mlp(OBS_DIM, N_ACTIONS, ck["config"]["hidden"])
        net.load_state_dict(ck["actor"])
    return net.eval()


def converge_episode(returns, frac=0.9, w=100):
    """First episode after which the moving average stays >= frac of its final plateau
    (measured relative to the initial level)."""
    ma = smooth(returns, w)
    final = ma[-w:].mean()
    start = ma[: w].mean()
    thr = start + frac * (final - start)
    above = ma >= thr
    for i in range(len(ma)):
        if above[i:].mean() > 0.95:
            return i + 1, thr
    return len(ma), thr


def expert_action(env):
    """Hand-crafted agronomist heuristic (reference only; reads the diagnosis log)."""
    s, o = env.s, env._obs()
    if not s.harvested:
        if o[3] < 0.3:
            return 1
        if s.weeks_since_diag >= 3:
            return 3
        if (s.diag_level > 0.05 and s.weeks_since_diag == 1) or o[5] > 0.4:
            return 4
        if s.fert_apps < 2 and 0.2 < o[1] < 0.6 and s.fert_left == 0:
            return 2
        return 5 if o[1] >= 0.99 else 0
    if o[10] > 0:
        return 6
    return 8 if (o[9] <= 0.5 + 1e-3 or s.week >= 39) else 0


def run_policy(policy, scenario, n=200, seed0=20_000):
    env = SmallholderFarmEnv(scenario=scenario)
    rets = []
    for ep in range(n):
        env.reset(seed=seed0 + ep)
        done, R = False, 0.0
        while not done:
            _, r, te, tr, _ = env.step(policy(env))
            R += r
            done = te or tr
        rets.append(R)
    return float(np.mean(rets))


def baselines(n=100):
    env = SmallholderFarmEnv()
    out = {}
    for scen in ("train", "unseen"):
        env = SmallholderFarmEnv(scenario=scen)
        rets = []
        for ep in range(n):
            env.reset(seed=10_000 + ep)
            env.action_space.seed(ep)
            done, R = False, 0.0
            while not done:
                _, r, te, tr, _ = env.step(env.action_space.sample())
                R += r
                done = te or tr
            rets.append(R)
        out[scen] = float(np.mean(rets))
        out["expert_" + scen] = run_policy(expert_action, scen)
    return out


def main():
    ASSETS.mkdir(exist_ok=True)
    sweeps = {a: load_sweep(a) for a in ALGOS}
    best = {a: int(sweeps[a].loc[sweeps[a]["mean_return"].idxmax(), "run"]) for a in ALGOS}
    print("best runs:", best)
    for a, r in best.items():
        src = MODEL_DIR / ("dqn" if a == "dqn" else "pg") / f"{a}_run{r:02d}.pt"
        shutil.copy(src, src.parent / f"best_{a}.pt")

    eps = {a: pd.read_csv(LOG_DIR / a / f"run{best[a]:02d}_episodes.csv") for a in ALGOS}
    upd = {a: pd.read_csv(LOG_DIR / a / f"run{best[a]:02d}_updates.csv") for a in ALGOS}

    # ---------------- Fig 1: cumulative reward per episode (best model of each method)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        y = eps[a]["return"].to_numpy()
        ax.plot(y, color=COLOR[a], alpha=0.18, linewidth=0.8)
        ax.plot(smooth(y, 100), color=COLOR[a], label="100-episode moving avg")
        ax.set_title(f"{LABEL[a]} (run {best[a]})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode return")
        ax.axhline(0, color=INK2, linewidth=0.8)
    fig.suptitle("Cumulative reward per episode - best configuration of each method", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_cumulative_rewards.png")
    plt.close(fig)

    # ---------------- Fig 2: training stability - DQN loss, PG entropy
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    u = upd["dqn"]
    axes[0].plot(u["frames"] / 1000, smooth(u["loss"], 10), color=COLOR["dqn"])
    axes[0].set_title("DQN objective (smooth-L1 TD loss)")
    axes[0].set_xlabel("Environment steps (k)")
    axes[0].set_ylabel("Loss")
    for a in ["reinforce", "ppo", "a2c"]:
        u = upd[a]
        axes[1].plot(u["frames"] / 1000, smooth(u["entropy"], 10), color=COLOR[a], label=LABEL[a])
    axes[1].axhline(np.log(N_ACTIONS), color=INK2, linestyle="--", linewidth=0.8)
    axes[1].text(60, np.log(N_ACTIONS) + 0.04, "max entropy ln(9)", color=INK2, fontsize=7)
    axes[1].set_ylim(0, np.log(N_ACTIONS) + 0.25)
    axes[1].set_title("Policy entropy (PG methods)")
    axes[1].set_xlabel("Environment steps (k)")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_training_stability.png")
    plt.close(fig)

    # ---------------- Fig 3: episodes to converge
    conv = {}
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), sharey=True)
    for ax, a in zip(axes, ALGOS):
        y = eps[a]["return"].to_numpy()
        ep_c, thr = converge_episode(y)
        conv[a] = {"episode": int(ep_c), "frames": int(eps[a]["frames"].iloc[min(ep_c, len(y)) - 1]),
                   "threshold": float(thr), "plateau": float(smooth(y, 100)[-100:].mean())}
        ax.plot(smooth(y, 100), color=COLOR[a])
        ax.axvline(ep_c, color=INK2, linestyle="--", linewidth=1)
        ax.text(ep_c, ax.get_ylim()[0] if False else np.nanmin(smooth(y, 100)), f" ep {ep_c}", color=INK, fontsize=7)
        ax.set_title(LABEL[a])
        ax.set_xlabel("Episode")
    axes[0].set_ylabel("Return (100-ep MA)")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_convergence.png")
    plt.close(fig)

    # ---------------- Fig 4: generalisation to unseen initial states
    base = baselines()
    gen = {}
    for a in ALGOS:
        net = load_net(a, best[a])
        tr = evaluate_net(net, 200, "train", seed0=20_000)
        un = evaluate_net(net, 200, "unseen", seed0=20_000)
        gen[a] = {"train_mean": tr["mean_return"], "train_std": tr["std_return"],
                  "unseen_mean": un["mean_return"], "unseen_std": un["std_return"],
                  "train_profit": tr["mean_profit"], "unseen_profit": un["mean_profit"],
                  "train_lost_t": tr["mean_lost_t"], "unseen_lost_t": un["mean_lost_t"],
                  "unseen_returns": un["returns"]}
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(ALGOS))
    w = 0.38
    ax = axes[0]
    tm = [gen[a]["train_mean"] for a in ALGOS]
    um = [gen[a]["unseen_mean"] for a in ALGOS]
    ax.bar(x - w / 2 - 0.01, tm, w, color=[COLOR[a] for a in ALGOS], label="Training distribution")
    ax.bar(x + w / 2 + 0.01, um, w, color=[COLOR[a] for a in ALGOS], alpha=0.45, hatch="//", edgecolor="white",
           label="Unseen initial states")
    for i in range(len(ALGOS)):
        ax.text(x[i] - w / 2, tm[i] + 0.2, f"{tm[i]:.1f}", ha="center", fontsize=7, color=INK)
        ax.text(x[i] + w / 2, um[i] + 0.2, f"{um[i]:.1f}", ha="center", fontsize=7, color=INK)
    ax.axhline(base["unseen"], color=INK2, linestyle=":", linewidth=1)
    ax.text(-0.45, base["unseen"] + 0.3, "random agent (unseen)", color=INK2, fontsize=7)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, [LABEL[a] for a in ALGOS])
    ax.set_ylabel("Mean return (200 eps)")
    ax.set_title("Seen vs unseen initial states")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#8a8984", label="Training distribution"),
                       Patch(facecolor="#8a8984", alpha=0.45, hatch="//", edgecolor="white", label="Unseen initial states")],
              frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.set_ylim(min(base["unseen"], 0) - 1.5, max(tm) + 3.5)
    ax = axes[1]
    bp = ax.boxplot([gen[a]["unseen_returns"] for a in ALGOS], patch_artist=True, widths=0.5,
                    medianprops={"color": INK})
    for patch, a in zip(bp["boxes"], ALGOS):
        patch.set_facecolor(COLOR[a])
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, 5), [LABEL[a] for a in ALGOS])
    ax.set_title("Return distribution on unseen states")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_generalization.png")
    plt.close(fig)

    # ---------------- Fig 5: hyper-parameter sweep overview
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4), sharey=True)
    for ax, a in zip(axes, ALGOS):
        df = sweeps[a]
        cols = [COLOR[a] if r == best[a] else "#c3c2b7" for r in df["run"]]
        ax.bar(df["run"], df["mean_return"], color=cols)
        ax.set_title(LABEL[a])
        ax.set_xlabel("Run")
        ax.set_xticks(df["run"])
        ax.axhline(0, color=INK2, linewidth=0.8)
    axes[0].set_ylabel("Mean eval return")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_sweep_overview.png")
    plt.close(fig)

    summary = {
        "best_runs": best,
        "convergence": conv,
        "generalization": {a: {k: v for k, v in g.items() if k != "unseen_returns"} for a, g in gen.items()},
        "random_baseline": base,
    }
    json.dump(summary, open(LOG_DIR / "results_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
