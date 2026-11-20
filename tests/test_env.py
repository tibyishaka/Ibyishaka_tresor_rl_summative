import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTIONS, MAX_WEEKS, OBS_DIM, SmallholderFarmEnv


def test_gymnasium_api():
    check_env(SmallholderFarmEnv(), skip_render_check=True)


def test_spaces_and_bounds():
    env = SmallholderFarmEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and env.action_space.n == len(ACTIONS)
    for _ in range(MAX_WEEKS):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break


def test_season_always_ends():
    env = SmallholderFarmEnv()
    env.reset(seed=1)
    for t in range(MAX_WEEKS):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert term and t < MAX_WEEKS


def test_immature_harvest_invalid():
    env = SmallholderFarmEnv()
    env.reset(seed=2)
    _, r, *_ = env.step(5)  # HARVEST at week 0
    assert not env.s.harvested and r < 0


def test_unseen_scenario_differs():
    a, b = SmallholderFarmEnv(scenario="train"), SmallholderFarmEnv(scenario="unseen")
    ha = [a.reset(seed=i)[0][2] for i in range(50)]
    hb = [b.reset(seed=i)[0][2] for i in range(50)]
    assert np.mean(hb) < np.mean(ha)
