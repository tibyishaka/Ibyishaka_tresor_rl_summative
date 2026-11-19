"""SmallholderFarmEnv - a mission-based Gymnasium environment.

The agent is a digital farm-advisory service (delivered over SMS / USSD / voice and
through the farmer's cooperative) that decides, week by week, what a Rwandan
smallholder maize farmer should do during one season so that losses are minimised,
crop disease is caught early and the harvest is sold at a fair price.

One step = one week. A season lasts at most ``MAX_WEEKS`` weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------------------
# Actions (discrete). Each one maps to a concrete advisory message sent to the farmer.
# --------------------------------------------------------------------------------------
ACTIONS = [
    "MONITOR",        # 0 - no intervention this week ("keep observing")
    "IRRIGATE",       # 1 - water the plot (watering can / small pump)
    "FERTILIZE",      # 2 - apply NPK/urea top-dressing
    "SMS_DIAGNOSE",   # 3 - farmer sends leaf photo / symptom code, ML model returns diagnosis
    "TREAT",          # 4 - apply recommended fungicide / bio-control
    "HARVEST",        # 5 - harvest the crop
    "STORE_COOP",     # 6 - move produce from home to cooperative hermetic storage
    "SELL_LOCAL",     # 7 - sell home stock to a local trader (immediate, ~70% of market price)
    "SELL_COOP",      # 8 - sell cooperative stock through aggregated contract (full price)
]
N_ACTIONS = len(ACTIONS)

OBS_NAMES = [
    "week",              # 0  season progress
    "growth_stage",      # 1  crop maturity (1 = ready)
    "crop_health",       # 2  canopy health / vigour (NDVI-like)
    "soil_moisture",     # 3  soil moisture sensor
    "rain_forecast",     # 4  probability of rain next week (weather API)
    "visible_symptoms",  # 5  noisy farmer-reported symptoms (only obvious once disease is advanced)
    "diagnosed_level",   # 6  last ML diagnosis result (infection severity)
    "diag_staleness",    # 7  weeks since last diagnosis / 10
    "market_price",      # 8  current price index / 2 (market price API, e.g. eSoko)
    "demand_forecast",   # 9  4-week price trend forecast (0.5 = flat)
    "home_stock",        # 10 produce held at home (t / 1.5)
    "coop_stock",        # 11 produce held in cooperative storage (t / 1.5)
    "harvested",         # 12 flag
    "fertilizer_active", # 13 remaining fertilizer effect / 4
]
OBS_DIM = len(OBS_NAMES)

MAX_WEEKS = 40
VALUE_PER_TONNE = 10.0   # revenue units per tonne at price index 1.0 (1 unit ~ 25,000 RWF)
COSTS = {
    "IRRIGATE": 0.15,
    "FERTILIZE": 0.40,
    "SMS_DIAGNOSE": 0.05,
    "TREAT": 0.50,
    "HARVEST": 0.20,
    "STORE_COOP": 0.10,
}
HOME_LOSS_RATE = 0.07    # weekly post-harvest loss at home (pests, moulds, aflatoxin)
COOP_LOSS_RATE = 0.01    # weekly loss in hermetic cooperative storage
COOP_FEE = 0.02          # storage fee per tonne per week
LOCAL_PRICE_FACTOR = 0.70
INVALID_PENALTY = 0.2
UNSOLD_PENALTY = 1.0     # per tonne still unsold when the season ends (food waste)
CROP_FAILURE_PENALTY = 3.0
HEALTH_LOSS_WEIGHT = 1.0
GROWTH_REWARD = 2.0      # shaping: reward per unit of crop growth achieved
HARVEST_BONUS = 1.5      # food-security bonus per tonne safely harvested
MIN_HARVEST_GROWTH = 0.7 # immature maize cannot be harvested for grain


@dataclass
class FarmState:
    week: int = 0
    growth: float = 0.05
    health: float = 1.0
    moisture: float = 0.5
    infection: float = 0.0
    symptoms: float = 0.0
    diag_level: float = 0.0
    weeks_since_diag: int = 10
    rain_phase: float = 0.0
    rained: bool = False
    price_trough: int = 21
    price_noise: float = 0.0
    home_stock: float = 0.0
    coop_stock: float = 0.0
    harvested: bool = False
    fert_left: int = 0
    fert_apps: int = 0
    yield_bonus: float = 0.0
    weeks_mature: int = 0
    revenue: float = 0.0
    costs: float = 0.0
    lost_tonnes: float = 0.0
    sold_tonnes: float = 0.0
    price_history: list = field(default_factory=list)


class SmallholderFarmEnv(gym.Env):
    """Weekly farm-advisory MDP for a smallholder maize plot."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, scenario: str = "train"):
        super().__init__()
        assert scenario in ("train", "unseen")
        self.scenario = scenario
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.s = FarmState()
        self._renderer = None
        self.last_action: int | None = None
        self.last_reward: float = 0.0
        self.episode_return: float = 0.0

    # ------------------------------------------------------------------ dynamics helpers
    def _rain_prob(self, week: int) -> float:
        # bimodal rainy seasons (Rwanda season A/B) with random phase per episode
        return float(np.clip(0.4 + 0.3 * math.sin(2 * math.pi * week / 20 + self.s.rain_phase), 0.05, 0.9))

    def _base_price(self, week: int) -> float:
        """Price index: falls towards the harvest glut, peaks in the lean season, then
        drops again as the next season's harvest approaches."""
        t = self.s.price_trough
        if week <= t:
            return 1.0 - 0.25 * week / t
        if week <= t + 12:
            return 0.75 + 0.70 * (week - t) / 12
        return 1.45 - 0.05 * (week - t - 12)

    def price(self, week: int | None = None) -> float:
        week = self.s.week if week is None else week
        return max(0.3, self._base_price(week) + self.s.price_noise)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        scenario = (options or {}).get("scenario", self.scenario)
        s = FarmState()
        s.rain_phase = float(rng.uniform(0, 2 * math.pi))
        if scenario == "train":
            s.health = float(rng.uniform(0.85, 1.0))
            s.moisture = float(rng.uniform(0.3, 0.7))
            s.price_trough = int(rng.integers(18, 25))
            s.infection = 0.0 if rng.random() > 0.1 else float(rng.uniform(0.05, 0.15))
        else:  # unseen initial states - outside the training distribution
            s.health = float(rng.uniform(0.6, 0.85))
            s.moisture = float(rng.choice([rng.uniform(0.08, 0.3), rng.uniform(0.7, 0.92)]))
            s.price_trough = int(rng.choice([rng.integers(14, 18), rng.integers(25, 29)]))
            s.infection = float(rng.uniform(0.1, 0.3)) if rng.random() < 0.5 else 0.0
        s.price_noise = float(rng.normal(0, 0.03))
        s.price_history = []
        self.s = s
        s.price_history.append(self.price())
        self._update_symptoms()
        self.last_action, self.last_reward, self.episode_return = None, 0.0, 0.0
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def _update_symptoms(self):
        s, rng = self.s, self.np_random
        if s.harvested:
            s.symptoms = 0.0
        elif s.infection > 0.35:
            s.symptoms = float(np.clip(s.infection + rng.normal(0, 0.1), 0, 1))
        else:
            s.symptoms = float(np.clip(abs(rng.normal(0, 0.08)), 0, 1))

    def _obs(self) -> np.ndarray:
        s = self.s
        trend = self._base_price(min(MAX_WEEKS, s.week + 4)) - self._base_price(s.week)
        obs = np.array(
            [
                s.week / MAX_WEEKS,
                s.growth,
                s.health,
                s.moisture,
                self._rain_prob(s.week + 1),
                s.symptoms,
                s.diag_level,
                min(s.weeks_since_diag, 10) / 10.0,
                self.price() / 2.0,
                0.5 + trend,
                s.home_stock / 1.5,
                s.coop_stock / 1.5,
                float(s.harvested),
                s.fert_left / 4.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _info(self) -> dict:
        s = self.s
        return {
            "week": s.week,
            "revenue": s.revenue,
            "costs": s.costs,
            "profit": s.revenue - s.costs,
            "sold_tonnes": s.sold_tonnes,
            "lost_tonnes": s.lost_tonnes,
            "price": self.price(),
        }

    def step(self, action: int):
        action = int(action)
        s, rng = self.s, self.np_random
        name = ACTIONS[action]
        reward = 0.0
        invalid = False
        health_before = s.health

        # ---------------- 1. apply advisory action
        if name in COSTS:
            cost = COSTS[name]
        else:
            cost = 0.0

        if name == "IRRIGATE":
            if s.harvested:
                invalid = True
            else:
                s.moisture = min(1.0, s.moisture + 0.35)
        elif name == "FERTILIZE":
            if s.harvested or s.fert_apps >= 3:
                invalid = True
            else:
                s.fert_apps += 1
                s.fert_left = 4
                if s.growth < 0.7:
                    s.yield_bonus += 0.1
        elif name == "SMS_DIAGNOSE":
            if s.harvested:
                invalid = True
            else:
                s.diag_level = float(np.clip(s.infection + rng.normal(0, 0.03), 0, 1))
                s.weeks_since_diag = 0
        elif name == "TREAT":
            if s.harvested:
                invalid = True
            elif s.infection > 0:
                s.infection = max(0.0, s.infection - 0.7)
            else:
                reward -= 0.1  # unnecessary chemical use (cost + environmental penalty)
        elif name == "HARVEST":
            if s.harvested or s.growth < MIN_HARVEST_GROWTH:
                invalid = True
            else:
                qty = (s.growth ** 1.5) * s.health * (1.0 + s.yield_bonus)
                s.home_stock += qty
                reward += HARVEST_BONUS * qty
                s.harvested = True
                s.infection = 0.0
        elif name == "STORE_COOP":
            if s.home_stock <= 1e-6:
                invalid = True
            else:
                s.coop_stock += s.home_stock
                s.home_stock = 0.0
        elif name == "SELL_LOCAL":
            if s.home_stock <= 1e-6:
                invalid = True
            else:
                rev = s.home_stock * self.price() * LOCAL_PRICE_FACTOR * VALUE_PER_TONNE
                reward += rev
                s.revenue += rev
                s.sold_tonnes += s.home_stock
                s.home_stock = 0.0
        elif name == "SELL_COOP":
            if s.coop_stock <= 1e-6:
                invalid = True
            else:
                rev = s.coop_stock * self.price() * VALUE_PER_TONNE
                reward += rev
                s.revenue += rev
                s.sold_tonnes += s.coop_stock
                s.coop_stock = 0.0

        if invalid:
            reward -= INVALID_PENALTY
            cost = 0.0
        reward -= cost
        s.costs += cost

        # ---------------- 2. one week of nature
        s.week += 1
        s.weeks_since_diag += 1
        s.rained = bool(rng.random() < self._rain_prob(s.week))
        if not s.harvested:
            s.moisture -= 0.12 + (0.05 if s.growth > 0.5 else 0.0)
            if s.rained:
                s.moisture += 0.35
            s.moisture = float(np.clip(s.moisture, 0.0, 1.0))

            # crop growth
            if s.moisture < 0.25:
                water_f = 0.5
            elif s.moisture > 0.85:
                water_f = 0.8
            else:
                water_f = 1.0
            fert_f = 1.4 if s.fert_left > 0 else 1.0
            new_growth = min(1.0, s.growth + 0.045 * s.health * water_f * fert_f)
            reward += GROWTH_REWARD * (new_growth - s.growth)
            s.growth = new_growth
            if s.growth >= 1.0:
                s.weeks_mature += 1

            # stresses
            if s.moisture < 0.25:
                s.health -= 0.05
            if s.weeks_mature > 2:
                s.health -= 0.06  # over-ripe crop left in the field: pests, lodging, theft

            # disease
            if s.infection <= 0:
                p_onset = 0.06 + (0.08 if s.moisture > 0.8 else 0.0)
                if rng.random() < p_onset:
                    s.infection = float(rng.uniform(0.05, 0.12))
            else:
                s.infection = min(1.0, s.infection + 0.1 + 0.05 * rng.random())
                s.health -= 0.12 * s.infection
            if s.infection <= 0 and 0.25 <= s.moisture <= 0.85:
                s.health += 0.02
            s.health = float(np.clip(s.health, 0.0, 1.0))
        else:
            s.moisture = float(np.clip(s.moisture - 0.05 + (0.2 if s.rained else 0.0), 0, 1))

        if s.fert_left > 0:
            s.fert_left -= 1

        # post-harvest losses and storage fees
        lost_home = s.home_stock * HOME_LOSS_RATE
        lost_coop = s.coop_stock * COOP_LOSS_RATE
        s.home_stock -= lost_home
        s.coop_stock -= lost_coop
        s.lost_tonnes += lost_home + lost_coop
        fee = COOP_FEE * s.coop_stock
        reward -= fee
        s.costs += fee

        # market
        s.price_noise = float(np.clip(0.8 * s.price_noise + rng.normal(0, 0.03), -0.15, 0.15))
        s.price_history.append(self.price())
        self._update_symptoms()

        # shaping: penalise crop-health loss caused during the week
        health_drop = max(0.0, health_before - s.health) if not s.harvested else 0.0
        reward -= HEALTH_LOSS_WEIGHT * health_drop

        # ---------------- 3. termination
        terminated = False
        if not s.harvested and s.health <= 0.1:
            terminated = True
            reward -= CROP_FAILURE_PENALTY
        if s.harvested and s.home_stock + s.coop_stock <= 1e-6:
            terminated = True  # whole harvest sold - season complete
        if not terminated and s.week >= MAX_WEEKS:
            # the season genuinely ends: unsold produce is wasted (true terminal state,
            # the week is part of the observation so no bootstrapping beyond it)
            terminated = True
            reward -= UNSOLD_PENALTY * (s.home_stock + s.coop_stock)
        truncated = False

        self.last_action = action
        self.last_reward = float(reward)
        self.episode_return += float(reward)
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import FarmRenderer

            self._renderer = FarmRenderer(visible=self.render_mode == "human")
        frame = self._renderer.draw(self)
        if self.render_mode == "rgb_array":
            return frame
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
