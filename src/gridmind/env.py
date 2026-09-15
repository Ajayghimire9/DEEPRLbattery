from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    battery_capacity_kwh: float = 15.0
    max_power_kw: float = 5.0
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    dt_hours: float = 1.0
    episode_hours: int = 24
    seed: int = 42


class MicrogridEnv:
    """Small deterministic simulator with explicit safety constraints.

    Action is battery power in [-1, 1]: negative=charge, positive=discharge.
    The environment clips infeasible actions and exposes the clipping penalty.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        if (
            self.cfg.battery_capacity_kwh <= 0
            or self.cfg.dt_hours <= 0
            or self.cfg.max_power_kw <= 0
        ):
            raise ValueError("Capacity, timestep and power must be positive")
        if not 0 < self.cfg.charge_efficiency <= 1 or not 0 < self.cfg.discharge_efficiency <= 1:
            raise ValueError("Efficiencies must be in (0, 1]")
        if self.cfg.episode_hours < 1:
            raise ValueError("Episode length must be positive")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.soc = 0.5
        self.load = 0.0
        self.pv = 0.0
        self.price = 0.0

    def _profile(self, t: int) -> tuple[float, float, float]:
        hour = t % 24
        load = 3.0 + 1.2 * np.sin((hour - 7) * np.pi / 12) + self.rng.normal(0, 0.12)
        pv = max(0.0, 5.0 * np.sin((hour - 6) * np.pi / 12))
        price = 0.12 + 0.08 * (1 + np.sin((hour - 17) * np.pi / 12))
        return max(load, 0.2), pv, price

    def _obs(self) -> np.ndarray:
        return np.asarray([self.soc, self.load / 6.0, self.pv / 5.0, self.price], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.soc = 0.5
        self.load, self.pv, self.price = self._profile(self.t)
        return self._obs(), {"soc": self.soc}

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        if self.t >= self.cfg.episode_hours:
            raise RuntimeError("Episode has ended; call reset before stepping")
        if not np.isfinite(action):
            raise ValueError("Action must be finite")
        requested = float(np.clip(action, -1.0, 1.0)) * self.cfg.max_power_kw
        cfg = self.cfg
        # Positive power supplies the AC bus. Charge/discharge losses belong
        # in the battery energy balance, not in the sign of grid demand.
        if requested < 0:
            available = (
                (1 - self.soc) * cfg.battery_capacity_kwh / (cfg.charge_efficiency * cfg.dt_hours)
            )
            feasible = -min(-requested, available)
            self.soc += -feasible * cfg.charge_efficiency * cfg.dt_hours / cfg.battery_capacity_kwh
        else:
            available = (
                self.soc * cfg.battery_capacity_kwh * cfg.discharge_efficiency / cfg.dt_hours
            )
            feasible = min(requested, available)
            self.soc -= (
                feasible * cfg.dt_hours / (cfg.discharge_efficiency * cfg.battery_capacity_kwh)
            )
        self.soc = float(np.clip(self.soc, 0, 1))
        net_grid = self.load - self.pv - feasible
        energy_cost = max(net_grid, 0.0) * self.price * self.cfg.dt_hours
        export_credit = max(-net_grid, 0.0) * self.price * 0.5 * self.cfg.dt_hours
        constraint_penalty = abs(requested - feasible) * 0.02
        terminal_penalty = 0.0
        self.t += 1
        done = self.t >= self.cfg.episode_hours
        if done:
            terminal_penalty = abs(self.soc - 0.5) * 0.2
        reward = -(energy_cost - export_credit + constraint_penalty + terminal_penalty)
        self.load, self.pv, self.price = self._profile(self.t)
        info = {
            "soc": self.soc,
            "grid_kw": net_grid,
            "cost": energy_cost,
            "constraint_penalty": constraint_penalty,
        }
        return self._obs(), float(reward), done, info
