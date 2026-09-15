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
        requested = float(np.clip(action, -1.0, 1.0)) * self.cfg.max_power_kw
        available_charge = (1.0 - self.soc) * self.cfg.battery_capacity_kwh / self.cfg.dt_hours
        available_discharge = self.soc * self.cfg.battery_capacity_kwh / self.cfg.dt_hours
        if requested < 0:
            feasible = -min(abs(requested), available_charge)
            grid_battery = feasible * self.cfg.charge_efficiency
            self.soc += abs(feasible) * self.cfg.charge_efficiency * self.cfg.dt_hours / self.cfg.battery_capacity_kwh
        else:
            feasible = min(requested, available_discharge)
            grid_battery = feasible / self.cfg.discharge_efficiency
            self.soc -= feasible * self.cfg.dt_hours / (self.cfg.battery_capacity_kwh * self.cfg.discharge_efficiency)

        net_grid = self.load - self.pv + grid_battery
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
        info = {"soc": self.soc, "grid_kw": net_grid, "cost": energy_cost, "constraint_penalty": constraint_penalty}
        return self._obs(), float(reward), done, info
