import numpy as np
import pytest

from gridmind.env import EnvConfig, MicrogridEnv


def test_discharge_reduces_grid_import_and_cannot_empty_below_zero():
    env = MicrogridEnv()
    env.reset(7)
    load, pv, initial = env.load, env.pv, env.soc
    _, _, _, info = env.step(1)
    delivered = (initial - env.soc) * env.cfg.battery_capacity_kwh * env.cfg.discharge_efficiency
    assert info["grid_kw"] == pytest.approx(load - pv - delivered)
    for _ in range(23):
        env.step(1)
        assert 0 <= env.soc <= 1
    with pytest.raises(RuntimeError):
        env.step(0)


def test_charge_draws_grid_power_and_capacity_is_respected():
    env = MicrogridEnv()
    env.reset(7)
    load, pv = env.load, env.pv
    _, _, _, info = env.step(-1)
    assert info["grid_kw"] > load - pv
    for _ in range(23):
        env.step(-1)
        assert 0 <= env.soc <= 1
    with pytest.raises(ValueError):
        MicrogridEnv(EnvConfig(charge_efficiency=0))
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.nan)
