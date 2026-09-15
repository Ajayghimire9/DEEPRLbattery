import numpy as np
from gridmind.env import MicrogridEnv


def test_reset_is_finite_and_bounded():
    env = MicrogridEnv()
    state, _ = env.reset(seed=123)
    assert np.isfinite(state).all()
    assert 0.0 <= state[0] <= 1.0


def test_soc_remains_safe_under_extreme_actions():
    env = MicrogridEnv()
    env.reset(seed=1)
    for action in (-1.0, 1.0, 1.0, -1.0) * 10:
        state, _, done, _ = env.step(action)
        assert 0.0 <= state[0] <= 1.0
        if done:
            env.reset(seed=1)
