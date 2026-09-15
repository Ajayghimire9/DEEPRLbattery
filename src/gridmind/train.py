from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .agent import DoubleDQN, Transition
from .env import MicrogridEnv


def train(episodes: int = 100, seed: int = 42) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = MicrogridEnv()
    agent = DoubleDQN(obs_dim=4)
    returns: list[float] = []
    epsilon = 1.0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action_idx = agent.select_action(state, epsilon)
            action = (action_idx / 10.0) - 1.0
            next_state, reward, done, _ = env.step(action)
            agent.replay.add(Transition(state, action_idx, reward, next_state, done))
            agent.train_step()
            state = next_state
            total += reward
        returns.append(total)
        epsilon = max(0.05, epsilon * 0.985)
        if (episode + 1) % 10 == 0:
            agent.sync_target()
    return {"episodes": episodes, "mean_return": float(np.mean(returns[-20:])), "best_return": float(np.max(returns)), "epsilon": epsilon}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="artifacts/training_summary.json")
    args = parser.parse_args()
    summary = train(args.episodes)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
