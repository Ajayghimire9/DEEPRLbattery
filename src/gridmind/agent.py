from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 21) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.data = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)


class DoubleDQN:
    """Double-DQN agent with a target network and gradient clipping."""

    def __init__(self, obs_dim: int, action_dim: int = 21, lr: float = 3e-4, gamma: float = 0.99) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.online = QNetwork(obs_dim, action_dim)
        self.target = QNetwork(obs_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.loss_fn = nn.SmoothL1Loss()
        self.steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    def train_step(self, batch_size: int = 64) -> float | None:
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        states = torch.tensor(np.stack([x.state for x in batch]), dtype=torch.float32)
        actions = torch.tensor([x.action for x in batch], dtype=torch.long)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32)
        dones = torch.tensor([x.done for x in batch], dtype=torch.float32)
        q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        self.steps += 1
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
