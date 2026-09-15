# GridMind — Safe Reinforcement Learning for Microgrid Energy Optimization

**Production-oriented reinforcement-learning platform for battery dispatch under energy, efficiency, and operational constraints.**

This project evolves the original microgrid DDQN research prototype into a reproducible ML engineering system: a constrained simulator, Double DQN learner, deterministic training pipeline, safety-focused tests, packaging, containerization, and CI.

## Architecture

```text
Energy / PV / Price Profiles
          │
          ▼
┌────────────────────────────┐
│ Constrained Microgrid Env  │
│ • SOC dynamics             │
│ • charge/discharge limits  │
│ • efficiency losses        │
│ • constraint penalties     │
└─────────────┬──────────────┘
              │ state
              ▼
┌────────────────────────────┐
│ Double DQN Agent           │
│ • online Q-network         │
│ • target network           │
│ • experience replay        │
│ • Huber loss               │
│ • gradient clipping        │
└─────────────┬──────────────┘
              │ policy
              ▼
      Battery Dispatch
              │
              ▼
      Reward / Safety KPIs

CI → Ruff → Pytest → Docker
```

## Engineering features

- **Safe action execution:** infeasible battery commands are clipped instead of violating state constraints.
- **Double DQN:** separates action selection from target evaluation to reduce Q-value overestimation.
- **Experience replay:** decorrelates sequential transitions and improves sample efficiency.
- **Target-network synchronization:** stabilizes temporal-difference learning.
- **Huber loss + gradient clipping:** robust optimization for noisy RL targets.
- **Deterministic experiments:** explicit NumPy/PyTorch seeds and isolated episode profiles.
- **Operational metrics:** reward, SOC, grid exchange, energy cost, and constraint penalties are returned from every transition.
- **Testable architecture:** environment, learner, and training loop are separated into importable modules.
- **Container-ready:** reproducible Python image for training.
- **CI:** automated linting and unit tests on pushes and pull requests.

## Project structure

```text
src/gridmind/
├── __init__.py
├── env.py       # constrained microgrid simulator
├── agent.py     # Double DQN, replay buffer, Q-network
└── train.py     # reproducible training entry point

tests/
└── test_env.py

.github/workflows/ci.yml
Dockerfile
pyproject.toml
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m gridmind.train --episodes 100
```

## Run with Docker

```bash
docker build -t gridmind .
docker run --rm gridmind
```

## Why this project matters

The important engineering problem is not simply training a neural network. An energy controller must operate inside physical constraints and expose measurable consequences of its decisions. GridMind therefore treats the simulator as an environment contract and makes safety signals first-class outputs of the learning loop.

## Roadmap

- Prioritized experience replay
- Distributional / dueling DQN variants
- Optuna hyperparameter search
- MLflow experiment tracking
- Offline evaluation against rule-based and MPC baselines
- Forecast-aware state representation
- FastAPI policy inference service
- Prometheus metrics and Grafana dashboards
- Kubernetes deployment

> Roadmap items are intentionally listed as future work rather than presented as implemented functionality.
