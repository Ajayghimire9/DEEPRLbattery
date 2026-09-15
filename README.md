# GridMind

Reinforcement learning for battery dispatch.

GridMind explores battery scheduling under energy and efficiency constraints. The packaged simulator and DDQN learner provide a small reproducible environment for checking controller behavior before attempting a larger energy experiment.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m gridmind.train --episodes 20
```

## Design decisions

Positive action means discharge into the AC bus; negative action means charging. Grid demand and battery losses use the same sign convention.

Charge and discharge limits account for efficiency, keeping state of charge within [0,1]. Invalid actions and steps after termination are rejected.

Double DQN uses experience replay, a target network, Huber loss and gradient clipping. Python, NumPy and PyTorch seeds are set for repeatable CPU experiments.

## Technology

Python, NumPy, PyTorch, pytest, Docker, GitHub Actions.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The packaged simulator uses synthetic profiles and permits grid imports and exports. It is distinct from the original off-grid thesis experiments in basecase.py, case_2.py and case_3.py. The summary reports training returns, not an independently validated policy improvement. Checkpoint-based serving and MLflow integration are not implemented.
