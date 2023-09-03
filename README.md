# DEEPRLbattery
# Microgrid Environment and Double Deep Q-Network (DDQN) Agent

This repository contains code for a custom gym environment representing a microgrid and a Double Deep Q-Network (DDQN) agent trained to make decisions within this environment.

## Microgrid Environment (MicrogridEnv)

The `MicrogridEnv` class represents a microgrid environment in which the DDQN agent learns to control energy storage and consumption based on various parameters.

### Features of the Microgrid Environment:

- Battery and hydrogen tank energy storage control
- Energy storage sizing and efficiency parameters
- Reward function modeling
- Time-based constraints and forecasts

## Double Deep Q-Network (DDQN) Agent

The `DoubleDQNAgent` class implements the DDQN algorithm to train an agent to make decisions within the microgrid environment.

### Features of the DDQN Agent:

- Neural network model architecture using Keras
- Experience replay for memory management
- Target network updates for improved stability
- Exploration-exploitation trade-off using epsilon-greedy policy
- Training and evaluation on different microgrid scenarios

## Usage and Examples

To run and evaluate the agent, follow these steps:

1. Set up the required datasets:
   - Load the consumption, production, and spotmarket data.
   - Resample the data to hourly timesteps using forward fill.

2. Initialize and Train the Agent:
   - Initialize the DDQN agent with the appropriate state and action sizes.
   - Train the agent using the training data and save the trained weights.

3. Evaluate the Agent:
   - Load the trained agent's weights.
   - Create separate environments for summer and winter test data.
   - Evaluate the agent's performance on the test data for both summer and winter periods.

4. Generate Specific Graphs:
   - Use the `generate_specific_graph` function to visualize the agent's actions, battery levels, consumption, and production over time.

## Requirements

The code is implemented in Python and requires the following libraries:

- NumPy
- gym
- pandas
- matplotlib
- keras (for building and training neural networks)




## Acknowledgments

This project is inspired by research in microgrid management and reinforcement learning. It provides a starting point for developing and experimenting with reinforcement learning algorithms in microgrid scenarios.

For any questions or inquiries, please contact Ajay Ghimire at ajayghimire42@gmail.com.
