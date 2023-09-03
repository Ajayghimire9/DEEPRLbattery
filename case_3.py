# -*- coding: utf-8 -*-
"""Case 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IksRnKLqEdz6WOM6X7BF9nwfGtsc2qiD
"""

from google.colab import drive
drive.mount('/content/drive')

# Importing the libraries
import gym
import random
from gym import spaces
import numpy as np
import pandas as pd
from collections import namedtuple
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime

# CUstom gym enviroenmnt
class MicrogridEnv(gym.Env):
    def __init__(self, consumption_data, production_data, spotmarket_data, n_past=5):
        super().__init__()

        # Microgrid parameters
        self.s_B = 0  # Initial energy in the battery
        self.s_H2 = 0  # Initial energy in the hydrogen tank


        self.n_past = n_past

        # Initialize time step
        self.t = 0

        # Battery discharge and charge efficiency
        self.eta_B = 0.9  # Discharge efficiency for battery
        self.zeta_B = 0.9  # Charge efficiency for battery

        # Electrolysis and fuel cells efficiencies
        self.eta_H2 = 0.65  # Efficiency when storing energy in hydrogen
        self.zeta_H2 = 0.65  # Efficiency when delivering energy from hydrogen

        # Energy storage sizing for battery and hydrogen
        self.x_B = 15 * 1000  # Wh (15 kWh as per text)
        self.x_H2 = 14000  # Wp (adjust as per text)

        # Reward function parameters
        self.k = 2  # Cost endured per kWh not supplied within the microgrid
        self.k_H2 = 0.1  # Revenue/cost per kWh of hydrogen produced/used


        # Number of past observations to consider
        self.n_past = n_past
        self.past_consumption = [0] * self.n_past
        self.past_production = [0] * self.n_past


        # Consumption and production data
        self.consumption_data = consumption_data
        self.production_data = production_data


        # Add forecast variables to state (for Case 3)
        self.rho_24 = 0  # Production forecast for the next 24 hours
        self.rho_48 = 0  # Production forecast for the next 48 hours

        # Update state vector
        self.state = np.concatenate(([self.s_B, self.s_H2, self._calculate_zeta_s(self.t), self.rho_24, self.rho_48],
                                     self.past_consumption,
                                     self.past_production))

        # Update observation space
        low_bounds = [0] * (5 + 2 * self.n_past)  # Battery, Hydrogen, zeta_s, rho_24, rho_48, past consumption and past production
        high_bounds = [np.inf] * (5 + 2 * self.n_past)
        self.observation_space = spaces.Box(low=np.array(low_bounds), high=np.array(high_bounds), dtype=np.float32)

        self.action_space = spaces.Discrete(3)  # Actions: (0) discharge, (1) idle, (2) charge

        # Reset the environment upon initialization
        self.reset()

    def _calculate_zeta_s(self, current_timestep):
        """Calculate the normalized smallest number of days to the solar solstice (21st of June)."""
        current_date = datetime.date(2007, 1, 1) + datetime.timedelta(hours=current_timestep)
        current_year = current_date.year
        solstice_this_year = datetime.date(current_year, 6, 21)
        solstice_prev_year = datetime.date(current_year - 1, 6, 21)
        solstice_next_year = datetime.date(current_year + 1, 6, 21)
        days_to_this_year_solstice = abs((solstice_this_year - current_date).days)
        days_to_prev_year_solstice = abs((solstice_prev_year - current_date).days)
        days_to_next_year_solstice = abs((solstice_next_year - current_date).days)
        min_days = min(days_to_this_year_solstice, days_to_prev_year_solstice, days_to_next_year_solstice)
        zeta_s = min_days / 182.5
        return zeta_s


    def step(self, action):

        # Define constraints
        a_B_bounds = [-self.zeta_B * self.s_B, self.x_B - self.s_B / self.eta_B]
        a_H2_bounds = [-self.zeta_H2 * self.s_H2, self.x_H2]

        # Map the action to a specific storage operation
        if action == 0:  # discharge hydrogen at full rate
            a_H2_t = a_H2_bounds[0]
        elif action == 1:  # keep hydrogen idle
            a_H2_t = 0
        elif action == 2:  # charge hydrogen at full rate
            a_H2_t = a_H2_bounds[1]
        else:
            raise ValueError("Invalid action. Actions must be 0 (discharge), 1 (idle), or 2 (charge).")

        # Calculate a_B_t based on constraints
        a_B_t = max(min(a_H2_t, a_B_bounds[1]), a_B_bounds[0])


        # Update the energy levels in the battery and hydrogen tank
        self.s_B += abs(a_B_t) if a_B_t >= 0 else -abs(a_B_t)
        self.s_H2 += abs(a_H2_t) if a_H2_t >= 0 else -abs(a_H2_t)


        # Ensure that the energy levels are within their bounds
        self.s_B = max(0, min(self.s_B, self.x_B))
        self.s_H2 = max(0, min(self.s_H2, self.x_H2))



        # Update past consumption and production
        self.past_consumption.pop(0)
        self.past_consumption.append(self.consumption_data[self.t])
        self.past_production.pop(0)
        self.past_production.append(self.production_data[self.t])



        # Update the state with added rho_24 and rho_48
        self.rho_24 = np.sum(self.production_data[self.t:self.t+24])  # Sum of production for next 24 hours
        self.rho_48 = np.sum(self.production_data[self.t:self.t+48])  # Sum of production for next 48 hours
        zeta_s = self._calculate_zeta_s(self.t)
        self.state = np.concatenate(([self.s_B, self.s_H2, zeta_s, self.rho_24, self.rho_48],
                                     self.past_consumption,
                                     self.past_production))


        # Calculate the net electricity demand d_t
        d_t = self.consumption_data[self.t] - self.production_data[self.t]

        # Calculate the power balance within the microgrid
        delta_t = -a_B_t - a_H2_t - d_t

        # Calculate reward
        r_H2 = self.k_H2 * a_H2_t if a_H2_t > 0 else 0
        r_minus = self.k * delta_t if delta_t < 0 else 0
        reward = r_H2 + r_minus

        # Bonus for opposite actions between battery and hydrogen
        bonus = 0.1  # Define a suitable bonus value; this value can be fine-tuned
        if (a_H2_t > 0 and a_B_t < 0) or (a_H2_t < 0 and a_B_t > 0):
            reward += bonus

        # Increase time step
        self.t += 1

        # Signal the end of the episode after a fixed number of steps (e.g., 24 hours)
        done = self.t >= 24

        return self.state, reward, done, {}

    def reset(self):
        # Reset time step
        self.t = 0

        # Reset the short-term storage (battery) and long-term storage (hydrogen) to their initial states.
        self.s_B = self.x_B / 2  # Assuming battery starts half-charged
        self.s_H2 = self.x_H2 / 2  # Assuming hydrogen storage starts half-filled

        # Initialize past consumption and production as zero
        self.past_consumption = [0]*self.n_past
        self.past_production = [0]*self.n_past

         # Update the state with added rho_24 and rho_48
        self.rho_24 = np.sum(self.production_data[self.t:self.t+24])  # Sum of production for next 24 hours
        self.rho_48 = np.sum(self.production_data[self.t:self.t+48])  # Sum of production for next 48 hours
        zeta_s = self._calculate_zeta_s(self.t)
        self.state = np.concatenate(([self.s_B, self.s_H2, zeta_s, self.rho_24, self.rho_48],
                                     self.past_consumption,
                                     self.past_production))

        # Return the initial state
        return self.state

    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass

# loading the datasets
pv_prod_test = np.load("/content/drive/MyDrive/BelgiumPV_prod_test.npy")
pv_prod_train = np.load("/content/drive/MyDrive/BelgiumPV_prod_train.npy")
nondet_cons_test = np.load("/content/drive/MyDrive/example_nondeterminist_cons_test.npy")
nondet_cons_train = np.load("/content/drive/MyDrive/example_nondeterminist_cons_train.npy")
spotmarket_data = pd.read_excel("/content/drive/MyDrive/spotmarket_data_2007-2013.xls")
consumption_data = nondet_cons_train
production_data = pv_prod_train

# Convert 'Date' column to datetime index in spotmarket_data
spotmarket_data['Date'] = pd.to_datetime(spotmarket_data['Date'])
spotmarket_data.set_index('Date', inplace=True)

# Resample the spotmarket_data to hourly timesteps using forward fill ('ffill')
spotmarket_data = spotmarket_data['BASE (00-24)'].resample('1H').ffill()
consumption_data = nondet_cons_train
production_data = pv_prod_train


# Determine the indices for summer and winter periods
# Assuming the data starts from January 1 and each day has 24 data points
summer_start = 31*24 + 28*24 + 31*24 + 30*24 + 31*24  # Start of June
summer_end = summer_start + 92*24  # End of August
winter_start = 0  # Start of December
winter_end = 31*24 + 31*24 + 28*24  # End of February

# Select summer and winter data
summer_consumption = consumption_data[summer_start:summer_end]
winter_consumption = consumption_data[winter_start:winter_end]
summer_production = production_data[summer_start:summer_end]
winter_production = production_data[winter_start:winter_end]


# Initialize the environment with general data
env = MicrogridEnv(consumption_data, production_data, spotmarket_data)

# Function to run a random episode
def run_random_episode(env, num_steps=24):
    state = env.reset()
    total_reward = 0
    for t in range(num_steps):
        action = env.action_space.sample()  # Take a random action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    print(f"Total reward: {total_reward}")

print("Running random episode for general environment:")
run_random_episode(env)

# Model with keras

states = env.observation_space.shape
state_size = states[0]
actions = env.action_space.n

from keras.layers import LSTM

class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.create_lstm_model()
        self.target_model = self.create_lstm_model()
        self.optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
        self.update_target_model()

    def create_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(self.state_size, 1), return_sequences=True))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(self.action_size, activation='linear'))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, self.state_size, 1))
            return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Double DQN logic
                action_from_main = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.gamma * self.target_model.predict(next_state)[0][action_from_main]

            target_f = self.model.predict(state)
            target_f[0][action] = target
            state = np.reshape(state, (1, self.state_size, 1))
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Save the model weights."""
        self.model.save_weights(filename)

    def load(self, filename):
        """Load the model weights."""
        self.model.load_weights(filename)

state_size = states[0]  # Number of elements in the state
agent = DoubleDQNAgent(state_size, actions)
num_episodes = 800
batch_size = 32
rewards = []
# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size, 1])  # Reshape for LSTM
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size, 1])  # Reshape for LSTM
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    # Update target model every episode
    agent.update_target_model()

# Plotting the reward per episode
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode during Training for Case 3')
plt.grid(True)
plt.show()

import os
# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained weights
agent.save('models/microgrid_doubledqn3_weights.h5')

# Initialize the test agent and load the trained weights for evaluation
test_agent = DoubleDQNAgent(state_size, actions)
test_agent.load('models/microgrid_doubledqn3_weights.h5')


# Determine Indices for Summer and Winter Periods for Test Data
summer_start_test = 31*24 + 28*24 + 31*24 + 30*24 + 31*24  # Start of June
summer_end_test = summer_start_test + 92*24  # End of August
winter_start_test = 0  # Start of December
winter_end_test = 31*24 + 31*24 + 28*24  # End of February

# Extract Summer and Winter Data from Test Data
summer_consumption_test = nondet_cons_test[summer_start_test:summer_end_test]
winter_consumption_test = nondet_cons_test[winter_start_test:winter_end_test]
summer_production_test = pv_prod_test[summer_start_test:summer_end_test]
winter_production_test = pv_prod_test[winter_start_test:winter_end_test]

# Create Environments for Summer and Winter Test Data
summer_test_env = MicrogridEnv(summer_consumption_test, summer_production_test, spotmarket_data, n_past=5)
winter_test_env = MicrogridEnv(winter_consumption_test, winter_production_test, spotmarket_data, n_past=5)

# Load the trained weights for evaluation
agent.load('models/microgrid_doubledqn3_weights.h5')

# Evaluate the trained agent on summer and winter test environments
def evaluate_agent(agent, env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state = np.reshape(state, [1, state_size, 1])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

summer_score = evaluate_agent(test_agent, summer_test_env)
winter_score = evaluate_agent(test_agent, winter_test_env)

print(f"Evaluation score on summer test data for case 3: {summer_score}")
print(f"Evaluation score on winter test data for case 3: {winter_score}")

# After training the agent on the training data, evaluate its policy on the summer and winter periods of the test data.

def generate_specific_graph(agent, env, state_size, title, n_past=5):
    states = []
    actions = []
    consumptions = []
    productions = []

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time_step = 0
    while not done:
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        states.append(next_state.flatten())
        actions.append(action)
        consumptions.append(env.consumption_data[env.t - 1])
        productions.append(env.production_data[env.t - 1])
        state = next_state
        time_step += 1

    states = np.array(states)
    time_steps = np.arange(time_step)  # Create an array of time steps from 0 to time_step-1
    fig, ax1 = plt.subplots(figsize=(10, 6))



     # Print consumptions and productions for debugging
    print("Consumptions:", consumptions)
    print("Productions:", productions)

    # Left Y-Axis 1: H Action
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('H Action', color='lightblue')
    action_labels = ['Discharge', 'Idle', 'Charge']
    actions = [action_labels[action] for action in actions]  # Map action values to labels
    ax1.plot(time_steps, actions, label='H Action', color='lightblue', marker='o', linestyle='-', markersize=3)
    ax1.set_ylim([-0.5, 2.5])  # Adjust the y-axis limits to align the ticks with labels
    ax1.set_yticks([0, 1, 2])  # Set custom y-axis ticks to match action labels
    ax1.set_yticklabels(action_labels)  # Set action labels as y-axis tick labels
    ax1.legend(loc='upper left')

    # Left Y-Axis 2: Battery Level (kWh)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 60))
    ax2.set_ylabel('Battery Level (kWh)', color='darkblue')
    battery_levels = states[:, 0] / 1000  # Convert from Wh to kWh
    ax2.plot(time_steps, battery_levels, label='Battery Level', color='darkblue', marker='o', linestyle='-', markersize=3)
    ax2.set_ylim([min(battery_levels), max(battery_levels)])
    ax2.tick_params(axis='y', labelcolor='darkblue')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

    # Right Y-Axis 1: Consumption
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))
    ax3.set_ylabel('Consumption', color='red')
    ax3.plot(time_steps, consumptions, label='Consumption', color='red', marker='o', linestyle='-', markersize=3)
    ax3.set_ylim([0, 10])  # Adjust the y-axis limits as needed
    ax3.tick_params(axis='y', labelcolor='red')

    # Right Y-Axis 2: Production
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 180))
    ax4.set_ylabel('Production', color='green')
    ax4.plot(time_steps, productions, label='Production', color='green', marker='o', linestyle='-', markersize=3)
    ax4.set_ylim([0, 10])  # Adjust the y-axis limits as needed
    ax4.tick_params(axis='y', labelcolor='green')


    # Adjust the y-axis limits for consumption and production
    max_value = max(max(consumptions), max(productions))
    ax3.set_ylim([0, max_value])  # Adjust the y-axis limits as needed for consumption
    ax4.set_ylim([0, max_value])  # Adjust the y-axis limits as needed f

    # Combine the legends from ax3 and ax4
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1, 0.9))

    plt.title(title)
    plt.show()

# Generate the Graphs for Summer and Winter on Test Data
generate_specific_graph(test_agent, summer_test_env, state_size, "Policy During Summer (Test Data) for case 3")
generate_specific_graph(test_agent, winter_test_env, state_size, "Policy During Winter (Test Data)for case 3")