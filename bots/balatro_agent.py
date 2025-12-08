from collections import defaultdict
import gymnasium as gym
import numpy as np
from typing import Any
from typing import cast
from gymnasium.spaces import Discrete

from balatrobot.env import BalatroEnv
from balatrobot.enums import Decks, Stakes

"""
In order to see how a trained agent performed, change the
"""

def obs_to_tuple(obs: dict) -> tuple:
        """
        Convert observation dict to a hashable tuple for Q-table indexing.
        Flatten numeric arrays to scalars or tuples.
        """
        values = []
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                # Flatten array and convert to tuple of floats
                values.extend(v.flatten().tolist())
            else:
                values.append(v)
        return tuple(values)

class BalatroAgent:
    def __init__(
        self,
        env: BalatroEnv,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):

        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        assert isinstance(env.action_space, Discrete), \
            "BalatroAgent requires a discrete action space"

        self.env = env
        self.q_values = defaultdict(
            lambda: np.zeros(cast(Discrete, env.action_space).n)
        )

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: dict) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        obs_key = obs_to_tuple(obs)
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs_key]))
    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: dict,
    ):
        obs_key = obs_to_tuple(obs)
        next_key = obs_to_tuple(next_obs)

        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_key])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs_key][action]
        )

        self.q_values[obs_key][action] = (
            self.q_values[obs_key][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# TRAIN THE AGENT
# hyperparameters
learning_rate = 0.01
n_episodes = 1000 # 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

import random
import string

def generate_random_string(length):
    # Define the pool of characters to choose from (e.g., letters and digits)
    characters = string.ascii_letters + string.digits
    # Use random.choices to pick 'length' characters from the pool
    # and then join them into a single string
    random_string = ''.join(random.choices(characters, k=length))
    print(random_string)
    return random_string

env = BalatroEnv(
        port=12346,
        deck=Decks.RED.value,
        stake=Stakes.WHITE.value,
        seed=generate_random_string(length=7),
        max_steps=500,
        render_mode="human",
    )

agent = BalatroAgent(
    env = env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Optional: Reload q table from old run
import pickle
reload_old_q_table = False # *** Change me ***
if reload_old_q_table:
    with open("saved_tables/qtable_final.pkl", "rb") as f:
        q = pickle.load(f)
    agent.q_values = defaultdict(
        lambda: np.zeros(cast(Discrete, agent.env.action_space).n),
        q
    )
    agent.epsilon = 0
    n_episodes = 1
    env.game_seed = "l324WIF"

from tqdm import tqdm

episode_rewards = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()  # reset returns obs and info
    done = False
    total_reward = 0.0

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        if not reload_old_q_table:
            agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done
        done = terminated or truncated
        obs = next_obs
        total_reward += reward

    # Save the results from each run
    episode_rewards.append(total_reward)

    # optional: print episode stats every 1000 episodes
    #if (episode + 1) % 1 == 0:
    #    print(f"Episode {episode+1}: total_reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}")

    agent.decay_epsilon()

# Plot the agent's reward over time
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Agent Training Performance")
plt.show()

# Save rewards to CSV
import csv
if not reload_old_q_table:
    with open("run_data/one_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(episode_rewards):
            writer.writerow([i + 1, r])

# Save the Q table to file
import pickle
if not reload_old_q_table:
    with open("saved_tables/qtable_final.pkl", "wb") as f:
        pickle.dump(dict(agent.q_values), f)

# Print the seed
print(env.game_seed)
