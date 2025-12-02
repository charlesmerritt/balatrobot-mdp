from typing import Optional
import numpy as np
import gymnasium as gym

# https://gymnasium.farama.org/introduction/create_custom_env/

class BalatroEnv(gym.Env): 

    # Constructor
    def __init__(self):
        super().__init__()

    def reset(self, seed=None, options=None):
        # Reset game

    def step(self, action):

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """