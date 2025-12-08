from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from balatrobot.env import BalatroEnv
from balatrobot.enums import Decks, Stakes

def make_env():
    def _init():
        return BalatroEnv(
            port=12346,
            deck=Decks.RED.value,
            stake=Stakes.WHITE.value,
            seed=None,         # or fixed seed for reproducibility
            max_steps=500,
            render_mode=None,  # turn off printing for training
        )
    return _init

# SB3 wants a VecEnv
env = DummyVecEnv([make_env()])
