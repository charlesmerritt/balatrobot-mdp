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

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    n_steps=1024,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1,
)

model.learn(total_timesteps=100_000)
model.save("ppo_balatro")

from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("ppo_balatro", env=env)

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=5,
    deterministic=True,
)
print("Mean reward:", mean_reward, "+/-", std_reward)
