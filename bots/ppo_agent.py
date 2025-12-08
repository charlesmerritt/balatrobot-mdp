"""
Train and evaluate a PPO agent on BalatroEnv using Stable-Baselines3.

Usage:
    uv run bots/ppo_agent.py        # train then evaluate
"""

from __future__ import annotations

import os
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from balatrobot.env import BalatroEnv
from balatrobot.enums import Decks, Stakes


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------


def make_env(
    port: int = 12346,
    deck: str = Decks.RED.value,
    stake: int = Stakes.WHITE.value,
    seed: str | None = None,
    max_steps: int = 500,
    render_mode: str | None = None,
) -> Callable[[], BalatroEnv]:
    """Return a thunk that creates a fresh BalatroEnv instance."""

    def _init() -> BalatroEnv:
        env = BalatroEnv(
            port=port,
            deck=deck,
            stake=stake,
            seed=seed,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        return env

    return _init


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_ppo(
    total_timesteps: int = 100_000,
    model_path: str = "ppo_balatro",
) -> PPO:
    """Train PPO on a single-vectorized BalatroEnv and save the model."""

    vec_env = DummyVecEnv([make_env(render_mode=None)])

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    # Save model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", model_path)
    model.save(save_path)
    print(f"Saved PPO model to {save_path}.zip")

    return model


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_ppo(
    model_path: str = "ppo_balatro",
    n_eval_episodes: int = 5,
) -> None:
    """Load a trained PPO model and run evaluation episodes."""

    vec_env = DummyVecEnv([make_env(render_mode=None)])

    load_path = os.path.join("models", model_path)
    model = PPO.load(load_path, env=vec_env)

    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    print(f"Evaluation over {n_eval_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------


def main() -> None:
    # 1) Train
    model = train_ppo(total_timesteps=100_000, model_path="ppo_balatro")

    # 2) Evaluate the trained model
    evaluate_ppo(model_path="ppo_balatro", n_eval_episodes=5)


if __name__ == "__main__":
    main()
