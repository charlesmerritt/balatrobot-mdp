# debug_env.py
import time
import numpy as np

from balatrobot.env import BalatroEnv
from balatrobot.enums import Decks, Stakes

def make_env():
    env = BalatroEnv(
        port=12346,
        deck=Decks.RED.value,
        stake=Stakes.WHITE.value,
        seed=None,
        max_steps=200,
        render_mode="human",  # or None for headless
    )
    return env

def pretty_obs(obs: dict) -> dict:
    """Compact view of the obs for debugging."""
    return {
        "state": int(obs["state"][0]),
        "chips": float(obs["chips"][0]),
        "dollars": float(obs["dollars"][0]),
        "round": float(obs["round"][0]),
        "hands_left": float(obs["hands_left"][0]),
        "discards_left": float(obs["discards_left"][0]),
        "hand_size": float(obs["hand_size"][0]),
        "joker_count": float(obs["joker_count"][0]),
        "blind_target": float(obs["blind_target"][0]),
        "current_hand_type": float(obs["current_hand_type"][0]),
        "deck_vector_sum": float(np.sum(obs["deck_vector"])),
        "joker_ids": obs["joker_ids"].tolist(),
    }

def random_policy(env: BalatroEnv, obs: dict) -> int:
    """Very dumb policy: random action, but you can replace this with your own."""
    return env.action_space.sample()

def main():
    env = make_env()
    obs, info = env.reset()
    print("Initial obs:", pretty_obs(obs))

    total_reward = 0.0
    step = 0
    done = False

    while not done and step < 200:
        action = random_policy(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(
            f"Step {step}: "
            f"action={action}, "
            f"reward={reward:.2f}, "
            f"total_reward={total_reward:.2f}, "
            f"state={int(next_obs['state'][0])}, "
            f"chips={float(next_obs['chips'][0])}"
        )

        # Optional: slow down so you can watch Balatro
        # time.sleep(0.2)

        obs = next_obs
        done = terminated or truncated

    print("Episode done. Total reward:", total_reward)

if __name__ == "__main__":
    main()
