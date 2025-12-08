from balatrobot.env import BalatroEnv
from balatrobot.enums import Decks, Stakes
import numpy as np

env = BalatroEnv(
    port=12346,
    deck=Decks.RED.value,
    stake=Stakes.WHITE.value,
    seed=None,
    max_steps=50,
    render_mode="human",
)

obs, info = env.reset()
print("Initial hand_cards:\n", obs["hand_cards"])
print("Initial joker_ids:", obs["joker_ids"])
print("Initial deck_vector sum:", np.sum(obs["deck_vector"]))

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("-" * 40)
    print("state:", int(obs["state"][0]))
    print("chips:", float(obs["chips"][0]))
    print("hand_cards:\n", obs["hand_cards"])
    print("joker_ids:", obs["joker_ids"])
    print("blind_target:", float(obs["blind_target"][0]))
    print("deck_vector sum:", np.sum(obs["deck_vector"]))
    print("eval_hand_type:", obs["eval_hand_type"])
    print("hand_target_rank:", obs["hand_target_rank"])
    print("hand_target_completeness:", obs["hand_target_completeness"])

    done = terminated or truncated
