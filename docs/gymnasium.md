# Gymnasium MDP Integration

BalatroBot provides a Gymnasium-compatible environment for training reinforcement learning agents to play Balatro.

## Overview

The `BalatroEnv` class wraps the BalatroBot client in a standard Gymnasium interface, providing:

- **Observation Space**: Dictionary containing game state features
- **Action Space**: Discrete actions mapped to game commands
- **Reward Function**: Configurable rewards based on game progress
- **Episode Management**: Automatic reset and termination handling

## Installation

Ensure you have the required dependencies:

```bash
pip install gymnasium numpy
# or with uv
uv add gymnasium numpy
```

## Basic Usage

```python
from balatrobot import BalatroEnv
from balatrobot.enums import Decks, Stakes

# Create the environment
env = BalatroEnv(
    port=12346,
    deck=Decks.RED.value,
    stake=Stakes.WHITE.value,
    seed="SEED123",
    max_steps=1000,
    render_mode="human",
)

# Reset to start a new episode
observation, info = env.reset()

# Run an episode
done = False
total_reward = 0

while not done:
    # Select an action (random for this example)
    action = env.action_space.sample()

    # Execute the action
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

    # Optionally render
    env.render()

print(f"Episode finished with total reward: {total_reward}")
env.close()
```

## MDP Structure

### State Space

The MDP follows the game's natural state transitions:

1. **MENU** → `start_run()` → **BLIND_SELECT**
2. **BLIND_SELECT** → `skip_or_select_blind()` → **SELECTING_HAND**
3. **SELECTING_HAND** → `play_hand_or_discard()` → **HAND_PLAYED/DRAW_TO_HAND**
4. **ROUND_EVAL** → `cash_out()` → **SHOP**
5. **SHOP** → `shop(next_round)` → **BLIND_SELECT**
6. Terminal states: **GAME_OVER** or max steps reached

### Observation Space

The observation is a dictionary with the following keys:

| Key             | Type        | Range   | Description                            |
| --------------- | ----------- | ------- | -------------------------------------- |
| `state`         | float32[1]  | [0, 27] | Current game state enum value          |
| `chips`         | float32[1]  | \[0, ∞) | Current chip count                     |
| `dollars`       | float32[1]  | \[0, ∞) | Current money amount                   |
| `round`         | float32[1]  | \[0, ∞) | Current round number                   |
| `hands_left`    | float32[1]  | [0, 10] | Hands remaining in round               |
| `discards_left` | float32[1]  | [0, 10] | Discards remaining in round            |
| `hand_size`     | float32[1]  | [0, 20] | Number of cards in hand                |
| `joker_count`   | float32[1]  | [0, 10] | Number of active jokers                |
| `deck_vector`   | float32[52] | {0, 1}  | Fixed 52-card composition (1 per card) |

### Action Space & Policies

The environment still exposes a `Discrete(64)` Gym action space for future compatibility, but the current policies operate internally (the external action argument is ignored for now). Built-in policies are selected through the `policy` parameter:

1. **`random`** – Always selects the blind, then randomly chooses to play or discard up to five cards each turn.
2. **`greedy`** – Evaluates every 5-card combination and always plays the highest-ranked poker hand (no discarding).
3. **`hybrid`** – Plays the best hand if its rank is above Three of a Kind; otherwise, if there is a 4-card flush draw, discards off-suit cards to fish for the flush, falling back to the best hand when no draw exists.

This structure allows apples-to-apples comparisons between baseline policies while we work toward re-exposing the action space for external agents.

### Reward Function

The default reward function is shaped specifically for Balatro progression:

1. **Chips gained**: `+0.5 × chips_delta`
2. **Money gained**: `+1.0 × dollars_delta`
3. **Round advancement**: `+10 × round_number` (round 1: +10, round 2: +20, ...)
4. **Ante advancement**: `+50 + 50 × ante_number` (ante 1: +100, ante 2: +150, ...)

No per-step penalty is currently applied. Customize the reward signal by subclassing `BalatroEnv` and overriding `_calculate_reward()`.

You can customize the reward function by subclassing `BalatroEnv` and overriding `_calculate_reward()`.

## Configuration

### Environment Parameters

```python
BalatroEnv(
    port=12346,              # BalatroBot API port
    deck="Red Deck",         # Starting deck
    stake=1,                 # Difficulty (1-8)
    seed=None,              # Optional seed for reproducibility
    max_steps=1000,         # Maximum steps per episode
    render_mode=None,       # "human" or "rgb_array"
    policy="greedy",         # "random", "greedy", or "hybrid"
)
```

### Render Modes

- **`"human"`**: Prints game state to console
- **`"rgb_array"`**: Returns screenshot as numpy array (requires PIL)
- **`None`**: No rendering

## Advanced Usage

### Custom Reward Function

```python
class CustomBalatroEnv(BalatroEnv):
    def _calculate_reward(self) -> float:
        """Custom reward focusing on money accumulation."""
        if self.current_state is None or self.current_state.game is None:
            return 0.0

        reward = 0.0
        game = self.current_state.game

        # Heavy reward for money
        dollars_gained = game.dollars - self.last_dollars
        reward += dollars_gained * 1.0

        # Bonus for jokers
        if isinstance(self.current_state.jokers, list):
            reward += len(self.current_state.jokers) * 0.5

        # Update tracking
        self.last_dollars = game.dollars

        return reward
```

### Integration with RL Libraries

#### Stable-Baselines3

```python
from stable_baselines3 import PPO
from balatrobot import BalatroEnv

# Create environment
env = BalatroEnv(max_steps=500)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Save model
model.save("balatro_ppo")

# Test trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

#### Ray RLlib

```python
from ray.rllib.algorithms.ppo import PPOConfig
from balatrobot import BalatroEnv

# Configure algorithm
config = (
    PPOConfig()
    .environment(BalatroEnv, env_config={
        "port": 12346,
        "max_steps": 500,
    })
    .framework("torch")
    .training(lr=3e-4)
)

# Build and train
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}")

algo.save("balatro_rllib")
```

## Error Handling

The environment handles common errors gracefully:

- **Connection failures**: Environment will attempt to reconnect on reset
- **Invalid actions**: Logged as warnings, game state retrieved
- **Game crashes**: Episode terminates with appropriate reward

## Debugging

Enable debug logging to see detailed information:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("balatrobot")
logger.setLevel(logging.DEBUG)

env = BalatroEnv()
# ... use environment
```

## Limitations

1. **Simplified Action Space**: The current action space uses a simplified encoding. Full Balatro gameplay would require more complex action representations.

2. **Observation Features**: The observation space captures core game state but doesn't include all available information (e.g., specific card details, joker effects).

3. **Real-time Constraint**: The environment requires a running Balatro game instance, making training slower than simulated environments.

4. **Partial Observability**: Some game information (future card draws, exact probabilities) is not observable.

## Future Enhancements

- **Action Masking**: Expose valid actions for current state
- **Hierarchical Actions**: Multi-level action space for shop and card selection
- **Rich Observations**: Include card embeddings and joker descriptions
- **Vectorized Environments**: Support for parallel game instances
- **Curriculum Learning**: Progressive difficulty through stake levels

## Examples

See the following files for complete examples:

- `bots/example_mdp.py`: Basic random agent
- `tests/balatrobot/test_env.py`: Unit tests and usage patterns

## Troubleshooting

**Issue**: Environment hangs on reset

**Solution**: Ensure Balatro is running with BalatroBot mod loaded and listening on the specified port.

---

**Issue**: Actions seem ineffective

**Solution**: Check that actions match the current game state. Enable debug logging to see action execution details.

---

**Issue**: Episode terminates immediately

**Solution**: Check the `max_steps` parameter and ensure the game isn't already in a terminal state.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [BalatroBot API Reference](../api/client.md)
- [Game State Models](../api/models.md)
