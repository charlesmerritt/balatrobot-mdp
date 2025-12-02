"""Tests for BalatroBot Gymnasium environment."""

from typing import cast

import pytest
from gymnasium import spaces

from balatrobot.enums import Decks, Stakes
from balatrobot.env import BalatroEnv


class TestBalatroEnv:
    """Test suite for BalatroEnv."""

    def test_env_creation(self):
        """Test that environment can be created with default parameters."""
        env = BalatroEnv()
        assert env is not None
        assert env.port == 12346
        assert env.deck == Decks.RED.value
        assert env.stake == Stakes.WHITE.value
        assert env.max_steps == 1000
        env.close()

    def test_env_custom_parameters(self):
        """Test environment creation with custom parameters."""
        env = BalatroEnv(
            port=12347,
            deck=Decks.BLUE.value,
            stake=Stakes.GREEN.value,
            seed="TEST123",
            max_steps=500,
        )
        assert env.port == 12347
        assert env.deck == Decks.BLUE.value
        assert env.stake == Stakes.GREEN.value
        assert env.game_seed == "TEST123"
        assert env.max_steps == 500
        env.close()

    def test_observation_space(self):
        """Test that observation space is properly defined."""
        env = BalatroEnv()
        obs_space = cast(spaces.Dict, env.observation_space)

        # Check that all expected keys are present
        expected_keys = {
            "state",
            "chips",
            "dollars",
            "round",
            "hands_left",
            "discards_left",
            "hand_size",
            "joker_count",
        }
        assert set(obs_space.spaces.keys()) == expected_keys
        env.close()

    def test_action_space(self):
        """Test that action space is properly defined."""
        env = BalatroEnv()
        action_space = cast(spaces.Discrete, env.action_space)
        assert action_space.n == 64
        env.close()

    def test_get_obs_no_state(self):
        """Test observation extraction with no game state."""
        env = BalatroEnv()
        env.current_state = None
        obs = env._get_obs()

        # All values should be zero or default
        assert obs["chips"][0] == 0.0
        assert obs["dollars"][0] == 0.0
        assert obs["round"][0] == 0.0
        env.close()

    def test_get_info(self):
        """Test info dictionary structure."""
        env = BalatroEnv()
        env.current_step = 5
        env.episode_reward = 10.5

        info = env._get_info()

        assert "episode_step" in info
        assert "episode_reward" in info
        assert "game_state" in info
        assert info["episode_step"] == 5
        assert info["episode_reward"] == 10.5
        env.close()

    def test_action_to_cards(self):
        """Test action to card index conversion."""
        env = BalatroEnv()

        # Test simple cases
        assert env._action_to_cards(0) == [0]
        assert env._action_to_cards(1) == [0]
        assert env._action_to_cards(2) == [1]
        assert env._action_to_cards(3) == [0, 1]

        # Test multi-card selection
        cards = env._action_to_cards(7)  # Binary: 00111
        assert 0 in cards and 1 in cards and 2 in cards
        env.close()

    @pytest.mark.skip(reason="Requires running Balatro game instance")
    def test_reset(self):
        """Test environment reset (requires game connection)."""
        env = BalatroEnv()
        obs, info = env.reset()

        # Check observation structure
        assert "state" in obs
        assert "chips" in obs
        assert "dollars" in obs

        # Check info structure
        assert "episode_step" in info
        assert info["episode_step"] == 0

        env.close()

    @pytest.mark.skip(reason="Requires running Balatro game instance")
    def test_step(self):
        """Test environment step (requires game connection)."""
        env = BalatroEnv()
        env.reset()

        # Take a step
        obs, reward, terminated, truncated, info = env.step(1)

        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    @pytest.mark.skip(reason="Requires running Balatro game instance")
    def test_episode_termination(self):
        """Test that episode terminates properly."""
        env = BalatroEnv(max_steps=10)
        env.reset()

        # Run until termination
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 20:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1

        # Should terminate within max_steps
        assert steps <= 10 or terminated or truncated

        env.close()

    def test_close(self):
        """Test environment cleanup."""
        env = BalatroEnv()
        env.close()
        assert not env._connected
