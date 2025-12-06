import logging
from tokenize import String
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import BalatroClient
from .deck import get_standard_deck_vector
from .enums import Decks, Stakes, State
from .exceptions import BalatroError
from .models import G

logger = logging.getLogger(__name__)


class BalatroEnv(gym.Env):
    """
    Gymnasium environment wrapper for BalatroBot.

    Follows Gymnasium API:
        - __init__
        - reset
        - step
        - render
        - close
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        port: int = 12346,
        deck: str = Decks.RED.value,
        stake: int = Stakes.WHITE.value,
        seed: Optional[str] = None,
        max_steps: int = 2000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.port = port
        self.deck = deck
        self.stake = stake
        self.game_seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.client: Optional[BalatroClient] = None
        self.current_state: Optional[G] = None

        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0

        # Deck vector (static feature)
        self.deck_vector = np.array(get_standard_deck_vector(), dtype=np.float32)

        # -----------------------------
        # DEFINE ACTION SPACE (GLOBAL)
        # -----------------------------
        # Max combinations of 5 cards = 32, + pass = 33
        MAX_HAND_ACTIONS = 33
        MAX_SHOP_ACTIONS = 10     # heuristic
        self.action_space = spaces.Discrete(MAX_HAND_ACTIONS + MAX_SHOP_ACTIONS)

        # -----------------------------
        # DEFINE OBSERVATION SPACE
        # -----------------------------
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Discrete(28),
                "chips": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                "dollars": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                "round": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
                "hands_left": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
                "discards_left": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
                "hand_size": spaces.Box(0, 20, shape=(1,), dtype=np.float32),
                "joker_count": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
                "deck_vector": spaces.Box(0, 1, (52,), dtype=np.float32),
            }
        )

    def _hand_action_from_index(self, action: int) -> tuple[str, list[int]]:
        """
        Converts an integer action into a hand play or discard command.

        Returns:
            ("pass", [])  if action == 32

            ("play", [list of card indices]) for actions 0–31
        """
        if self.current_state is None or self.current_state.hand is None:
            return "pass", []

        if action == 32:
            return "pass", []

        cards = []
        # use the hand size dynamically
        hand_size = len(self.current_state.hand.cards) if self.current_state.hand else 0
        for i in range(hand_size):
            if action & (1 << i):
                cards.append(i)

        # return the correct string the API expects
        return "play_hand", cards

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, seed=None, options=None):
        if self.client is None:
            self.client = BalatroClient(port=self.port)

        # Connect to the server
        try:
            self.client.connect()
        except Exception as e:
            logger.error("Failed to connect to BalatroBot API: %s", e)
            raise

        super().reset(seed=seed)
        self.steps = 0
        self.episode_reward = 0.0

        # Go to menu
        try:
            self.client.send_message("go_to_menu", {})
        except BalatroError:
            pass

        # Start new run
        try:
            response = self.client.send_message(
                "start_run",
                {"deck": self.deck, "stake": self.stake, "seed": self.game_seed},
            )
            self.current_state = G(**response)
        except BalatroError as e:
            logger.error("Failed to start run: %s", e)
            self.current_state = None

        return self._get_obs(), self._get_info()

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action: int):
        self.steps += 1

        self._apply_action(action)
        reward = self._compute_reward()

        terminated = self._terminal()
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ============================================================
    # ACTION HANDLING
    # ============================================================
    def _apply_action(self, action: int):
        if self.current_state is None:
            return

        assert self.client is not None, "Client must be initialized in reset() before calling step()"

        state = self.current_state.state

        try:
            if state == State.MENU.value:
                resp = self.client.send_message(
                    "start_run",
                    {"deck": self.deck, "stake": self.stake, "seed": self.game_seed},
                )
                self.current_state = G(**resp)

            elif state == State.BLIND_SELECT.value:
                # Map: 0=skip, 1=select
                act = "skip" if action == 0 else "select"
                resp = self.client.send_message("skip_or_select_blind", {"action": act})
                self.current_state = G(**resp)

            elif state == State.SELECTING_HAND.value:
                act, cards = self._hand_action_from_index(action)
                resp = self.client.send_message(
                    "play_hand_or_discard", {"action": act, "cards": cards}
                )
                self.current_state = G(**resp)

            elif state in (State.HAND_PLAYED.value, State.DRAW_TO_HAND.value):
                resp = self.client.send_message("get_game_state", {})
                self.current_state = G(**resp)

            elif state == State.ROUND_EVAL.value:
                resp = self.client.send_message("cash_out", {})
                self.current_state = G(**resp)

            elif state == State.SHOP.value:
                resp = self._apply_shop_action(action)
                self.current_state = G(**resp)

            else:
                resp = self.client.send_message("get_game_state", {})
                self.current_state = G(**resp)

        except BalatroError:
            # Recovery: just refresh game state
            resp = self.client.send_message("get_game_state", {})
            self.current_state = G(**resp)

    def _apply_shop_action(self, action: int):
        assert self.client is not None, "Client must be initialized in reset() before calling step()"
        assert self.current_state is not None, "No game state available"
        assert self.current_state.game is not None, "No game state game available"

        shop_cards: Any | list[Any] = getattr(self.current_state.game.shop, "cards", [])
        if action == 0:
            return self.client.send_message("shop", {"action": "next_round"})
        if 1 <= action <= len(shop_cards):
            return self.client.send_message(
                "shop", {"action": "buy_card", "index": action - 1}
            )
        return self.client.send_message("shop", {"action": "next_round"})

    # ============================================================
    # REWARD + TERMINATION
    # ============================================================
    def _compute_reward(self):
        if not self.current_state or not self.current_state.game:
            return 0.0
        return float(self.current_state.game.chips)

    def _terminal(self):
        if not self.current_state:
            return False
        if self.current_state.state == State.GAME_OVER.value:
            return True
        if self.current_state.game and self.current_state.game.won:
            return True
        return False

    # ============================================================
    # OBS + INFO
    # ============================================================
    def _get_obs(self):
        if self.current_state is None:
            return {
                "state": np.array([State.MENU.value]),
                "chips": np.array([0.0]),
                "dollars": np.array([0.0]),
                "round": np.array([0.0]),
                "hands_left": np.array([0.0]),
                "discards_left": np.array([0.0]),
                "hand_size": np.array([0.0]),
                "joker_count": np.array([0.0]),
                "deck_vector": self.deck_vector,
            }

        assert self.current_state is not None
        assert self.current_state.game is not None

        game = self.current_state.game
        hand = self.current_state.hand

        return {
            "state": np.array([self.current_state.state], dtype=np.float32),
            "chips": np.array([float(game.chips)], dtype=np.float32),
            "dollars": np.array([float(game.dollars)], dtype=np.float32),
            "round": np.array([float(game.round)], dtype=np.float32),
            "hands_left": np.array([float(game.current_round.hands_left)], dtype=np.float32)
            if game.current_round else np.array([0.0]),
            "discards_left": np.array([float(game.current_round.discards_left)], dtype=np.float32)
            if game.current_round else np.array([0.0]),
            "hand_size": np.array([float(hand.config.card_count)] if hand and hand.config else [0.0]),
            "joker_count": np.array([len(self.current_state.jokers)], dtype=np.float32),
            "deck_vector": self.deck_vector,
        }

    def _get_info(self):
        return {
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "raw_state": self.current_state.model_dump() if self.current_state else None,
        }

    # ============================================================
    # RENDER / CLOSE
    # ============================================================
    def render(self):
        if self.render_mode == "human":
            print("Game state:", self.current_state)

    def close(self):
        """Clean up environment resources."""
        if self.client and self._connected:
            logger.info("Closing BalatroBot connection")
            self.client.disconnect()
            self._connected = False
