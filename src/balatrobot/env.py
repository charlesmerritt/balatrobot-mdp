import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatrobot.hand_evaluator import evaluate_hand
from .client import BalatroClient
from .deck import get_standard_deck_vector
from .enums import Decks, Stakes, State
from .exceptions import BalatroError
from .models import G
from .utils import (
    joker_id_from_card,
    HANDNAME_TO_ID,
    RANK_TO_ID,
    SUIT_TO_ID,
    MAX_HAND_SIZE,
    MAX_JOKERS,
    eval_hand_features_from_hand,
    MAX_HAND_RANK_VALUE,
)


logger = logging.getLogger(__name__)


class BalatroEnv(gym.Env):
    """
    Gymnasium environment wrapper for BalatroBot.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        port: int = 12346,
        deck: str = Decks.RED.value,
        stake: int = Stakes.WHITE.value,
        seed: Optional[str] = None,
        max_steps: int = 20000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.port = port
        self.deck = deck
        self.stake = stake
        self.game_seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.last_error_penalty = 0.0
        self.prev_chips = 0

        self.client: Optional[BalatroClient] = None
        self.current_state: Optional[G] = None
        self.prev_state: Optional[int] = None

        # Episode tracking
        self.steps = 0
        self.episode_reward = 0.0

        # -----------------------------
        # DEBUGGING
        # -----------------------------
        self.policy = None

        # Deck vector (static feature: default 52 card deck)
        self.deck_vector = np.array(get_standard_deck_vector(), dtype=np.float32)

        # -----------------------------
        # DEFINE ACTION SPACE (GLOBAL)
        # -----------------------------
        # Max combinations of up to 5 cards from a hand size of 8, 8 choose 5! = 218, *2 for play or discard. +2 for planet cards = 438 each hand.
        MAX_HAND_ACTIONS = 438
        MAX_SHOP_ACTIONS = 44 # next round, reroll, buy item 1, buy item 2, buy voucher, buy pack 1, buy pack 2 + pack permutations
        # Pack permutations: There are standard, jumbo, and mega packs. Mega: 5 permute 2! + 2 skips = 27, Jumbo: 5 permute 1 + 1 skip = 6, Standard: 3 permute 1 + 1 skip = 4, total 37 from pack selection.
        # 37 (booster packs) + 7 base shop choices = 44 each reroll.
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
        "current_hand_type": spaces.Discrete(len(HANDNAME_TO_ID)),  # Lua-based
        "hand_cards": spaces.Box(
            low=0,
            high=13,
            shape=(MAX_HAND_SIZE, 2),
            dtype=np.float32,
        ),
        "blind_target": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
        "joker_ids": spaces.Box(
            low=0,
            high=np.inf,
            shape=(MAX_JOKERS,),
            dtype=np.float32,
        ),
        # Evaluator-based features
        "eval_hand_type": spaces.Discrete(len(HANDNAME_TO_ID)),
        "hand_target_rank": spaces.Box(
            0,
            MAX_HAND_RANK_VALUE,
            shape=(1,),
            dtype=np.float32,
        ),
        "hand_target_completeness": spaces.Box(
            0.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        ),
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

        if self.current_state is not None:
            self.prev_state = self.current_state.state
        else:
            self.prev_state = None

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

        except BalatroError as e:
            # Penalize invalid card selections
            if "Invalid number of cards" in str(e):
                self.last_error_penalty = -10.0

            # Penalize any error a little bit
            self.last_error_penalty += -10.0

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
        reward = 0.0

        if not self.current_state or not self.current_state.game:
            print("ERROR: No game state available")
            return reward

        game = self.current_state.game
        cr = game.current_round

        # 1) Primary signal: change in total chips
        # Chips only increase when a hand finishes scoring.
        delta_chips = float(game.chips) - float(self.prev_chips)
        if delta_chips > 0:
            reward += delta_chips
        self.prev_chips = float(game.chips)

        # 2) Optional mult-based bonus (keep, but don't rely on state gate)
        if cr and cr.current_hand and cr.current_hand.mult and cr.current_hand.mult > 0:
            reward += float(cr.current_hand.mult)

        # 3) Penalties from errors
        reward += self.last_error_penalty
        self.last_error_penalty = 0.0
        print(
            "DEBUG REWARD",
            "chips=", game.chips,
            "prev_chips=", self.prev_chips,
            "delta=", delta_chips,
            "mult=", cr.current_hand.mult if cr and cr.current_hand else None,
        )
        return float(reward)

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
                "state": np.array([State.MENU.value], dtype=np.float32),
                "chips": np.array([0.0], dtype=np.float32),
                "dollars": np.array([0.0], dtype=np.float32),
                "round": np.array([0.0], dtype=np.float32),
                "hands_left": np.array([0.0], dtype=np.float32),
                "discards_left": np.array([0.0], dtype=np.float32),
                "hand_size": np.array([0.0], dtype=np.float32),
                "joker_count": np.array([0.0], dtype=np.float32),
                "deck_vector": self.deck_vector,
                "current_hand_type": np.array([0.0], dtype=np.float32),
                "hand_cards": np.zeros((MAX_HAND_SIZE, 2), dtype=np.float32),
                "blind_target": np.array([0.0], dtype=np.float32),
                "joker_ids": np.zeros((MAX_JOKERS,), dtype=np.float32),
                "eval_hand_type": np.array([0.0], dtype=np.float32),
                "hand_target_rank": np.array([0.0], dtype=np.float32),
                "hand_target_completeness": np.array([0.0], dtype=np.float32),
            }

        assert self.current_state is not None
        assert self.current_state.game is not None

        game = self.current_state.game
        hand = self.current_state.hand
        cr = game.current_round

        # Lua-reported hand type
        lua_hand_type_id = 0
        if cr and cr.current_hand and cr.current_hand.handname:
            lua_hand_type_id = HANDNAME_TO_ID.get(cr.current_hand.handname, 0)

        # Build hand_cards array: shape (MAX_HAND_SIZE, 2)
        hand_cards_arr = np.zeros((MAX_HAND_SIZE, 2), dtype=np.float32)
        if hand and hand.cards:
            for i, card in enumerate(hand.cards[:MAX_HAND_SIZE]):
                base = getattr(card, "base", None)
                if not base:
                    continue
                suit = getattr(base, "suit", "")
                value = getattr(base, "value", "")
                rank_id = RANK_TO_ID.get(str(value), 0)
                suit_id = SUIT_TO_ID.get(str(suit), 0)
                hand_cards_arr[i, 0] = rank_id
                hand_cards_arr[i, 1] = suit_id

        # Blind target
        blind_target = 0.0
        blinds = getattr(self.current_state, "blinds", None)
        if blinds and game.blind_on_deck:
            blind_name = game.blind_on_deck  # "Small", "Big", or boss key/name
            key = blind_name.lower()
            blind_info = getattr(blinds, key, None)
            if blind_info and blind_info.score is not None:
                blind_target = float(blind_info.score)

        # Joker IDs
        jokers = getattr(self.current_state, "jokers", None)
        joker_ids_arr = np.zeros((MAX_JOKERS,), dtype=np.float32)
        if isinstance(jokers, dict) and "cards" in jokers:
            cards = jokers["cards"]
        elif jokers is not None and hasattr(jokers, "cards"):
            cards = jokers.cards
        else:
            cards = []
        for i, card in enumerate(list(cards)[:MAX_JOKERS]):
            joker_ids_arr[i] = float(joker_id_from_card(card))

        # Evaluator-based hand features
        eval_hand_type_id, hand_target_rank, hand_target_completeness = \
            eval_hand_features_from_hand(hand)

        return {
            "state": np.array([self.current_state.state], dtype=np.float32),
            "chips": np.array([float(game.chips)], dtype=np.float32),
            "dollars": np.array([float(game.dollars)], dtype=np.float32),
            "round": np.array([float(game.round)], dtype=np.float32),
            "hands_left": np.array(
                [float(game.current_round.hands_left)], dtype=np.float32
            )
            if game.current_round
            else np.array([0.0], dtype=np.float32),
            "discards_left": np.array(
                [float(game.current_round.discards_left)], dtype=np.float32
            )
            if game.current_round
            else np.array([0.0], dtype=np.float32),
            "hand_size": np.array(
                [float(hand.config.card_count)] if hand and hand.config else [0.0],
                dtype=np.float32,
            ),
            "joker_count": np.array(
                [len(getattr(self.current_state, "jokers", []) or [])], dtype=np.float32
            ),
            "deck_vector": self.deck_vector,
            "current_hand_type": np.array([float(lua_hand_type_id)], dtype=np.float32),
            "hand_cards": hand_cards_arr,
            "blind_target": np.array([blind_target], dtype=np.float32),
            "joker_ids": joker_ids_arr,
            "eval_hand_type": np.array([float(eval_hand_type_id)], dtype=np.float32),
            "hand_target_rank": np.array([float(hand_target_rank)], dtype=np.float32),
            "hand_target_completeness": np.array(
                [float(hand_target_completeness)], dtype=np.float32
            ),
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
