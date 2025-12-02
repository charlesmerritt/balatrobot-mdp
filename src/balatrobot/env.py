"""Gymnasium MDP environment for Balatro game using BalatroBot."""

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import BalatroClient
from .enums import Decks, Stakes, State
from .exceptions import BalatroError
from .deck import get_standard_deck_vector
from .hand_evaluator import get_best_hand_info
from .models import G

logger = logging.getLogger(__name__)


class BalatroEnv(gym.Env):
    """Gymnasium environment for Balatro game.

    This environment wraps the BalatroBot client to provide a standard
    Gymnasium interface for reinforcement learning agents.

    The MDP structure follows these state transitions:
    - MENU -> start_run() -> BLIND_SELECT
    - BLIND_SELECT -> skip_or_select_blind() -> SELECTING_HAND
    - SELECTING_HAND -> play_hand_or_discard() -> HAND_PLAYED/DRAW_TO_HAND
    - After round completion -> cash_out() -> SHOP
    - SHOP -> shop(next_round) -> BLIND_SELECT
    - Terminal states: GAME_OVER, or when episode limit reached

    Observation Space:
        Dict containing game state information including:
        - chips: current chips
        - dollars: current money
        - round: current round number
        - hands_left: hands remaining
        - discards_left: discards remaining
        - hand_cards: list of cards in hand
        - jokers: list of active jokers
        - state: current game phase

    Action Space:
        Discrete actions depending on current game state:
        - In BLIND_SELECT: [0=skip, 1=select]
        - In SELECTING_HAND: [0-31] for card combinations + [32=pass]
        - In SHOP: [0=next_round, 1-N for shop actions]

    Rewards:
        - Positive reward for chips gained
        - Positive reward for advancing rounds
        - Negative reward for losing
        - Small negative reward per step (time penalty)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    VALID_POLICIES = {"random", "greedy"}

    def __init__(
        self,
        port: int = 12347,
        deck: str = Decks.RED.value,
        stake: int = Stakes.WHITE.value,
        seed: str | None = None,
        max_steps: int = 1000,
        render_mode: str | None = None,
        policy: str = "greedy",
    ):
        """Initialize Balatro Gymnasium environment.

        Args:
            port: Port to connect to BalatroBot
            deck: Starting deck name
            stake: Difficulty stake (1-8)
            seed: Optional game seed for reproducibility
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()

        self.port = port
        self.deck = deck
        self.stake = stake
        self.game_seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        policy_normalized = policy.lower()
        if policy_normalized not in self.VALID_POLICIES:
            raise ValueError(
                f"Invalid policy '{policy}'. Must be one of {sorted(self.VALID_POLICIES)}"
            )
        self.policy = policy_normalized
        self.deck_vector = np.array(get_standard_deck_vector(), dtype=np.float32)
        self._last_action_log: dict[str, Any] | None = None

        # Initialize client (connection happens in reset)
        self.client: BalatroClient | None = None
        self._connected = False

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_chips = 0
        self.last_dollars = 0
        self.last_round = 0
        self.last_ante = 0

        # Game state tracking
        self.current_state: G | None = None

        # Define action and observation spaces
        # Action space is simplified - we'll use a discrete action space
        # that maps to different actions depending on game state
        self.action_space = spaces.Discrete(64)  # Extended action space

        # Observation space - dictionary of game state features
        self.observation_space = spaces.Dict({
            "state": spaces.Discrete(28),  # Game state enum
            "chips": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "dollars": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "round": spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            "hands_left": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "discards_left": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "hand_size": spaces.Box(0, 20, shape=(1,), dtype=np.float32),
            "joker_count": spaces.Box(0, 10, shape=(1,), dtype=np.float32),
            "deck_vector": spaces.Box(0, 1, shape=(52,), dtype=np.float32),
        })

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Extract observation from current game state.

        Returns:
            Dictionary containing normalized game state features
        """
        if self.current_state is None:
            # Return default observation
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
            }

        state_value = self.current_state.state if self.current_state.state is not None else State.MENU.value
        game = self.current_state.game
        hand = self.current_state.hand

        # Extract game metrics
        chips = float(game.chips) if game else 0.0
        dollars = float(game.dollars) if game else 0.0
        round_num = float(game.round) if game else 0.0

        # Extract round metrics
        hands_left = 0.0
        discards_left = 0.0
        if game and game.current_round:
            hands_left = float(game.current_round.hands_left)
            discards_left = float(game.current_round.discards_left)

        # Extract hand size
        hand_size = 0.0
        if hand and hand.config:
            hand_size = float(hand.config.card_count)

        # Count jokers
        joker_count = 0.0
        if isinstance(self.current_state.jokers, list):
            joker_count = float(len(self.current_state.jokers))

        return {
            "state": np.array([state_value], dtype=np.float32),
            "chips": np.array([chips], dtype=np.float32),
            "dollars": np.array([dollars], dtype=np.float32),
            "round": np.array([round_num], dtype=np.float32),
            "hands_left": np.array([hands_left], dtype=np.float32),
            "discards_left": np.array([discards_left], dtype=np.float32),
            "hand_size": np.array([hand_size], dtype=np.float32),
            "joker_count": np.array([joker_count], dtype=np.float32),
            "deck_vector": self.deck_vector,
        }

    def _get_info(self) -> dict[str, Any]:
        """Get additional episode information.

        Returns:
            Dictionary with supplementary information
        """
        return {
            "episode_step": self.current_step,
            "episode_reward": self.episode_reward,
            "game_state": self.current_state.model_dump() if self.current_state else None,
        }

    def _calculate_reward(self) -> float:
        """Calculate reward based on state transition.

        Reward scheme:
        - 0.5 per chip earned
        - 1.0 per dollar earned
        - 10 * round_number per round passed (round 1: +10, round 2: +20, etc.)
        - 50 + (50 * ante_number) per ante passed (ante 1: +100, ante 2: +150, etc.)

        Returns:
            Reward value for the current step
        """
        if self.current_state is None or self.current_state.game is None:
            return 0.0

        reward = 0.0
        game = self.current_state.game

        # Reward for gaining chips: 0.5 per chip
        chips_gained = game.chips - self.last_chips
        reward += chips_gained * 0.5

        # Reward for gaining money: 1.0 per dollar
        dollars_gained = game.dollars - self.last_dollars
        reward += dollars_gained * 1.0

        # Bonus reward for advancing rounds: 10 * round_number
        if game.round > self.last_round:
            round_bonus = 10.0 * game.round
            reward += round_bonus
            logger.info(f"Advanced to round {game.round}! Bonus reward: +{round_bonus}")

        # Big bonus for advancing antes: 50 + (50 * ante_number)
        # In Balatro: round / 3 gives the current ante (rounds 1-3 = ante 1, etc.)
        current_ante = (game.round + 2) // 3  # Calculate ante from round
        if current_ante > self.last_ante:
            if current_ante > 1:
                ante_bonus = 50.0 + (50.0 * current_ante)
                reward += ante_bonus
                logger.info(
                    f"Advanced to ante {current_ante}! Bonus reward: +{ante_bonus}"
                )

        # Update tracking variables
        self.last_chips = game.chips
        self.last_dollars = game.dollars
        self.last_round = game.round
        self.last_ante = current_ante

        return reward

    def _is_terminal(self) -> bool:
        """Check if current state is terminal.

        Returns:
            True if episode should end
        """
        if self.current_state is None:
            return False

        # Check for game over state
        if self.current_state.state == State.GAME_OVER.value:
            logger.info("Episode terminated: GAME_OVER state reached")
            return True

        # Check step limit
        if self.current_step >= self.max_steps:
            logger.info(f"Episode terminated: max steps ({self.max_steps}) reached")
            return True

        # Check if won
        if self.current_state.game and self.current_state.game.won:
            logger.info("Episode terminated: Game won!")
            return True

        return False

    def _execute_action(self, action: int) -> None:
        """Execute action in the game based on current state.

        Args:
            action: Action index to execute

        Raises:
            BalatroError: If action execution fails
        """
        if not self.client:
            raise RuntimeError("Client not connected")

        state = self.current_state.state if self.current_state else State.MENU.value

        try:
            if state == State.MENU.value:
                # Start a new run from menu
                logger.debug("Action: Starting new run")
                response = self.client.send_message(
                    "start_run",
                    {"deck": self.deck, "stake": self.stake, "seed": self.game_seed}
                )
                self.current_state = G(**response)

            elif state == State.BLIND_SELECT.value:
                # Always select blind (never skip to avoid boss blind errors)
                logger.debug("Action: select blind")
                response = self.client.send_message(
                    "skip_or_select_blind",
                    {"action": "select"}
                )
                self.current_state = G(**response)

            elif state == State.SELECTING_HAND.value:
                action_name, cards = self._get_hand_action()
                response = self.client.send_message(
                    "play_hand_or_discard",
                    {"action": action_name, "cards": cards}
                )
                self.current_state = G(**response)

            elif state == State.HAND_PLAYED.value or state == State.DRAW_TO_HAND.value:
                # Wait for game to transition
                logger.debug("Action: Getting current state (waiting for transition)")
                response = self.client.send_message("get_game_state", {})
                self.current_state = G(**response)

            elif state == State.ROUND_EVAL.value:
                # Cash out after round
                logger.debug("Action: Cashing out")
                response = self.client.send_message("cash_out", {})
                self.current_state = G(**response)

            elif state == State.SHOP.value:
                if self.policy == "random":
                    shop_action = self._random_shop_action()
                else:
                    shop_action = self._greedy_shop_action()

                logger.debug("Action: Shop command %s", shop_action)
                response = self.client.send_message("shop", shop_action)
                self.current_state = G(**response)

            else:
                # For other states, just get current state
                logger.debug(f"Action: Getting state (current state: {state})")
                response = self.client.send_message("get_game_state", {})
                self.current_state = G(**response)

        except BalatroError as e:
            logger.warning(f"Action execution failed: {e}")
            # Don't raise - just get current state and continue
            response = self.client.send_message("get_game_state", {})
            self.current_state = G(**response)

    def _action_to_cards(self, action: int) -> list[int]:
        """Convert action index to card indices (legacy helper)."""
        if action == 0:
            return [0]

        cards = []
        for i in range(5):
            if action & (1 << i):
                cards.append(i)

        return cards if cards else [0]

    def get_last_action_log(self) -> dict[str, Any] | None:
        """Return details about the last action chosen by the policy."""

        return self._last_action_log

    def _get_hand_action(self) -> tuple[str, list[int]]:
        """Get action (play or discard) for the current hand based on policy."""

        hand_cards = self._get_current_hand_cards()
        if not hand_cards:
            return "play_hand", [0]

        card_dicts = [
            card.model_dump() if hasattr(card, "model_dump") else card
            for card in hand_cards
        ]

        if self.policy == "random":
            action = self._random_hand_action(len(hand_cards))
        else:
            action = self._greedy_hand_action(card_dicts)

        self._record_action_log(card_dicts, action)
        return action

    def _get_current_hand_cards(self) -> list:
        if (
            self.current_state
            and self.current_state.hand
            and self.current_state.hand.cards
        ):
            return list(self.current_state.hand.cards)
        return []

    def _random_hand_action(self, hand_size: int) -> tuple[str, list[int]]:
        if hand_size <= 0:
            return "play_hand", [0]

        action_name = random.choice(["play_hand", "discard"])
        max_select = min(5, hand_size)
        num_cards = random.randint(1, max_select)
        cards = random.sample(range(hand_size), num_cards)
        logger.debug(
            "Random policy selected %s with cards %s",
            action_name,
            cards,
        )
        return action_name, cards

    def _greedy_hand_action(self, card_dicts: list[dict[str, Any]]) -> tuple[str, list[int]]:
        indices, hand_name, _ = get_best_hand_info(card_dicts)
        logger.debug("Greedy policy playing %s with cards %s", hand_name, indices)
        return "play_hand", indices

    def _random_shop_action(self) -> dict[str, Any]:
        """Randomly choose a shop action; may buy a joker if affordable."""

        if not self.current_state or not self.current_state.game:
            return {"action": "next_round"}

        game = self.current_state.game
        shop_state = getattr(game, "shop", None)
        dollars = getattr(game, "dollars", 0)

        affordable_jokers: list[int] = []
        if shop_state and getattr(shop_state, "jokers", None):
            jokers_area = shop_state.jokers
            cards = getattr(jokers_area, "cards", []) or []
            for idx, card in enumerate(cards):
                ability = card.get("ability") if isinstance(card, dict) else None
                if ability and ability.get("set") == "Joker":
                    cost = card.get("cost", 0)
                    if cost <= dollars:
                        affordable_jokers.append(idx)

        if affordable_jokers:
            joker_index = random.choice(affordable_jokers)
            logger.debug(
                "Random policy buying joker at index %s from shop (affordable cards: %s)",
                joker_index,
                affordable_jokers,
            )
            return {"action": "buy_card", "index": joker_index}

        logger.debug("Random policy advancing to next round (no affordable jokers)")
        return {"action": "next_round"}

    def _greedy_shop_action(self) -> dict[str, Any]:
        """Greedy policy buys the most expensive affordable joker, otherwise advances."""

        if not self.current_state or not self.current_state.game:
            return {"action": "next_round"}

        game = self.current_state.game
        shop_state = getattr(game, "shop", None)
        dollars = getattr(game, "dollars", 0)

        best_index = None
        best_cost = -1

        if shop_state and getattr(shop_state, "jokers", None):
            jokers_area = shop_state.jokers
            cards = getattr(jokers_area, "cards", []) or []
            for idx, card in enumerate(cards):
                ability = card.get("ability") if isinstance(card, dict) else None
                if ability and ability.get("set") == "Joker":
                    cost = card.get("cost", 0)
                    if cost <= dollars and cost > best_cost:
                        best_cost = cost
                        best_index = idx

        if best_index is not None:
            logger.debug(
                "Greedy policy buying joker at index %s for %s dollars",
                best_index,
                best_cost,
            )
            return {"action": "buy_card", "index": best_index}

        logger.debug("Greedy policy advancing to next round (no affordable jokers)")
        return {"action": "next_round"}

    def _record_action_log(
        self,
        hand_cards: list[dict[str, Any]],
        action: tuple[str, list[int]],
    ) -> None:
        action_name, card_indices = action

        hand_labels = [self._format_card(card) for card in hand_cards]
        selected_labels = []
        for idx in card_indices:
            if 0 <= idx < len(hand_cards):
                selected_labels.append(hand_labels[idx])
            else:
                selected_labels.append(str(idx))

        self._last_action_log = {
            "policy": self.policy,
            "action": action_name,
            "indices": card_indices,
            "hand": hand_labels,
            "selected": selected_labels,
        }

    @staticmethod
    def _format_card(card: dict[str, Any]) -> str:
        base = card.get("base") if isinstance(card, dict) else None
        if not isinstance(base, dict):
            return "?"
        rank = base.get("value", "?")
        suit = base.get("suit", "?")
        suit_letter = suit[0] if isinstance(suit, str) and suit else "?"
        return f"{rank}{suit_letter}"

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_chips = 0
        self.last_dollars = 0
        self.last_round = 0
        self.last_ante = 0

        # Connect to client if not already connected
        if not self._connected:
            self.client = BalatroClient(port=self.port)
            self.client.connect()
            self._connected = True
            logger.info("Connected to BalatroBot on port %d", self.port)

        # Go to menu first
        try:
            self.client.send_message("go_to_menu", {})
        except BalatroError as e:
            logger.warning(f"Failed to go to menu: {e}")

        # Start a new run
        try:
            response = self.client.send_message(
                "start_run",
                {
                    "deck": self.deck,
                    "stake": self.stake,
                    "seed": self.game_seed,
                }
            )
            self.current_state = G(**response)
            logger.info(f"Started new run with {self.deck} at stake {self.stake}")
        except BalatroError as e:
            logger.error(f"Failed to start run: {e}")
            # Return default observation
            self.current_state = None

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Execute the action
        self._execute_action(action)

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if terminal
        terminated = self._is_terminal()
        truncated = False  # We use terminated for max_steps

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "human":
            # Print current game state
            if self.current_state:
                state_name = State(self.current_state.state).name if self.current_state.state else "UNKNOWN"
                print("\n=== Balatro Game State ===")
                print(f"State: {state_name}")
                if self.current_state.game:
                    print(f"Chips: {self.current_state.game.chips}")
                    print(f"Dollars: {self.current_state.game.dollars}")
                    print(f"Round: {self.current_state.game.round}")
                    if self.current_state.game.current_round:
                        print(f"Hands Left: {self.current_state.game.current_round.hands_left}")
                        print(f"Discards Left: {self.current_state.game.current_round.discards_left}")
                print(f"Episode Step: {self.current_step}/{self.max_steps}")
                print(f"Episode Reward: {self.episode_reward:.2f}")
                print("=" * 30)
            return None

        elif self.render_mode == "rgb_array":
            # Take a screenshot if client supports it
            if self.client:
                try:
                    screenshot_path = self.client.screenshot()
                    # Load and return as numpy array
                    from PIL import Image
                    img = Image.open(screenshot_path)
                    return np.array(img)
                except Exception as e:
                    logger.warning(f"Failed to capture screenshot: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        if self.client and self._connected:
            logger.info("Closing BalatroBot connection")
            self.client.disconnect()
            self._connected = False
