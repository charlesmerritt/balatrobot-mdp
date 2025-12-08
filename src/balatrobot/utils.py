import numpy as np
from typing import Any, Optional
from .deck import SUITS, RANKS
from .hand_evaluator import evaluate_hand, get_best_hand_target, HAND_RANKS


# Design decision, use Banner to limit jokers and vouchers that would alter this.
MAX_HAND_SIZE = 8
# Design decision, use Banner to limit negatives that would alter this.
MAX_JOKERS = 5

MAX_HAND_RANK_VALUE = max(HAND_RANKS.values())

HANDNAME_TO_ID = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
    "Royal Flush": 9,
    "Five of a Kind": 10,
    "Flush House": 11,
    "Flush Five": 12,
}

RANK_TO_ID = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}

SUIT_TO_ID = {
    "Spades": 0,
    "Hearts": 1,
    "Diamonds": 2,
    "Clubs": 3,
}

JOKER_KEY_TO_ID: dict[str, int] = {}

JOKER_UNKNOWN_ID = 0

def joker_id_from_card(card) -> int:
    config = getattr(card, "config", None)
    center = getattr(config, "center", None) if config is not None else None
    key = ""
    if isinstance(center, dict):
        key = str(center.get("key") or center.get("id") or "")
    if not key:
        return JOKER_UNKNOWN_ID
    if key not in JOKER_KEY_TO_ID:
        JOKER_KEY_TO_ID[key] = len(JOKER_KEY_TO_ID) + 1  # start at 1
    return JOKER_KEY_TO_ID[key]


def encode_hand_cards(hand: Any, max_hand_size: int = MAX_HAND_SIZE) -> np.ndarray:
    """Encode up to ``max_hand_size`` cards from the current hand as (rank_id, suit_id).

    The mapping uses the same ordering as ``SUITS`` and ``RANKS`` in ``deck.py`` so
    that indices are consistent with the deck vector representation.
    """

    arr = np.zeros((max_hand_size, 2), dtype=np.float32)
    if not hand or not getattr(hand, "cards", None):
        return arr

    for i, card in enumerate(hand.cards[:max_hand_size]):
        base = getattr(card, "base", None)
        if not base:
            continue
        suit = str(getattr(base, "suit", ""))
        value = str(getattr(base, "value", ""))
        try:
            suit_id = SUITS.index(suit)
        except ValueError:
            suit_id = 0
        try:
            rank_id = RANKS.index(value)
        except ValueError:
            rank_id = 0
        arr[i, 0] = float(rank_id)
        arr[i, 1] = float(suit_id)

    return arr


def card_index_from_suit_rank(suit: str, rank: str) -> Optional[int]:
    """Return linear deck index for a (suit, rank) pair using SUITS/RANKS order.

    Returns ``None`` if the suit or rank is unknown.
    """

    try:
        s = SUITS.index(suit)
        r = RANKS.index(rank)
    except ValueError:
        return None
    return s * len(RANKS) + r


def compute_blind_target(state: Any) -> float:
    """Compute current blind chip target from the game state, if available."""

    game = getattr(state, "game", None)
    blinds = getattr(state, "blinds", None)
    if not game or not blinds or not getattr(game, "blind_on_deck", None):
        return 0.0

    blind_name = str(game.blind_on_deck)
    key = blind_name.lower()
    blind_info = getattr(blinds, key, None)
    score = getattr(blind_info, "score", None) if blind_info is not None else None
    return float(score) if score is not None else 0.0


def update_deck_vector_for_shop_buy(deck_vector: np.ndarray, shop_cards: Any, index: int) -> None:
    """Update ``deck_vector`` belief when a playing card is purchased from the shop.

    This assumes ``deck_vector`` follows the SUITS/RANKS ordering from ``deck.py``.
    Non-playing cards are ignored.
    """

    if not shop_cards or index < 0 or index >= len(shop_cards):
        return

    card = shop_cards[index]
    base = getattr(card, "base", None)
    if not base:
        return

    suit = str(getattr(base, "suit", ""))
    rank = str(getattr(base, "value", ""))
    idx = card_index_from_suit_rank(suit, rank)
    if idx is None:
        return

    if 0 <= idx < deck_vector.shape[0]:
        deck_vector[idx] += 1.0

def to_evaluator_cards_from_hand(hand: Any) -> list[dict[str, Any]]:
    """Convert a Balatro hand object into the dict format expected by hand_evaluator.

    Each card becomes: {"base": {"value": <rank_str>, "suit": <suit_str>}}.
    """
    cards: list[dict[str, Any]] = []
    if not hand or not getattr(hand, "cards", None):
        return cards

    for card in hand.cards:
        base = getattr(card, "base", None)
        if not base:
            continue
        value = getattr(base, "value", None)
        suit = getattr(base, "suit", None)
        if value is None or suit is None:
            continue
        cards.append({"base": {"value": str(value), "suit": str(suit)}})
    return cards


def eval_hand_features_from_hand(hand: Any) -> tuple[int, int, float]:
    """Return evaluator-based features (hand_type_id, target_rank, target_completeness).

    - hand_type_id: integer ID from HANDNAME_TO_ID for the evaluated 5-card hand
    - target_rank: rank_value of the best near-complete hand (HandTarget.rank_value)
    - target_completeness: HandTarget.completeness in [0, 1]
    """
    cards = to_evaluator_cards_from_hand(hand)
    if not cards:
        return 0, 0, 0.0

    # Primary hand type from evaluator
    hand_name, _rank_value = evaluate_hand(cards)
    hand_type_id = HANDNAME_TO_ID.get(hand_name, 0)

    # Target info (near-complete strongest hand)
    target = get_best_hand_target(cards)
    target_rank = int(target.rank_value)
    target_completeness = float(target.completeness)

    return hand_type_id, target_rank, target_completeness
