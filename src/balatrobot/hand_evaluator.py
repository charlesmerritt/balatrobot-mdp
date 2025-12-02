"""Hand evaluation utilities for selecting best poker hands in Balatro."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

# Balatro hand rankings (higher = better)
HAND_RANKS = {
    "Flush Five": 11,
    "Flush House": 10,
    "Five of a Kind": 9,
    "Straight Flush": 8,
    "Four of a Kind": 7,
    "Full House": 6,
    "Flush": 5,
    "Straight": 4,
    "Three of a Kind": 3,
    "Two Pair": 2,
    "Pair": 1,
    "High Card": 0,
}

# Card rank values for straights (Ace can be high or low)
RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 11,
    "Queen": 12,
    "King": 13,
    "Ace": 14,
}

RANK_VALUES_ACE_LOW = {
    "Ace": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 11,
    "Queen": 12,
    "King": 13,
}

VALUE_TO_RANK = {value: rank for rank, value in RANK_VALUES.items()}


@dataclass
class HandTarget:
    """Represents a target poker hand and its completeness.

    Attributes:
        name: Hand name (e.g., "Straight", "Flush").
        rank_value: Numerical strength based on HAND_RANKS.
        completeness: Percentage of completion (0-1).
        contributing_indices: Card indices contributing toward the target hand.
    """

    name: str
    rank_value: int
    completeness: float
    contributing_indices: list[int]


def evaluate_hand(cards: list[dict[str, Any]]) -> tuple[str, int]:
    """Evaluate a poker hand and return its type and rank.

    Args:
        cards: List of card objects from game state

    Returns:
        Tuple of (hand_name, rank_value)
    """
    if not cards or len(cards) == 0:
        return "High Card", 0

    # Extract ranks and suits
    ranks = []
    suits = []

    for card in cards:
        rank = get_card_rank(card)
        suit = get_card_suit(card)
        if rank:
            ranks.append(rank)
        if suit:
            suits.append(suit)

    if not ranks:
        return "High Card", 0

    # Count occurrences
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Check for flush
    is_flush = len(suit_counts) == 1 and len(cards) >= 5

    # Check for straight
    is_straight = False
    if len(cards) >= 5:
        is_straight = _is_straight(ranks)

    # Get count patterns
    counts = sorted(rank_counts.values(), reverse=True)

    # Evaluate hand type (Balatro-specific hands)
    if len(cards) == 5:
        if counts == [5]:
            if is_flush:
                return "Flush Five", HAND_RANKS["Flush Five"]
            return "Five of a Kind", HAND_RANKS["Five of a Kind"]

        if counts == [3, 2]:
            if is_flush:
                return "Flush House", HAND_RANKS["Flush House"]
            return "Full House", HAND_RANKS["Full House"]

        if is_straight and is_flush:
            return "Straight Flush", HAND_RANKS["Straight Flush"]

        if counts == [4, 1]:
            return "Four of a Kind", HAND_RANKS["Four of a Kind"]

        if is_flush:
            return "Flush", HAND_RANKS["Flush"]

        if is_straight:
            return "Straight", HAND_RANKS["Straight"]

        if counts == [3, 1, 1]:
            return "Three of a Kind", HAND_RANKS["Three of a Kind"]

        if counts == [2, 2, 1]:
            return "Two Pair", HAND_RANKS["Two Pair"]

        if counts == [2, 1, 1, 1]:
            return "Pair", HAND_RANKS["Pair"]

    # For less than 5 cards, check basic patterns
    if counts[0] >= 4:
        return "Four of a Kind", HAND_RANKS["Four of a Kind"]
    if counts[0] >= 3:
        if len(counts) >= 2 and counts[1] >= 2:
            return "Full House", HAND_RANKS["Full House"]
        return "Three of a Kind", HAND_RANKS["Three of a Kind"]
    if counts[0] >= 2:
        if len(counts) >= 2 and counts[1] >= 2:
            return "Two Pair", HAND_RANKS["Two Pair"]
        return "Pair", HAND_RANKS["Pair"]

    return "High Card", HAND_RANKS["High Card"]


def _is_straight(ranks: list[str]) -> bool:
    """Check if ranks form a straight.

    Args:
        ranks: List of card rank strings

    Returns:
        True if ranks form a straight
    """
    if len(ranks) < 5:
        return False

    # Convert to numeric values
    values = [RANK_VALUES.get(r, 0) for r in ranks if r in RANK_VALUES]
    if len(values) < 5:
        return False

    values = sorted(set(values))

    # Check regular straight
    for i in range(len(values) - 4):
        if values[i + 4] - values[i] == 4:
            return True

    # Check Ace-low straight (A-2-3-4-5)
    ace_low_values = [
        RANK_VALUES_ACE_LOW.get(r, 0) for r in ranks if r in RANK_VALUES_ACE_LOW
    ]
    ace_low_values = sorted(set(ace_low_values))
    for i in range(len(ace_low_values) - 4):
        if ace_low_values[i + 4] - ace_low_values[i] == 4:
            return True

    return False


def select_best_hand(cards: list[dict[str, Any]], hand_size: int = 5) -> list[int]:
    """Backwards-compatible helper returning best hand indices only."""
    indices, _, _ = get_best_hand_info(cards, hand_size)
    return indices


def get_best_hand_info(
    cards: list[dict[str, Any]],
    hand_size: int = 5,
) -> tuple[list[int], str, int]:
    """Select the best card combination and return extra metadata.

    Args:
        cards: List of all cards in hand from game state
        hand_size: Number of cards to select (default 5)

    Returns:
        Tuple of (card indices, hand name, rank value)
    """
    if not cards:
        return [0], "High Card", HAND_RANKS["High Card"]

    # If we have <= hand_size cards, play them all but still evaluate
    if len(cards) <= hand_size:
        hand_name, rank = evaluate_hand(cards)
        return list(range(len(cards))), hand_name, rank

    best_rank = -1
    best_indices = list(range(hand_size))
    best_name = "High Card"

    for combo in combinations(range(len(cards)), hand_size):
        selected_cards = [cards[i] for i in combo]
        hand_name, rank = evaluate_hand(selected_cards)

        if rank > best_rank:
            best_rank = rank
            best_name = hand_name
            best_indices = list(combo)

    return best_indices, best_name, best_rank


def get_best_hand_target(cards: list[dict[str, Any]]) -> HandTarget:
    """Return the most valuable near-complete hand target for hybrid policy."""

    default_indices = list(range(len(cards)))
    best_target = HandTarget(
        name="High Card",
        rank_value=HAND_RANKS["High Card"],
        completeness=0.0,
        contributing_indices=default_indices,
    )

    if not cards:
        return best_target

    rank_to_indices = _group_by_rank(cards)
    suit_to_indices = _group_by_suit(cards)

    # Straight flush and flush evaluation
    for suit, indices in suit_to_indices.items():
        if len(indices) < 2:
            continue
        straight_length, straight_indices = _find_best_straight_indices(
            rank_to_indices, indices
        )
        completeness = min(straight_length / 5.0, 1.0)
        best_target = _consider_target(
            best_target,
            "Straight Flush",
            completeness,
            straight_indices,
        )

        flush_completeness = min(len(indices) / 5.0, 1.0)
        best_target = _consider_target(
            best_target,
            "Flush",
            flush_completeness,
            indices,
        )

    # Straight evaluation
    straight_length, straight_indices = _find_best_straight_indices(rank_to_indices)
    best_target = _consider_target(
        best_target,
        "Straight",
        min(straight_length / 5.0, 1.0),
        straight_indices,
    )

    # Set-based evaluations (Four of a Kind, Full House, etc.)
    best_target = _evaluate_set_targets(best_target, rank_to_indices)

    return best_target


def get_flush_draw_indices(
    cards: Iterable[dict[str, Any]],
) -> tuple[str | None, list[int]]:
    """Return suit and indices if there's a flush draw (>=4 cards of same suit)."""

    suits = [get_card_suit(card) for card in cards]
    suit_counts = Counter(suits)
    suit_counts.pop(None, None)

    if not suit_counts:
        return None, []

    suit, count = suit_counts.most_common(1)[0]
    if not suit or count < 4:
        return None, []

    indices = [i for i, card in enumerate(cards) if get_card_suit(card) == suit]
    return suit, indices


def get_card_rank(card: dict[str, Any]) -> str | None:
    base = card.get("base") if card else None
    return base.get("value") if isinstance(base, dict) else None


def get_card_suit(card: dict[str, Any]) -> str | None:
    base = card.get("base") if card else None
    return base.get("suit") if isinstance(base, dict) else None


def _group_by_rank(cards: list[dict[str, Any]]) -> dict[str, list[int]]:
    rank_to_indices: dict[str, list[int]] = {}
    for idx, card in enumerate(cards):
        rank = get_card_rank(card)
        if not rank:
            continue
        rank_to_indices.setdefault(rank, []).append(idx)
    return rank_to_indices


def _group_by_suit(cards: list[dict[str, Any]]) -> dict[str, list[int]]:
    suit_to_indices: dict[str, list[int]] = {}
    for idx, card in enumerate(cards):
        suit = get_card_suit(card)
        if not suit:
            continue
        suit_to_indices.setdefault(suit, []).append(idx)
    return suit_to_indices


def _find_best_straight_indices(
    rank_to_indices: dict[str, list[int]],
    subset_indices: list[int] | None = None,
) -> tuple[int, list[int]]:
    """Return the length and indices of the best straight (>=3 cards)."""

    value_rank_pairs = []
    for rank, indices in rank_to_indices.items():
        filtered = [i for i in indices if subset_indices is None or i in subset_indices]
        if not filtered:
            continue
        value = RANK_VALUES.get(rank)
        if value is None:
            continue
        value_rank_pairs.append((value, rank))

    if not value_rank_pairs:
        return 0, []

    values = sorted({value for value, _ in value_rank_pairs})
    best_sequence = _longest_consecutive_sequence(values)

    # Handle Ace-low straights
    if 14 in values:
        ace_low_values = sorted(set(values + [1]))
        ace_sequence = _longest_consecutive_sequence(ace_low_values)
        if len(ace_sequence) > len(best_sequence):
            best_sequence = [14 if v == 1 else v for v in ace_sequence]

    if len(best_sequence) < 3:
        return len(best_sequence), []

    indices: list[int] = []
    used_indices: set[int] = set()
    for value in best_sequence:
        rank = _value_to_rank(value)
        if not rank or rank not in rank_to_indices:
            continue
        for idx in rank_to_indices[rank]:
            if idx in used_indices:
                continue
            if subset_indices is not None and idx not in subset_indices:
                continue
            indices.append(idx)
            used_indices.add(idx)
            break

    return len(best_sequence), indices


def _longest_consecutive_sequence(values: list[int]) -> list[int]:
    if not values:
        return []

    best = current = [values[0]]
    for value in values[1:]:
        if value == current[-1]:
            continue
        if value == current[-1] + 1:
            current.append(value)
        else:
            if len(current) > len(best):
                best = current[:]
            current = [value]
    if len(current) > len(best):
        best = current
    return best


def _value_to_rank(value: int) -> str | None:
    if value == 1:
        return "Ace"
    return VALUE_TO_RANK.get(value)


def _evaluate_set_targets(
    best_target: HandTarget,
    rank_to_indices: dict[str, list[int]],
) -> HandTarget:
    counts = sorted(
        ((rank, indices) for rank, indices in rank_to_indices.items()),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    # Four of a Kind
    for rank, indices in counts:
        count = len(indices)
        completeness = min(count / 4.0, 1.0)
        best_target = _consider_target(
            best_target,
            "Four of a Kind",
            completeness,
            indices[: min(4, count)],
        )

    # Full House & Three of a Kind
    for i, (triple_rank, triple_indices) in enumerate(counts):
        triple_count = len(triple_indices)
        if triple_count < 2:
            continue

        for j, (pair_rank, pair_indices) in enumerate(counts):
            if i == j:
                continue
            pair_count = len(pair_indices)
            if pair_count < 2:
                continue

            used_triple = triple_indices[: min(3, triple_count)]
            used_pair = pair_indices[: min(2, pair_count)]
            completeness = min((len(used_triple) + len(used_pair)) / 5.0, 1.0)
            best_target = _consider_target(
                best_target,
                "Full House",
                completeness,
                used_triple + used_pair,
            )
            break  # Only need the best pair partner

        # Three of a Kind target
        completeness = min(triple_count / 3.0, 1.0)
        best_target = _consider_target(
            best_target,
            "Three of a Kind",
            completeness,
            triple_indices[: min(3, triple_count)],
        )

    # Two Pair target
    pair_candidates = [indices[:2] for _, indices in counts if len(indices) >= 2]
    if len(pair_candidates) >= 2:
        used = pair_candidates[0] + pair_candidates[1]
        completeness = min(len(used) / 4.0, 1.0)
        best_target = _consider_target(
            best_target,
            "Two Pair",
            completeness,
            used,
        )

    return best_target


def _consider_target(
    current_target: HandTarget,
    hand_name: str,
    completeness: float,
    indices: list[int],
) -> HandTarget:
    if not indices or completeness <= 0:
        return current_target

    rank_value = HAND_RANKS.get(hand_name)
    if rank_value is None:
        return current_target

    if rank_value > current_target.rank_value or (
        rank_value == current_target.rank_value
        and completeness > current_target.completeness
    ):
        return HandTarget(hand_name, rank_value, completeness, indices)

    return current_target
