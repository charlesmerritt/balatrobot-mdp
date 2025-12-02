from __future__ import annotations

from dataclasses import dataclass
from typing import List

SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
RANKS = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
    "Ace",
]


@dataclass(frozen=True)
class Card:
    """Representation of a single card in the deck."""

    suit: str
    rank: str

    def as_dict(self) -> dict:
        return {"suit": self.suit, "rank": self.rank}


def get_standard_deck() -> List[Card]:
    """Return a standard 52-card deck composition."""

    return [Card(suit=suit, rank=rank) for suit in SUITS for rank in RANKS]


def get_standard_deck_vector() -> list[int]:
    """Return a vector representation of the standard deck (1 per card)."""

    return [1 for _ in range(len(SUITS) * len(RANKS))]
