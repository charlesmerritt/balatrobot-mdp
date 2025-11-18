from __future__ import annotations

import argparse
import logging
import random
import string
from typing import Literal

from balatrobot import BalatroEnv
from balatrobot.enums import Decks, Stakes

POLICIES = ("random", "greedy")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a BalatroBot policy against a running Balatro instance."
    )
    parser.add_argument("--port", type=int, default=12347, help="BalatroBot TCP port")
    parser.add_argument(
        "--deck",
        choices=[deck.value for deck in Decks],
        default=Decks.RED.value,
        help="Starting deck",
    )
    parser.add_argument(
        "--stake",
        type=int,
        choices=[stake.value for stake in Stakes],
        default=Stakes.WHITE.value,
        help="Stake difficulty",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="rng",
        help="Run seed; pass 'rng' to auto-generate a random 7-character seed",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="greedy",
        help="Policy strategy to execute",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps before stopping"
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", None],
        default="human",
        help="Rendering mode",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Disable rendering entirely"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-step action logs (hand + selected cards)",
    )
    return parser.parse_args()


def run_policy(args: argparse.Namespace) -> None:
    seed = resolve_seed(args.seed)
    render_mode: Literal["human", "rgb_array"] | None
    if args.headless:
        render_mode = None
    else:
        render_mode = args.render_mode

    env = BalatroEnv(
        port=args.port,
        deck=args.deck,
        stake=args.stake,
        seed=seed,
        max_steps=args.max_steps,
        render_mode=render_mode,
        policy=args.policy,
    )

    total_reward = 0.0
    step = 0
    logger.info(
        "Running policy=%s deck=%s stake=%s seed=%s",
        args.policy,
        args.deck,
        args.stake,
        seed,
    )

    try:
        obs, info = env.reset()
        logger.debug("Initial observation: %s", obs)

        done = False
        while not done and step < args.max_steps:
            _, reward, terminated, truncated, info = env.step(0)
            if args.verbose:
                log_action(env, step)
            total_reward += reward
            step += 1
            done = terminated or truncated
    finally:
        env.close()

    logger.info("Episode finished after %s steps", step)
    logger.info("Total reward: %.2f", total_reward)


def resolve_seed(seed_arg: str | None) -> str | None:
    if seed_arg is None:
        return None

    if seed_arg.lower() == "rng":
        alphabet = string.ascii_uppercase + string.digits
        return "".join(random.choices(alphabet, k=7))

    return seed_arg


def log_action(env: BalatroEnv, step: int) -> None:
    log = env.get_last_action_log()
    if not log:
        return

    hand_str = ", ".join(log.get("hand", []))
    selected_str = ", ".join(log.get("selected", []))
    logger.info(
        "Step %d | policy=%s | action=%s | hand=[%s] | selected=[%s]",
        step,
        log.get("policy"),
        log.get("action"),
        hand_str,
        selected_str,
    )


if __name__ == "__main__":
    run_policy(parse_args())
