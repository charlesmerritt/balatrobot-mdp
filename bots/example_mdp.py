import logging

from balatrobot.enums import Decks, Stakes
from balatrobot.env import BalatroEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """Example of using the BalatroBot Gymnasium environment."""
    logger.info("BalatroBot Gymnasium Environment Example")

    # Create the environment
    env = BalatroEnv(
        port=12346,
        deck=Decks.RED.value,
        stake=Stakes.WHITE.value,
        seed="345b456",
        max_steps=500,
        render_mode="human",
    )

    try:
        # Reset the environment
        observation, info = env.reset()
        logger.info("Environment reset successfully")
        logger.info(f"Initial observation: {observation}")

        # Run a simple episode
        total_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < 100:
            # Take a random action for demonstration
            action = env.action_space.sample()

            # Execute the action
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Render the environment
            env.render()

            # Log progress every 10 steps
            if step_count % 10 == 0:
                logger.info(
                    f"Step {step_count}: reward={reward:.2f}, "
                    f"total_reward={total_reward:.2f}"
                )

        logger.info(f"Episode finished after {step_count} steps")
        logger.info(f"Total reward: {total_reward:.2f}")

    finally:
        # Clean up
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()
