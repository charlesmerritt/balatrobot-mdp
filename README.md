<div align="center">
  <h1>BalatroBot</h1>
  <p align="center">
    <a href="https://github.com/coder/balatrobot/releases">
      <img alt="GitHub release" src="https://img.shields.io/github/v/release/coder/balatrobot?include_prereleases&sort=semver&style=for-the-badge&logo=github"/>
    </a>
    <a href="https://pypi.org/project/balatrobot/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/balatrobot?style=for-the-badge&logo=pypi&logoColor=white"/>
    </a>
    <a href="https://discord.gg/TPn6FYgGPv">
      <img alt="Discord" src="https://img.shields.io/badge/discord-server?style=for-the-badge&logo=discord&logoColor=%23FFFFFF&color=%235865F2"/>
    </a>
  </p>
  <div><img src="https://github.com/user-attachments/assets/514f85ab-485d-48f5-80fc-721eafad5192" alt="balatrobot" width="256" height="256"></div>
  <p><em>A framework for Balatro bot development</em></p>
</div>

---

BalatroBot is a Python framework designed to help developers create automated bots for the card game Balatro. The framework provides a comprehensive API for interacting with the game, handling game state, making strategic decisions, and executing actions. Whether you're building a simple bot or a sophisticated AI player, BalatroBot offers the tools and structure needed to get started quickly.

## ✨ Features

- **🎮 Gymnasium Integration**: Standard reinforcement learning environment compatible with Stable-Baselines3, Ray RLlib, and other RL libraries
- **🔌 Client-Server API**: Easy-to-use Python client for controlling Balatro through TCP connection
- **📊 Rich Game State**: Complete access to game state including cards, jokers, chips, money, and more
- **🧪 Testing Tools**: Checkpoint system for saving/loading game states and comprehensive test suite
- **📝 Type Safety**: Full Pydantic models for all game state and API interactions

## 🚀 Quick Start

### Using the Gymnasium Environment

```python
from balatrobot import BalatroEnv

# Create a Gymnasium-compatible environment
env = BalatroEnv(max_steps=500, render_mode="human")

# Standard Gymnasium interface
observation, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### Using the Direct Client API

```python
from balatrobot import BalatroClient

with BalatroClient() as client:
    client.send_message("start_run", {"deck": "Red Deck", "stake": 1})
    client.send_message("skip_or_select_blind", {"action": "select"})
    # ... more game actions
```

## 📚 Documentation

https://coder.github.io/balatrobot/

- [Gymnasium MDP Integration Guide](docs/gymnasium.md)
- [Client API Reference](https://coder.github.io/balatrobot/)

## 🙏 Acknowledgments

This project is a fork of the original [balatrobot](https://github.com/besteon/balatrobot) repository. We would like to acknowledge and thank the original contributors who laid the foundation for this framework:

- [@phughesion](https://github.com/phughesion)
- [@besteon](https://github.com/besteon)
- [@giewev](https://github.com/giewev)

The original repository provided the initial API and botting framework that this project has evolved from. We appreciate their work in creating the foundation for Balatro bot development.
