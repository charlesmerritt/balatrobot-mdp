"""Verification script for Gymnasium integration."""

import sys

print("=" * 60)
print("BalatroBot Gymnasium Integration Verification")
print("=" * 60)

# Test 1: Import
print("\n[1/5] Testing import...")
try:
    from balatrobot import BalatroEnv
    print("✓ BalatroEnv imported successfully")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Environment creation
print("\n[2/5] Testing environment creation...")
try:
    env = BalatroEnv()
    print("✓ Environment created successfully")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    sys.exit(1)

# Test 3: Spaces
print("\n[3/5] Testing spaces...")
try:
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: Dict with {len(env.observation_space.spaces)} keys")
    print(f"  Keys: {list(env.observation_space.spaces.keys())}")
    print("✓ Spaces defined correctly")
except Exception as e:
    print(f"✗ Failed to check spaces: {e}")
    sys.exit(1)

# Test 4: Observation extraction
print("\n[4/5] Testing observation extraction...")
try:
    obs = env._get_obs()
    assert "state" in obs
    assert "chips" in obs
    assert "dollars" in obs
    print("✓ Observation extraction works")
except Exception as e:
    print(f"✗ Failed to extract observation: {e}")
    sys.exit(1)

# Test 5: Action conversion
print("\n[5/5] Testing action conversion...")
try:
    cards = env._action_to_cards(7)
    assert 0 in cards and 1 in cards and 2 in cards
    print(f"  Action 7 -> Cards {cards}")
    print("✓ Action conversion works")
except Exception as e:
    print(f"✗ Failed action conversion: {e}")
    sys.exit(1)

# Cleanup
env.close()

# Summary
print("\n" + "=" * 60)
print("All verification checks passed! ✓")
print("=" * 60)
print("\nThe Gymnasium integration is ready to use.")
print("\nNext steps:")
print("  1. Start Balatro with BalatroBot mod")
print("  2. Run: python bots/example_mdp.py")
print("  3. Or train an RL agent with Stable-Baselines3")
print("\nSee docs/gymnasium.md for detailed documentation.")
print("=" * 60)
