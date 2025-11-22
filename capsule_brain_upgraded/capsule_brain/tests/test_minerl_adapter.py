"""Test the MineRLEnvAdapter if MineRL is installed.

This test will be skipped at runtime if the ``minerl`` package is not
available.  It ensures that the adapter can reset the environment and
perform a single random step without raising an exception.  The
discrete action space size is determined by the adapter's action map.
"""

import pytest

try:
    import minerl  # type: ignore
    MINERL_AVAILABLE = True
except ImportError:
    MINERL_AVAILABLE = False

if MINERL_AVAILABLE:
    from capsule_brain.skills.minecraft_agent.envs.minerl_env import MineRLEnvAdapter


@pytest.mark.skipif(not MINERL_AVAILABLE, reason="minerl package not available")
def test_minerl_env_adapter() -> None:
    env = MineRLEnvAdapter(env_name="MineRLTreechop-v0")
    obs = env.reset()
    assert "pixels" in obs
    # Step with a random action index within the adapter's action map
    import random

    action = random.randint(0, len(env.action_map) - 1)
    obs2, reward, done, info = env.step(action)
    assert "pixels" in obs2