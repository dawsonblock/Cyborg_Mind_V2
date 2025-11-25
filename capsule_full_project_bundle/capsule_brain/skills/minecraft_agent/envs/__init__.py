"""Environment adapters for Minecraft skill."""

from .minerl_env import MineRLEnvAdapter  # noqa: F401
try:
    from .minedojo_env import MineDojoEnvAdapter  # noqa: F401
except Exception:
    # MineDojo may not be installed
    pass
