"""2D navigation RL environment (unicycle robot, static circular obstacles)."""

from gymnasium.envs.registration import register

from .config import DEFAULT_CONFIG, EnvConfig
from .env import NavEnv2D

__all__ = ["NavEnv2D", "EnvConfig", "DEFAULT_CONFIG"]

# No `max_episode_steps` here on purpose: the environment reports truncation
# itself, so wrapping it in gymnasium's TimeLimit would be redundant.
register(id="NavRL2D-v0", entry_point="navrl2d.env:NavEnv2D")
