from .interfaces import (
    IGame,
    IPlayer,
    IAction,
    IUtilityFunction,
    IApproximator,
    ICoalitionRule,
)
from .game_state import GameState
from .strategy_profile import StrategyProfile

# Define what gets imported when `from robust_equilibrium_approximator.core import *` is used
__all__ = [
    "IGame",
    "IPlayer",
    "IAction",
    "IUtilityFunction",
    "IApproximator",
    "ICoalitionRule",
    "GameState",
    "StrategyProfile",
]

# No complex error handling or edge cases are typically required in __init__.py
# beyond ensuring submodules are importable. Python's default behavior for
# missing modules or syntax errors in submodules will raise appropriate
# ImportError or SyntaxError, which is generally desired for foundational files.