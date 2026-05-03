"""
A 'Robust Equilibrium Approximator' library for computationally intractable game theory concepts.

This library provides approximate solutions for problems like the minimum-gain analogue
of the strong equilibrium, which is intractable for practical game sizes.
It implements a hybrid approach combining Monte Carlo Tree Search (MCTS)
or Reinforcement Learning (RL) agents to explore the vast space of coalition deviations,
identifying high-impact coalitions and learning to find strategies that minimize their
worst-case gains without exhaustive enumeration.

It offers pluggable interfaces for defining game utility functions and coalition
formation rules, and provides approximate solver algorithms that scale to larger
numbers of players by leveraging sampling and learned heuristics.
"""

__version__ = "0.1.0"

# Import core interfaces for convenience
from .core.interfaces import (
    IGame,
    IPlayer,
    IAction,
    IUtilityFunction,
    IApproximator,
    ICoalitionRule,
)

# Import key approximators
from .approximators.mcts_solver import MCTSSolver
from .approximators.rl_agent_solver import RLAgentSolver
from .approximators.hybrid_solver import HybridSolver

# Import key coalition components
from .coalitions.manager import CoalitionManager
from .coalitions.rules import CoalitionRules

# Import game state and strategy profile
from .core.game_state import GameState
from .core.strategy_profile import StrategyProfile

# Optional: Import an example game for quick start
try:
    from .games.example_min_gain_game import ExampleMinGainGame
except ImportError:
    # This allows the __init__.py to be imported even if games.example_min_gain_game
    # has unresolved dependencies or is not fully implemented yet.
    # In a real scenario, this might indicate a problem or a feature not yet ready.
    pass


# Define __all__ for explicit imports
__all__ = [
    "IGame",
    "IPlayer",
    "IAction",
    "IUtilityFunction",
    "IApproximator",
    "ICoalitionRule",
    "MCTSSolver",
    "RLAgentSolver",
    "HybridSolver",
    "CoalitionManager",
    "CoalitionRules",
    "GameState",
    "StrategyProfile",
    "ExampleMinGainGame", # Only if imported successfully
]

# Dynamically add ExampleMinGainGame to __all__ if it was successfully imported
if 'ExampleMinGainGame' in locals():
    __all__.append("ExampleMinGainGame")

# Basic logging configuration for the library
from .utils import logging_config
logging_config.setup_logging()