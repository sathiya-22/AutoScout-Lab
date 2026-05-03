```python
# robust_equilibrium_approximator/games/__init__.py

"""
The 'games' package provides concrete implementations of various games
that adhere to the core interfaces defined in robust_equilibrium_approximator.core.
It includes abstract base classes for defining new games and specific examples
relevant to the minimum-gain strong equilibrium problem.
"""

# Import and expose the abstract base class for games.
# This class provides a foundational structure that concrete game implementations
# should inherit from, ensuring adherence to the IGame interface.
try:
    from .base import BaseGame
except ImportError as e:
    raise ImportError(f"Could not import BaseGame from .base. Ensure base.py exists and BaseGame is defined. Error: {e}")

# Import and expose a concrete example game implementation.
# This serves as a practical demonstration of how to define a game
# within the framework, particularly one relevant to the minimum-gain
# strong equilibrium concept.
try:
    from .example_min_gain_game import ExampleMinGainGame
except ImportError as e:
    raise ImportError(f"Could not import ExampleMinGainGame from .example_min_gain_game. Ensure example_min_gain_game.py exists and ExampleMinGainGame is defined. Error: {e}")

# Define __all__ for explicit control over what is exported when
# `from robust_equilibrium_approximator.games import *` is used.
__all__ = [
    "BaseGame",
    "ExampleMinGainGame",
]
```