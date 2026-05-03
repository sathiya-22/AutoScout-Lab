from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from robust_equilibrium_approximator.core.interfaces import (
    IGame,
    IApproximator,
    IStrategyProfile,
    IUtilityFunction
)
from robust_equilibrium_approximator.core.game_state import GameState
from robust_equilibrium_approximator.utils.logging_config import setup_logging

# Initialize logger for the approximators module
logger = setup_logging(__name__)

class BaseApproximator(IApproximator, ABC):
    """
    Abstract base class for all robust equilibrium approximators.

    This class establishes the common interface and foundational structure for
    algorithms designed to find approximate robust equilibria in games where
    direct computation is intractable. It ensures a consistent way to initialize
    and invoke different approximation strategies (e.g., MCTS, RL, hybrid).
    """

    def __init__(self, game: IGame, utility_function: IUtilityFunction, **kwargs):
        """
        Initializes the base approximator with the specific game it will analyze
        and the utility function used to evaluate outcomes.

        Args:
            game (IGame): The game instance for which an equilibrium will be approximated.
                          Must be an object implementing the IGame interface.
            utility_function (IUtilityFunction): The utility function responsible for
                                                  calculating player gains/losses from game states.
                                                  Must be an object implementing the IUtilityFunction interface.
            **kwargs: Arbitrary keyword arguments that can be used to pass
                      solver-specific configurations or general settings
                      to the derived approximator classes.
        
        Raises:
            TypeError: If `game` or `utility_function` do not conform to their
                       respective interface types.
        """
        if not isinstance(game, IGame):
            logger.error(f"Invalid type for 'game' in BaseApproximator.__init__. Expected IGame, got {type(game)}.")
            raise TypeError(f"Provided 'game' must be an instance of IGame, but got {type(game)}.")
        if not isinstance(utility_function, IUtilityFunction):
            logger.error(f"Invalid type for 'utility_function' in BaseApproximator.__init__. Expected IUtilityFunction, got {type(utility_function)}.")
            raise TypeError(f"Provided 'utility_function' must be an instance of IUtilityFunction, but got {type(utility_function)}.")

        self._game: IGame = game
        self._utility_function: IUtilityFunction = utility_function
        self._config: Dict[str, Any] = kwargs
        logger.info(f"Initialized BaseApproximator for game: '{game.get_name()}' with config: {self._config}")

    @abstractmethod
    def approximate_equilibrium(self,
                                initial_state: GameState,
                                max_iterations: int,
                                time_limit_seconds: Optional[int] = None,
                                **kwargs) -> IStrategyProfile:
        """
        Abstract method to compute an approximate robust equilibrium strategy profile.

        Derived classes must implement this method with their specific
        approximation algorithm logic (e.g., MCTS, RL, hybrid approaches).
        This method should leverage sampling and learned heuristics to avoid
        combinatorial explosion for large games.

        Args:
            initial_state (GameState): The starting state of the game from which
                                       the approximation process begins.
            max_iterations (int): The maximum number of iterations or computational
                                  steps the approximation algorithm is allowed to run.
            time_limit_seconds (Optional[int]): An optional upper bound on the
                                                execution time in seconds. If provided
                                                and exceeded, the algorithm should
                                                gracefully terminate and return the
                                                best approximation found so far. Defaults to None.
            **kwargs: Additional parameters specific to the particular approximation
                      algorithm (e.g., exploration constants for MCTS, neural network
                      hyperparameters for RL, batch sizes, etc.).

        Returns:
            IStrategyProfile: An approximate robust equilibrium strategy profile
                              that attempts to minimize the worst-case gain for
                              any deviating coalition.

        Raises:
            NotImplementedError: If the subclass fails to implement this abstract method.
            ValueError: For invalid input parameters (e.g., non-positive max_iterations).
        """
        if not isinstance(initial_state, GameState):
            logger.error(f"Invalid type for 'initial_state'. Expected GameState, got {type(initial_state)}.")
            raise TypeError(f"Provided 'initial_state' must be an instance of GameState, but got {type(initial_state)}.")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            logger.error(f"Invalid value for 'max_iterations'. Expected a positive integer, got {max_iterations}.")
            raise ValueError("max_iterations must be a positive integer.")
        if time_limit_seconds is not None and (not isinstance(time_limit_seconds, int) or time_limit_seconds <= 0):
            logger.error(f"Invalid value for 'time_limit_seconds'. Expected a positive integer or None, got {time_limit_seconds}.")
            raise ValueError("time_limit_seconds must be a positive integer or None.")

        logger.info(f"Approximation started for game '{self._game.get_name()}' "
                    f"with {max_iterations} iterations and time limit: {time_limit_seconds}s.")
        raise NotImplementedError("Subclasses must implement the 'approximate_equilibrium' method.")

    def get_game(self) -> IGame:
        """
        Returns the game instance associated with this approximator.
        """
        return self._game

    def get_utility_function(self) -> IUtilityFunction:
        """
        Returns the utility function used by this approximator.
        """
        return self._utility_function

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration dictionary that was passed during initialization.
        """
        return self._config

    # Future common methods could include:
    # - `reset(self)`: To clear internal state for a new approximation run.
    # - `save_progress(self, path: str)`: To serialize the current state of the approximator.
    # - `load_progress(self, path: str)`: To load a previously saved state.