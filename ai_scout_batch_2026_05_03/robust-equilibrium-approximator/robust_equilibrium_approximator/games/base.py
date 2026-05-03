from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Assuming these interfaces are defined in the core.interfaces module
from robust_equilibrium_approximator.core.interfaces import (
    IPlayer,
    IAction,
    IGameState,
    IUtilityFunction,
    IGame,
    IStrategyProfile,
)


class AbstractPlayer(IPlayer, ABC):
    """
    Abstract base class for a game player.
    Provides a unique identifier and a name for the player, implementing IPlayer.
    """

    def __init__(self, player_id: str, name: Optional[str] = None):
        if not player_id:
            raise ValueError("Player ID cannot be empty or None.")
        self._player_id = player_id
        self._name = name if name is not None else f"Player_{player_id}"

    @property
    def player_id(self) -> str:
        """Returns the unique identifier for the player."""
        return self._player_id

    @property
    def name(self) -> str:
        """Returns the descriptive name of the player."""
        return self._name

    def __hash__(self) -> int:
        """Returns a hash for the player based on their ID."""
        return hash(self._player_id)

    def __eq__(self, other: Any) -> bool:
        """Compares two players for equality based on their IDs."""
        if not isinstance(other, IPlayer):
            return NotImplemented
        return self._player_id == other.player_id

    def __repr__(self) -> str:
        """Returns a string representation of the player."""
        return f"AbstractPlayer(id='{self.player_id}', name='{self.name}')"


class AbstractAction(IAction, ABC):
    """
    Abstract base class for a player's action, implementing IAction.
    Actions are typically immutable and identifiable.
    """

    def __init__(self, action_id: str, description: Optional[str] = None):
        if not action_id:
            raise ValueError("Action ID cannot be empty or None.")
        self._action_id = action_id
        self._description = description if description is not None else f"Action_{action_id}"

    @property
    def action_id(self) -> str:
        """Returns the unique identifier for the action."""
        return self._action_id

    @property
    def description(self) -> str:
        """Returns a descriptive string for the action."""
        return self._description

    def __hash__(self) -> int:
        """Returns a hash for the action based on its ID."""
        return hash(self._action_id)

    def __eq__(self, other: Any) -> bool:
        """Compares two actions for equality based on their IDs."""
        if not isinstance(other, IAction):
            return NotImplemented
        return self._action_id == other.action_id

    def __repr__(self) -> str:
        """Returns a string representation of the action."""
        return f"AbstractAction(id='{self.action_id}', desc='{self.description}')"


class AbstractGameState(IGameState, ABC):
    """
    Abstract base class for representing the state of a game, implementing IGameState.
    Game states should be immutable or have clear methods for state transitions
    to facilitate approximation algorithms (e.g., MCTS often relies on state immutability
    for node representation).
    """

    @abstractmethod
    def get_current_player_id(self) -> Optional[str]:
        """
        Returns the ID of the player whose turn it is in this state,
        or None if simultaneous moves, all players have acted, or the game is terminal.
        """
        pass

    @abstractmethod
    def to_hashable(self) -> Any:
        """
        Returns a hashable representation of the game state.
        This is crucial for using game states as keys in dictionaries or elements in sets,
        especially in search algorithms like MCTS.
        """
        pass

    def __hash__(self) -> int:
        """Provides a hash for the game state, delegating to to_hashable()."""
        return hash(self.to_hashable())

    def __eq__(self, other: Any) -> bool:
        """Compares two game states for equality, delegating to to_hashable()."""
        if not isinstance(other, IGameState):
            return NotImplemented
        return self.to_hashable() == other.to_hashable()

    @abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the game state."""
        pass


class AbstractUtilityFunction(IUtilityFunction, ABC):
    """
    Abstract base class for a utility function, implementing IUtilityFunction.
    It calculates player utilities for a given game state.
    """

    @abstractmethod
    def calculate_utility(self, state: IGameState, player_id: str) -> float:
        """
        Calculates the utility (payoff) for a specific player in a given game state.
        :param state: The current game state.
        :param player_id: The ID of the player for whom to calculate utility.
        :return: The utility value (float) for the specified player.
        """
        pass


class AbstractGame(IGame, ABC):
    """
    Abstract base class for defining a game, implementing IGame,
    within the Robust Equilibrium Approximator framework.
    It provides the foundational structure for setting up players, actions,
    and game dynamics, requiring concrete implementations for game-specific logic.
    """

    def __init__(self, players: List[IPlayer], utility_function: IUtilityFunction):
        if not players:
            raise ValueError("A game must have at least one player.")
        if not all(isinstance(p, IPlayer) for p in players):
            raise TypeError("All elements in 'players' must be instances of IPlayer.")
        if not isinstance(utility_function, IUtilityFunction):
            raise TypeError("'utility_function' must be an instance of IUtilityFunction.")

        # Store players in a dict for efficient lookup by ID
        self._players: Dict[str, IPlayer] = {p.player_id: p for p in players}
        self._utility_function = utility_function

    @property
    def num_players(self) -> int:
        """Returns the total number of players in the game."""
        return len(self._players)

    def get_players(self) -> List[IPlayer]:
        """Returns a list of all players participating in the game."""
        return list(self._players.values())

    def get_player_by_id(self, player_id: str) -> Optional[IPlayer]:
        """
        Retrieves a player by their ID.
        :param player_id: The ID of the player to retrieve.
        :return: The IPlayer instance if found, otherwise None.
        """
        return self._players.get(player_id)

    def get_utility_function(self) -> IUtilityFunction:
        """Returns the utility function used for this game."""
        return self._utility_function

    @abstractmethod
    def get_initial_state(self) -> IGameState:
        """
        Returns the initial state of the game from which play begins.
        """
        pass

    @abstractmethod
    def get_possible_actions(self, player_id: str, state: IGameState) -> List[IAction]:
        """
        Returns a list of all possible actions a specific player can take from a given state.
        :param player_id: The ID of the player whose actions are being queried.
        :param state: The current game state.
        :return: A list of IAction instances.
        """
        pass

    @abstractmethod
    def get_state_from_strategy_profile(self, strategy_profile: IStrategyProfile) -> IGameState:
        """
        Given a complete strategy profile (an action for each player, or a set of simultaneous actions),
        determines and returns the resulting game state.
        This is particularly relevant for strategic-form games or when evaluating
        simultaneous actions. For extensive-form games, this might represent
        a single turn's transition or require further game play simulation.
        :param strategy_profile: A mapping of player IDs to their chosen actions.
        :return: The new IGameState after all actions in the profile are applied.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: IGameState) -> bool:
        """
        Checks if the given game state is a terminal state (i.e., the game has ended
        and no further actions can be taken).
        :param state: The game state to check.
        :return: True if the state is terminal, False otherwise.
        """
        pass

    @abstractmethod
    def get_outcome_utilities(self, state: IGameState) -> Dict[str, float]:
        """
        Calculates and returns the utilities (payoffs) for all players in a given
        (usually terminal) game state. This method often delegates to the
        `AbstractUtilityFunction` but can also provide cached or pre-calculated
        outcomes if the state implies them.
        :param state: The game state for which to calculate utilities.
        :return: A dictionary mapping player IDs to their respective utility values.
        """
        pass<ctrl63>