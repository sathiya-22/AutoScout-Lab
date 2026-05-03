```python
from typing import Dict, Any, List, Optional, TypeVar, Generic, Hashable
from robust_equilibrium_approximator.core.interfaces import IPlayer, IGame


# Define a generic type for game-specific data, constrained to be Hashable.
# This constraint is important for __hash__ and __eq__ methods of GameState,
# allowing GameState instances to be used as keys in dictionaries or elements in sets.
GameStateData = TypeVar('GameStateData', bound=Hashable)


class GameState(Generic[GameStateData]):
    """
    Represents the current state of a game at a particular point in time.

    This class is generic over `GameStateData` to allow game-specific data
    structures to be included directly in the state, while providing
    common attributes required by the approximators and coalition manager.
    It supports immutability for easier use in search algorithms.
    """

    def __init__(
        self,
        game: IGame,
        players: List[IPlayer],
        current_round: int = 0,
        game_specific_data: Optional[GameStateData] = None,
        actions_history: Optional[List[Dict[str, Any]]] = None,
        is_terminal: bool = False,
        terminal_reason: Optional[str] = None
    ):
        """
        Initializes a new GameState.

        Args:
            game: The IGame instance this state belongs to.
            players: A list of IPlayer objects currently in the game.
            current_round: The current round or turn number (0-indexed usually).
            game_specific_data: An optional generic object or dictionary holding
                                data specific to the particular game implementation.
                                This data must be hashable.
            actions_history: A list of dictionaries, where each dictionary represents
                             actions taken in a previous round/step.
                             Example: [{'player_id': 'p1', 'action': action_obj, 'details': {...}}, ...]
            is_terminal: True if this state is a terminal state of the game.
            terminal_reason: An optional string explaining why the state is terminal.

        Raises:
            TypeError: If 'game' is not an IGame or 'players' is not a list of IPlayer.
            ValueError: If 'current_round' is negative.
        """
        if not isinstance(game, IGame):
            raise TypeError("The 'game' argument must be an instance of IGame.")
        if not isinstance(players, list) or not all(isinstance(p, IPlayer) for p in players):
            raise TypeError("The 'players' argument must be a list of IPlayer instances.")
        if not isinstance(current_round, int) or current_round < 0:
            raise ValueError("Current round must be a non-negative integer.")
        if game_specific_data is not None and not isinstance(game_specific_data, Hashable):
            # This check is technically redundant due to TypeVar(..., bound=Hashable)
            # but serves as an explicit runtime check if type hints are ignored.
            pass

        self._game: IGame = game
        self._players: List[IPlayer] = players
        self._current_round: int = current_round
        self._game_specific_data: Optional[GameStateData] = game_specific_data
        self._actions_history: List[Dict[str, Any]] = actions_history if actions_history is not None else []
        self._is_terminal: bool = is_terminal
        self._terminal_reason: Optional[str] = terminal_reason

    @property
    def game(self) -> IGame:
        """Returns the IGame instance this state belongs to."""
        return self._game

    @property
    def players(self) -> List[IPlayer]:
        """Returns the list of IPlayer objects in the game."""
        return self._players

    @property
    def current_round(self) -> int:
        """Returns the current round or turn number."""
        return self._current_round

    @property
    def game_specific_data(self) -> Optional[GameStateData]:
        """Returns the game-specific data for this state."""
        return self._game_specific_data

    @property
    def actions_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of actions taken in the game.
        Returns a copy to prevent external modification of the internal state.
        """
        return list(self._actions_history)

    @property
    def is_terminal(self) -> bool:
        """Returns True if this state is a terminal state."""
        return self._is_terminal

    @property
    def terminal_reason(self) -> Optional[str]:
        """Returns the reason for termination if it's a terminal state."""
        return self._terminal_reason

    def update_state(
        self,
        new_game_specific_data: GameStateData,
        actions_taken_this_round: Optional[List[Dict[str, Any]]] = None,
        increment_round: bool = True,
        is_terminal: Optional[bool] = None,
        terminal_reason: Optional[str] = None
    ) -> 'GameState[GameStateData]':
        """
        Creates and returns a new GameState instance reflecting updates.
        This ensures immutability of game states, which is often beneficial
        in search algorithms like MCTS, as it avoids side effects and simplifies
        state management.

        Args:
            new_game_specific_data: The updated game-specific data.
            actions_taken_this_round: A list of actions taken in the current round
                                      to be added to the history. Each item in the list
                                      should be a dictionary, e.g.,
                                      `{'player_id': 'p1', 'action': action_obj, 'details': {...}}`.
            increment_round: If True, increments the current_round by 1.
            is_terminal: Optional boolean to explicitly set terminal status.
                         If None, inherits from current state unless updated.
            terminal_reason: Optional string for terminal reason.

        Returns:
            A new GameState instance with the updated information.

        Raises:
            TypeError: If 'actions_taken_this_round' is not a list of dictionaries.
        """
        updated_actions_history = list(self._actions_history)
        if actions_taken_this_round:
            if not isinstance(actions_taken_this_round, list) or not all(isinstance(a, dict) for a in actions_taken_this_round):
                raise TypeError("actions_taken_this_round must be a list of dictionaries.")
            updated_actions_history.extend(actions_taken_this_round)

        new_round = self._current_round + 1 if increment_round else self._current_round

        new_is_terminal = is_terminal if is_terminal is not None else self._is_terminal
        new_terminal_reason = terminal_reason if terminal_reason is not None else self._terminal_reason

        return GameState(
            game=self._game,
            players=self._players,  # Players list is typically static for a game instance
            current_round=new_round,
            game_specific_data=new_game_specific_data,
            actions_history=updated_actions_history,
            is_terminal=new_is_terminal,
            terminal_reason=new_terminal_reason
        )

    def get_player_by_id(self, player_id: str) -> Optional[IPlayer]:
        """
        Retrieves a player by their ID from the current game state's player list.

        Args:
            player_id: The unique identifier of the player.

        Returns:
            The IPlayer object if found, otherwise None.
        """
        for player in self._players:
            if player.player_id == player_id:
                return player
        return None

    def __eq__(self, other: object) -> bool:
        """
        Compares two GameState objects for equality.
        Two states are considered equal if they belong to the same game,
        have the same players (order-sensitive), are at the same round,
        and have equivalent game-specific data and terminal status.
        Actions history is typically not part of state identity for equality
        as it describes *how* a state was reached, not its intrinsic properties.
        """
        if not isinstance(other, GameState):
            return NotImplemented
        
        # All components used for equality must themselves implement __eq__
        # The TypeVar bound `Hashable` typically implies they implement `__eq__`.
        return (
            self._game == other._game and
            tuple(self._players) == tuple(other._players) and  # Convert list to tuple for consistent comparison
            self._current_round == other._current_round and
            self._game_specific_data == other._game_specific_data and
            self._is_terminal == other._is_terminal
        )

    def __hash__(self) -> int:
        """
        Returns a hash value for the GameState object.
        This is crucial for using GameState instances as keys in dictionaries
        or elements in sets, especially in search algorithms like MCTS.
        Requires `game`, `players` (converted to a hashable tuple),
        `current_round`, `game_specific_data`, and `is_terminal` to be hashable.
        """
        return hash((
            self._game,
            tuple(self._players),  # Convert list to tuple for hashing
            self._current_round,
            self._game_specific_data,
            self._is_terminal
        ))

    def __repr__(self) -> str:
        """
        Provides a string representation of the GameState for debugging purposes.
        """
        game_id = getattr(self._game, 'game_id', 'Unknown')
        game_data_type = type(self._game_specific_data).__name__ if self._game_specific_data else 'None'
        return (
            f"GameState(game_id='{game_id}', "
            f"round={self._current_round}, "
            f"num_players={len(self._players)}, "
            f"terminal={self._is_terminal}, "
            f"game_data_type={game_data_type})"
        )
```