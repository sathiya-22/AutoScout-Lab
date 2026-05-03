```python
import abc
from typing import Any, Dict, List, Optional, FrozenSet, Union


class IAction(abc.ABC):
    """
    Interface for an action that a player can take in a game.
    Actions should be comparable and hashable to allow for efficient storage
    and comparison within game states, strategy profiles, and search trees.
    """

    @abc.abstractmethod
    def get_id(self) -> Any:
        """
        Returns a unique identifier for the action. This ID is used for hashing and equality checks.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """
        Defines equality comparison for actions based on their ID.
        """
        if not isinstance(other, IAction):
            return NotImplemented
        return self.get_id() == other.get_id()

    def __hash__(self) -> int:
        """
        Defines the hash for an action, essential for using actions in sets or as dictionary keys.
        """
        return hash(self.get_id())

    def __repr__(self) -> str:
        """
        Provides a string representation of the action, useful for debugging.
        """
        return f"{self.__class__.__name__}(id={self.get_id()})"


class IPlayer(abc.ABC):
    """
    Interface for a player participating in a game.
    Players should be comparable and hashable, similar to actions.
    """

    @abc.abstractmethod
    def get_id(self) -> Any:
        """
        Returns a unique identifier for the player. This ID is used for hashing and equality checks.
        """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """
        Defines equality comparison for players based on their ID.
        """
        if not isinstance(other, IPlayer):
            return NotImplemented
        return self.get_id() == other.get_id()

    def __hash__(self) -> int:
        """
        Defines the hash for a player, essential for using players in sets or as dictionary keys.
        """
        return hash(self.get_id())

    def __repr__(self) -> str:
        """
        Provides a string representation of the player, useful for debugging.
        """
        return f"{self.__class__.__name__}(id={self.get_id()})"


class IGameState(abc.ABC):
    """
    Interface for representing the current state of a game.
    Game states should ideally be immutable or have clear copy semantics to avoid unexpected side effects
    when traversed by approximation algorithms (e.g., MCTS). They must be hashable for tree structures.
    """

    @abc.abstractmethod
    def get_players(self) -> List[IPlayer]:
        """
        Returns a list of all players relevant to this game state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_description(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing key information describing the current state.
        This can include board positions, scores, resources, etc., depending on the game.
        This is primarily for inspection and serialization, not necessarily for hashing.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash of the game state. This is crucial for using states as keys in
        data structures like dictionaries or for detecting visited states in search algorithms.
        Implementations should ensure that `__eq__` implies identical hashes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Compares two game states for equality. Essential for robust state management.
        """
        raise NotImplementedError


class IUtilityFunction(abc.ABC):
    """
    Interface for a utility function that calculates payoffs for players or coalitions.
    This function quantifies the outcome of a game for specific entities.
    """

    @abc.abstractmethod
    def calculate_utility(self, state: IGameState, target: Union[IPlayer, FrozenSet[IPlayer]]) -> float:
        """
        Calculates the utility (payoff) for a given player or coalition in a specific game state.

        Args:
            state: The game state for which to calculate utility.
            target: The IPlayer or frozenset of IPlayer for whom the utility is calculated.
                    A frozenset represents a coalition.

        Returns:
            The utility value as a float.
        """
        raise NotImplementedError


class IGame(abc.ABC):
    """
    Interface for defining a game, providing core mechanics and accessors to game elements.
    It encapsulates the rules and dynamics of how players interact and how the state evolves.
    """

    @abc.abstractmethod
    def get_num_players(self) -> int:
        """
        Returns the total number of players in the game.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_players(self) -> List[IPlayer]:
        """
        Returns a list of all players in the game.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_initial_state(self) -> IGameState:
        """
        Returns the initial state of the game from which play begins.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_available_actions(self, player: IPlayer, state: IGameState) -> List[IAction]:
        """
        Returns a list of all actions available to a specific player in a given game state.
        This can vary based on game rules and the current state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, state: IGameState, player: IPlayer, action: IAction) -> IGameState:
        """
        Applies an action by a player to the current state and returns the new game state.
        It is critical that this method returns a *new* IGameState instance if IGameState
        implementations are immutable, to preserve the history of states.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal_state(self, state: IGameState) -> bool:
        """
        Checks if the given game state is a terminal state, meaning the game has ended.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_utility_function(self) -> IUtilityFunction:
        """
        Returns the utility function specific to this game, used to evaluate outcomes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_player_turn(self, state: IGameState) -> Optional[IPlayer]:
        """
        For turn-based games, returns the player whose turn it is.
        Returns None if it's a simultaneous-move game where all players act at once,
        or if the game is in a terminal state.
        """
        raise NotImplementedError


class IStrategyProfile(abc.ABC):
    """
    Interface for a strategy profile, which represents a complete set of strategies,
    one for each player, dictating their actions in every possible game state.
    """

    @abc.abstractmethod
    def get_player_strategy(self, player: IPlayer) -> Any:
        """
        Returns the specific strategy object for a given player. The type of this
        object can vary (e.g., a simple IAction for constant strategies, a policy function
        (state -> action), or a state-action mapping for more complex strategies).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_strategies(self) -> Dict[IPlayer, Any]:
        """
        Returns a dictionary mapping all players to their respective strategy objects
        within this profile.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_action_for_state(self, player: IPlayer, state: IGameState) -> IAction:
        """
        Given a specific game state, this method determines and returns the action
        that the specified player would take according to this strategy profile.
        This is crucial for simulating game play under the profile.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Provides a human-readable description of the strategy profile, useful for
        logging and analysis.
        """
        raise NotImplementedError


class ICoalitionRule(abc.ABC):
    """
    Interface for defining rules and algorithms pertaining to coalition formation.
    This includes generating potential coalitions and validating their adherence to specific criteria.
    """

    @abc.abstractmethod
    def generate_potential_coalitions(
        self, players: List[IPlayer], state: Optional[IGameState] = None
    ) -> List[FrozenSet[IPlayer]]:
        """
        Generates a list of potential coalitions based on the given players and an optional
        game state. The specific rules for generation (e.g., all subsets, only coalitions
        meeting certain size or structural criteria) are implemented by concrete classes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid_coalition(self, coalition: FrozenSet[IPlayer], state: Optional[IGameState] = None) -> bool:
        """
        Validates if a given coalition adheres to the rules defined by this interface.
        This can include checking for minimum/maximum size, player types, or other
        game-specific conditions. Can optionally consider the current game state for validation.
        """
        raise NotImplementedError


class IApproximator(abc.ABC):
    """
    Interface for algorithms that approximate robust equilibria in games.
    These approximators are designed to tackle the computational intractability
    of finding exact solutions by leveraging sampling, heuristics, or learning methods.
    """

    @abc.abstractmethod
    def approximate_equilibrium(self, game: IGame, **kwargs) -> IStrategyProfile:
        """
        Approximates a robust equilibrium for the given game.
        This is the primary method for invoking an approximation algorithm.

        Args:
            game: The game for which to find an approximate equilibrium.
            **kwargs: Additional parameters specific to the approximator (e.g.,
                      number of iterations, computational budget, exploration parameters,
                      convergence criteria).

        Returns:
            An IStrategyProfile representing the approximated robust equilibrium.
            This profile should guide players to minimize the worst-case deviation gain
            of any coalition, in accordance with the minimum-gain analogue concept.
        """
        raise NotImplementedError
```