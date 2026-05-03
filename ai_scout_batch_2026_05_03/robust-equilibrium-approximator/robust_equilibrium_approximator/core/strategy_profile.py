```python
from typing import Dict, Any, Hashable, frozenset

from robust_equilibrium_approximator.core.interfaces import (
    IPlayer,
    IAction,
    IGame,
    IUtilityFunction,
)


class StrategyProfile:
    """
    Represents a complete strategy profile for a game, mapping each player to their chosen action.

    A strategy profile defines the actions taken by all players simultaneously or sequentially
    at a given stage of a game. It is a fundamental concept for evaluating outcomes and
    identifying deviations. This class manages and facilitates the evaluation of these strategies.
    """

    def __init__(self, strategies: Dict[IPlayer, IAction]):
        """
        Initializes a StrategyProfile.

        Args:
            strategies (Dict[IPlayer, IAction]): A dictionary mapping each player
                                                 (implementing IPlayer and being hashable)
                                                 to their chosen action (implementing IAction).

        Raises:
            ValueError: If `strategies` is empty or contains keys/values that do not
                        implement IPlayer/IAction respectively.
            TypeError: If IPlayer instances are not hashable (required for internal dictionary
                       and for hashing the StrategyProfile itself).
        """
        if not strategies:
            raise ValueError("Strategy profile cannot be empty.")

        # Basic type checking and hashability check
        for player, action in strategies.items():
            if not isinstance(player, IPlayer):
                raise ValueError(
                    f"Key must be an IPlayer instance, got {type(player).__name__} for {player}"
                )
            if not isinstance(action, IAction):
                raise ValueError(
                    f"Value must be an IAction instance, got {type(action).__name__} for {action}"
                )
            # Check if player is hashable (required for dict keys and frozenset below)
            try:
                hash(player)
            except TypeError as e:
                raise TypeError(
                    f"IPlayer instance '{player}' must be hashable. Original error: {e}"
                ) from e
            # Check if action is hashable (required for frozenset in __hash__)
            try:
                hash(action)
            except TypeError as e:
                raise TypeError(
                    f"IAction instance '{action}' must be hashable. Original error: {e}"
                ) from e

        # Store the strategies. Assuming IPlayer and IAction are treated as immutable
        # data structures/identifiers once passed to the profile.
        self._strategies: Dict[IPlayer, IAction] = strategies

    def get_strategy_for_player(self, player: IPlayer) -> IAction:
        """
        Retrieves the action chosen by a specific player in this profile.

        Args:
            player (IPlayer): The player for whom to retrieve the strategy.

        Returns:
            IAction: The action chosen by the player.

        Raises:
            KeyError: If the player is not part of this strategy profile.
        """
        try:
            return self._strategies[player]
        except KeyError:
            player_id_str = getattr(player, "player_id", str(player))
            raise KeyError(
                f"Player with ID '{player_id_str}' not found in this strategy profile."
            )

    def get_all_strategies(self) -> Dict[IPlayer, IAction]:
        """
        Returns a shallow copy of the complete strategy profile.

        Returns:
            Dict[IPlayer, IAction]: A new dictionary mapping all players to their chosen actions.
        """
        return self._strategies.copy()

    def get_players(self) -> frozenset[IPlayer]:
        """
        Returns a frozenset of all players involved in this strategy profile.

        Returns:
            frozenset[IPlayer]: A frozenset containing all players.
        """
        return frozenset(self._strategies.keys())

    def evaluate_profile_utility(
        self, game: IGame, utility_function: IUtilityFunction
    ) -> Dict[IPlayer, float]:
        """
        Evaluates the utility for each player given this strategy profile and a utility function.

        This method acts as a convenience wrapper, delegating the actual utility calculation
        to the provided IUtilityFunction, passing itself and the game context.
        For games with dynamic states or multiple stages, the `IUtilityFunction`'s
        implementation might need to account for these complexities.

        Args:
            game (IGame): The game instance providing the context for evaluation.
            utility_function (IUtilityFunction): The utility function to use for calculating
                                                  player payoffs.

        Returns:
            Dict[IPlayer, float]: A dictionary mapping each player to their calculated utility.
        """
        return utility_function.calculate_utilities(game, self)

    def __len__(self) -> int:
        """
        Returns the number of players whose strategies are defined in this profile.
        """
        return len(self._strategies)

    def __eq__(self, other: Any) -> bool:
        """
        Checks if two StrategyProfile instances are equal.
        Equality is defined by having the same players and the same actions for each player.
        """
        if not isinstance(other, StrategyProfile):
            return NotImplemented
        return self._strategies == other._strategies

    def __hash__(self) -> int:
        """
        Computes a hash for the StrategyProfile.
        This allows StrategyProfile instances to be used as keys in dictionaries or elements in sets.
        It requires that all IPlayer and IAction instances within the profile are hashable.
        """
        # A frozenset of (player, action) tuples ensures order independence and hashability.
        return hash(frozenset(self._strategies.items()))

    def __repr__(self) -> str:
        """
        Returns a string representation of the StrategyProfile, showing players and their actions.
        """
        strategy_parts = []
        for player, action in self._strategies.items():
            player_id_str = getattr(player, "player_id", str(player))
            action_id_str = getattr(action, "action_id", str(action))
            strategy_parts.append(f"{player_id_str}: {action_id_str}")
        return f"StrategyProfile({{{', '.join(strategy_parts)}}})"
```