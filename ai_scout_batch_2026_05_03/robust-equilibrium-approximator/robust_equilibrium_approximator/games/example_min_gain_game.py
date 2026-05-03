from typing import Dict, List, Set, Tuple, Any
from robust_equilibrium_approximator.core.interfaces import IGame, IPlayer, IAction, IUtilityFunction
from robust_equilibrium_approximator.core.game_state import GameState
from robust_equilibrium_approximator.core.strategy_profile import StrategyProfile


class ExamplePlayer(IPlayer):
    """
    A concrete implementation of IPlayer for the example game.
    Players are identified by a unique string ID.
    """
    def __init__(self, player_id: str):
        if not isinstance(player_id, str) or not player_id:
            raise ValueError("Player ID must be a non-empty string.")
        self._id = player_id

    @property
    def id(self) -> str:
        return self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ExamplePlayer):
            return NotImplemented
        return self._id == other.id

    def __repr__(self) -> str:
        return f"Player(id='{self._id}')"


class ContributionAction(IAction):
    """
    A concrete implementation of IAction representing a player's contribution.
    The action value is a non-negative integer representing the contribution amount.
    """
    def __init__(self, contribution_value: int):
        if not isinstance(contribution_value, int) or contribution_value < 0:
            raise ValueError("Contribution value must be a non-negative integer.")
        self._value = contribution_value

    @property
    def value(self) -> Any:
        return self._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContributionAction):
            return NotImplemented
        return self._value == other.value

    def __repr__(self) -> str:
        return f"ContributionAction(value={self._value})"


class MinGainGameUtility(IUtilityFunction):
    """
    A utility function for a simple cooperative contribution game.
    Utility for a player is calculated as:
    (base_reward + shared_benefit_from_total_contribution) - (cost_of_individual_contribution)

    - `base_reward`: A baseline utility every player gets.
    - `contribution_cost_factor`: Multiplier for the cost of a player's own contribution.
    - `share_factor`: Multiplier for how much the total contribution translates into shared benefit.
    """
    def __init__(self, base_reward: float = 5.0, contribution_cost_factor: float = 0.5, share_factor: float = 1.0):
        if not all(isinstance(f, (int, float)) and f >= 0 for f in [base_reward, contribution_cost_factor, share_factor]):
            raise ValueError("Utility factors must be non-negative numbers.")
        self.base_reward = float(base_reward)
        self.contribution_cost_factor = float(contribution_cost_factor)
        self.share_factor = float(share_factor)

    def calculate_utility(self, strategy_profile: StrategyProfile, player: IPlayer) -> float:
        """
        Calculates the utility for a given player based on the overall strategy profile.
        """
        player_contributions: Dict[str, int] = {}
        all_players_in_profile = set()

        for p, action in strategy_profile.get_all_actions():
            all_players_in_profile.add(p)
            if isinstance(action, ContributionAction):
                player_contributions[p.id] = action.value
            else:
                # If an action is not a ContributionAction, assume 0 contribution from that player
                player_contributions[p.id] = 0

        total_contribution = sum(player_contributions.values())
        
        # Individual contribution of the target player
        player_contribution = player_contributions.get(player.id, 0)

        num_players = len(all_players_in_profile)
        if num_players == 0:
            return 0.0 # No players, no utility

        # Shared benefit from the common pool, distributed equally
        # This could be designed in many ways; here, it's a portion of total_contribution per player.
        shared_benefit = (total_contribution * self.share_factor) / num_players

        # Cost of player's own contribution
        individual_cost = player_contribution * self.contribution_cost_factor

        utility = self.base_reward + shared_benefit - individual_cost
        return utility


class ExampleMinGainGame(IGame):
    """
    A concrete game implementation designed to exemplify the minimum-gain strong equilibrium problem.
    This is a single-shot cooperative contribution game where players choose how much to contribute.
    Their utility depends on a shared pool (sum of all contributions) and their individual cost of contribution.
    """
    def __init__(self, num_players: int, min_contribution: int = 0, max_contribution: int = 10):
        if not isinstance(num_players, int) or num_players <= 0:
            raise ValueError("Number of players must be a positive integer.")
        if not all(isinstance(c, int) for c in [min_contribution, max_contribution]):
            raise ValueError("Contribution range values must be integers.")
        if not (0 <= min_contribution <= max_contribution):
            raise ValueError("Invalid contribution range: min_contribution must be <= max_contribution and non-negative.")

        self._players: List[IPlayer] = [ExamplePlayer(f"P{i+1}") for i in range(num_players)]
        
        # All players have the same discrete set of available contribution actions
        self._available_actions: Set[IAction] = {
            ContributionAction(c) for c in range(min_contribution, max_contribution + 1)
        }
        
        self._utility_function: IUtilityFunction = MinGainGameUtility()
        # For a single-shot game, the initial state is simple.
        self._initial_state: GameState = GameState(current_round=0, is_terminal=False)

    @property
    def players(self) -> List[IPlayer]:
        return list(self._players) # Return a copy to prevent external modification

    def get_actions(self, player: IPlayer) -> Set[IAction]:
        if player not in self._players:
            raise ValueError(f"Player {player.id} is not part of this game.")
        return set(self._available_actions) # Return a copy

    def get_utility_function(self) -> IUtilityFunction:
        return self._utility_function

    def get_initial_state(self) -> GameState:
        return self._initial_state

    def is_terminal(self, state: GameState) -> bool:
        """
        For this single-shot game, the game is terminal after one round of actions (i.e., after the initial state).
        """
        return state.is_terminal

    def get_next_state(self, current_state: GameState, strategy_profile: StrategyProfile) -> GameState:
        """
        In this single-shot game, choosing a strategy profile directly leads to a terminal state.
        """
        if current_state.is_terminal:
            return current_state # Already terminal, no further state change
        
        # In a single-shot game, any action leads to a terminal state immediately.
        return GameState(current_round=current_state.current_round + 1, is_terminal=True)

    def get_payoffs(self, strategy_profile: StrategyProfile) -> Dict[IPlayer, float]:
        """
        Calculates the payoffs for all players given a complete strategy profile.
        """
        payoffs: Dict[IPlayer, float] = {}
        for player in self._players:
            # Check if the player has an action assigned in the strategy profile.
            # If not, it means the profile is incomplete for this player, which might be an error state.
            if strategy_profile.get_action(player) is not None:
                payoffs[player] = self._utility_function.calculate_utility(strategy_profile, player)
            else:
                # Assign a default payoff (e.g., 0) or raise an error for incomplete profiles
                # For robustness, we assume 0 if a player's action is not specified in the profile.
                payoffs[player] = 0.0 
        return payoffs

    def get_num_players(self) -> int:
        """
        Returns the total number of players in the game.
        """
        return len(self._players)