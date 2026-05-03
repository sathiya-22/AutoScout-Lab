import abc
import collections
import logging
import random
import time
from typing import List, Dict, Set, Tuple, Any, Optional, Callable

# Configure basic logging for the prototype
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RobustEquilibriumApproximator')

# --- 1. robust_equilibrium_approximator/core/interfaces.py ---
class IPlayer(abc.ABC):
    """Abstract interface for a game player."""
    @property
    @abc.abstractmethod
    def player_id(self) -> str:
        pass

    def __hash__(self):
        return hash(self.player_id)

    def __eq__(self, other):
        return isinstance(other, IPlayer) and self.player_id == other.player_id

    def __repr__(self):
        return f"Player({self.player_id})"

class IAction(abc.ABC):
    """Abstract interface for a player's action."""
    @property
    @abc.abstractmethod
    def action_id(self) -> str:
        pass

    def __hash__(self):
        return hash(self.action_id)

    def __eq__(self, other):
        return isinstance(other, IAction) and self.action_id == other.action_id

    def __repr__(self):
        return f"Action({self.action_id})"

class IUtilityFunction(abc.ABC):
    """Abstract interface for a game's utility function."""
    @abc.abstractmethod
    def calculate_utility(self, strategy_profile: Dict[IPlayer, IAction], player: IPlayer) -> float:
        """Calculates the utility (gain) for a specific player given a strategy profile."""
        pass

class IGame(abc.ABC):
    """Abstract interface for a game."""
    @abc.abstractmethod
    def get_players(self) -> List[IPlayer]:
        pass

    @abc.abstractmethod
    def get_actions(self, player: IPlayer) -> List[IAction]:
        pass

    @abc.abstractmethod
    def get_utility_function(self) -> IUtilityFunction:
        pass

    @abc.abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Checks if the game state is terminal (for sequential games, not strictly needed for normal form)."""
        pass

    @abc.abstractmethod
    def get_initial_state(self) -> Any:
        """Returns the initial state of the game."""
        pass

class ICoalitionRule(abc.ABC):
    """Abstract interface for generating potential coalitions."""
    @abc.abstractmethod
    def generate_coalitions(self, players: List[IPlayer]) -> List[Set[IPlayer]]:
        """Generates a list of possible coalitions from the given players."""
        pass

class IApproximator(abc.ABC):
    """Abstract interface for an equilibrium approximator."""
    @abc.abstractmethod
    def solve(self, game: IGame, coalition_rule: ICoalitionRule, **kwargs) -> Dict[IPlayer, IAction]:
        """
        Solves the game to find an approximate robust equilibrium strategy profile.
        Returns the approximate strategy profile for all players.
        """
        pass

# --- 2. robust_equilibrium_approximator/core/game_state.py ---
class GameState:
    """
    Represents the current state of a game. For normal form games, this is primarily
    the set of players and their available actions, possibly current strategy profile.
    For sequential games, it would include turn order, board state, etc.
    """
    def __init__(self, players: List[IPlayer], current_strategy_profile: Optional[Dict[IPlayer, IAction]] = None):
        self._players = players
        self._current_strategy_profile = current_strategy_profile if current_strategy_profile is not None else {}

    @property
    def players(self) -> List[IPlayer]:
        return self._players

    @property
    def current_strategy_profile(self) -> Dict[IPlayer, IAction]:
        return self._current_strategy_profile

    def update_strategy(self, player: IPlayer, action: IAction) -> 'GameState':
        """Returns a new GameState with the updated strategy for a specific player."""
        new_profile = self._current_strategy_profile.copy()
        new_profile[player] = action
        return GameState(self._players, new_profile)

    def __repr__(self):
        return f"GameState(players={[p.player_id for p in self._players]}, profile={self._current_strategy_profile})"

# --- 3. robust_equilibrium_approximator/core/strategy_profile.py ---
class StrategyProfile:
    """Manages and evaluates strategy profiles for all players."""
    def __init__(self, game: IGame, profile: Dict[IPlayer, IAction]):
        self._game = game
        self._profile = profile
        self._utility_function = game.get_utility_function()

    @property
    def profile(self) -> Dict[IPlayer, IAction]:
        return self._profile

    def get_player_action(self, player: IPlayer) -> Optional[IAction]:
        """Returns the action chosen by a specific player in this profile."""
        return self._profile.get(player)

    def evaluate_player_utility(self, player: IPlayer) -> float:
        """Evaluates the utility for a specific player given this strategy profile."""
        return self._utility_function.calculate_utility(self._profile, player)

    def create_deviating_profile(self, deviating_coalition: Set[IPlayer],
                                 deviating_actions: Dict[IPlayer, IAction]) -> 'StrategyProfile':
        """
        Creates a new strategy profile where the deviating_coalition members
        choose `deviating_actions`, and others stick to the original profile.
        """
        new_profile = self._profile.copy()
        for player in deviating_coalition:
            if player not in deviating_actions:
                raise ValueError(f"Action not provided for deviating player {player.player_id}")
            new_profile[player] = deviating_actions[player]
        return StrategyProfile(self._game, new_profile)

    def __repr__(self):
        profile_str = ", ".join(f"{p.player_id}: {a.action_id}" for p, a in self._profile.items())
        return f"StrategyProfile({profile_str})"

# --- 4. robust_equilibrium_approximator/games/base.py ---
# (Abstract base classes are already in interfaces.py and core.py)

# --- 5. robust_equilibrium_approximator/games/example_min_gain_game.py ---
class BasicPlayer(IPlayer):
    def __init__(self, player_id: str):
        self._player_id = player_id

    @property
    def player_id(self) -> str:
        return self._player_id

class BasicAction(IAction):
    def __init__(self, action_id: str):
        self._action_id = action_id

    @property
    def action_id(self) -> str:
        return self._action_id

class ExampleMinGainUtility(IUtilityFunction):
    """
    A simple utility function for a two-player game.
    Imagine a coordination game or a simple resource allocation.
    Utility for a player depends on their own action and the other player's action.
    This aims to be a simplified representation where
    Player 1 prefers action A1, Player 2 prefers A2.
    If both pick "Cooperate", they both get 5. If one "Defect" and other "Cooperate",
    Defector gets 10, Cooperator gets 0. If both "Defect", both get 1.
    This is a basic Prisoner's Dilemma like structure.
    """
    def calculate_utility(self, strategy_profile: Dict[IPlayer, IAction], player: IPlayer) -> float:
        p1 = BasicPlayer("P1")
        p2 = BasicPlayer("P2")
        a_cooperate = BasicAction("Cooperate")
        a_defect = BasicAction("Defect")

        p1_action = strategy_profile.get(p1)
        p2_action = strategy_profile.get(p2)

        if p1_action is None or p2_action is None:
            return 0.0 # Should not happen in a complete profile

        if p1_action == a_cooperate and p2_action == a_cooperate:
            return 5.0 # Both cooperate
        elif p1_action == a_defect and p2_action == a_cooperate:
            return 10.0 if player == p1 else 0.0 # P1 defects, P2 cooperates
        elif p1_action == a_cooperate and p2_action == a_defect:
            return 0.0 if player == p1 else 10.0 # P1 cooperates, P2 defects
        elif p1_action == a_defect and p2_action == a_defect:
            return 1.0 # Both defect
        else:
            return 0.0 # Unknown actions, default to 0

class ExampleMinGainGame(IGame):
    """
    A concrete implementation of a game for the minimum-gain analogue.
    A simple 2-player game with 2 actions each.
    """
    def __init__(self):
        self._players = [BasicPlayer("P1"), BasicPlayer("P2")]
        self._actions = {
            self._players[0]: [BasicAction("Cooperate"), BasicAction("Defect")],
            self._players[1]: [BasicAction("Cooperate"), BasicAction("Defect")]
        }
        self._utility_function = ExampleMinGainUtility()

    def get_players(self) -> List[IPlayer]:
        return self._players

    def get_actions(self, player: IPlayer) -> List[IAction]:
        return self._actions.get(player, [])

    def get_utility_function(self) -> IUtilityFunction:
        return self._utility_function

    def is_terminal(self, state: Any) -> bool:
        # For a normal form game, it's always terminal after actions are chosen.
        # This implementation simplifies to a single-shot game.
        return True

    def get_initial_state(self) -> Any:
        return GameState(self.get_players()) # Empty strategy profile initially

# --- 6. robust_equilibrium_approximator/coalitions/rules.py ---
class AllSubsetsCoalitionRule(ICoalitionRule):
    """Generates all non-empty subsets of players as potential coalitions."""
    def generate_coalitions(self, players: List[IPlayer]) -> List[Set[IPlayer]]:
        coalitions: List[Set[IPlayer]] = []
        num_players = len(players)
        for i in range(1, 1 << num_players): # Iterate from 1 to 2^N - 1 to get non-empty subsets
            coalition_set = set()
            for j in range(num_players):
                if (i >> j) & 1:
                    coalition_set.add(players[j])
            coalitions.append(coalition_set)
        # Exclude the full set if only 'proper' coalitions are desired,
        # but for strong equilibrium, the grand coalition can also deviate.
        return coalitions

# --- 7. robust_equilibrium_approximator/coalitions/manager.py ---
class CoalitionManager:
    """
    Responsible for dynamically forming coalitions, validating them,
    and evaluating their deviation incentives and potential gains.
    """
    def __init__(self, game: IGame, coalition_rule: ICoalitionRule):
        self._game = game
        self._coalition_rule = coalition_rule
        self._all_players = game.get_players()
        # Pre-generate all potential coalitions once if the rule is static
        self._potential_coalitions = self._coalition_rule.generate_coalitions(self._all_players)
        logger.info(f"Initialized CoalitionManager with {len(self._potential_coalitions)} potential coalitions.")

    def get_potential_coalitions(self) -> List[Set[IPlayer]]:
        return self._potential_coalitions

    def evaluate_coalition_deviation(self,
                                     current_profile: StrategyProfile,
                                     coalition: Set[IPlayer],
                                     verbose: bool = False) -> Tuple[float, Optional[Dict[IPlayer, IAction]]]:
        """
        Evaluates the maximum gain a coalition can achieve by deviating from
        the `current_profile`, assuming others stick to their actions.
        Returns the maximum aggregate gain and the deviating actions that achieve it.
        This is a simplified exhaustive search for the best deviation for the coalition.
        For larger games, this would need approximation/sampling too.
        """
        if not coalition:
            return 0.0, None

        logger.debug(f"Evaluating deviation for coalition: {[p.player_id for p in coalition]}")

        # Get actions for players outside the coalition (they don't deviate)
        non_coalition_players = [p for p in self._all_players if p not in coalition]
        non_coalition_actions = {p: current_profile.get_player_action(p) for p in non_coalition_players}

        max_coalition_gain = -float('inf')
        best_deviating_actions: Optional[Dict[IPlayer, IAction]] = None

        # Generate all possible joint actions for the coalition members
        coalition_actions_list: List[List[IAction]] = [self._game.get_actions(p) for p in coalition]
        from itertools import product
        all_coalition_joint_actions = product(*coalition_actions_list)

        for deviating_joint_actions_tuple in all_coalition_joint_actions:
            deviating_actions_map = dict(zip(coalition, deviating_joint_actions_tuple))

            # Combine non-coalition actions with current deviating actions
            hypothetical_profile_map: Dict[IPlayer, IAction] = {**non_coalition_actions, **deviating_actions_map}

            # Create a StrategyProfile for evaluation
            hypothetical_profile = StrategyProfile(self._game, hypothetical_profile_map)

            # Calculate aggregate gain for the coalition
            current_aggregate_gain = sum(current_profile.evaluate_player_utility(p) for p in coalition)
            deviating_aggregate_gain = sum(hypothetical_profile.evaluate_player_utility(p) for p in coalition)
            
            # The "gain" in "minimum-gain analogue" usually refers to the *additional* gain.
            # A strong equilibrium allows no coalition to *strictly improve* all its members.
            # Here, we approximate the minimum gain by minimizing the maximum aggregate gain *achievable* by a coalition.
            # So we look for the max possible deviating_aggregate_gain
            
            # For "minimum-gain analogue of strong equilibrium", we're looking for strategies such that
            # *no* coalition can achieve a *strictly greater gain for all its members*.
            # This is hard to model directly with aggregate gain.
            # Let's adjust for "aggregate gain" as per solution sketch where it says "minimize the smallest gain achievable by any deviating coalition"
            # which can be interpreted as minimizing the maximum gain achievable by the "worst" coalition.
            # Let's use the strictly improved criterion for simplicity for now:
            # A deviation is beneficial if *all* members strictly improve their utility.
            all_members_strictly_improved = True
            for member in coalition:
                if hypothetical_profile.evaluate_player_utility(member) <= current_profile.evaluate_player_utility(member):
                    all_members_strictly_improved = False
                    break
            
            if all_members_strictly_improved:
                # If all members strictly improve, this is a valid deviation.
                # We want to find the deviation that maximizes this positive aggregate change.
                # Or, if we are trying to find an equilibrium where such deviations don't exist,
                # we want to find a strategy profile where this deviation gain is minimized.
                # For MCTS reward, we want to penalize this.
                current_total_gain_if_all_strictly_improve = deviating_aggregate_gain - current_aggregate_gain

                if current_total_gain_if_all_strictly_improve > max_coalition_gain:
                    max_coalition_gain = current_total_gain_if_all_strictly_improve
                    best_deviating_actions = deviating_actions_map
                    if verbose:
                        logger.debug(f"  New best deviation for coalition {[p.player_id for p in coalition]}: {deviating_actions_map}, aggregate gain improvement: {max_coalition_gain}")
        
        # If no strictly improving deviation found, max_coalition_gain remains -inf. Return 0.
        if max_coalition_gain == -float('inf'):
            return 0.0, None

        return max_coalition_gain, best_deviating_actions

# --- 8. robust_equilibrium_approximator/approximators/mcts_solver.py ---
class MCTSNode:
    """Represents a node in the MCTS tree."""
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None,
                 player_to_move: Optional[IPlayer] = None, action_taken: Optional[IAction] = None):
        self.game_state = game_state
        self.parent = parent
        self.player_to_move = player_to_move # Player who chose action to reach this state
        self.action_taken = action_taken # Action taken to reach this state

        self.visits = 0
        self.total_reward = 0.0 # Reward accumulated from simulations passing through this node
        self.children: Dict[IAction, 'MCTSNode'] = {} # Actions lead to child nodes

    def is_fully_expanded(self, game: IGame) -> bool:
        """Check if all possible actions for the current player have been explored."""
        # This MCTS is designed for finding a strategy profile (normal form game), not a sequential game.
        # So, 'player to move' is more about selecting an action for *a* player to explore the profile space.
        # A node represents a partial strategy profile.
        # For a normal form game, we explore strategy profiles, so a node might be a partial assignment.
        # For this MCTS, let's assume a node represents a *partial strategy profile*,
        # and expanding means assigning an action for the 'next' unassigned player.
        # This is a simplification. A more complex MCTS would define states based on which player
        # is currently selecting their strategy in the search, or explore the space of full profiles.

        # For simplicity, let's say a node is fully expanded if it has simulated
        # strategies for all players, or if all actions for the current player being
        # considered for strategy assignment have children.
        # This MCTS focuses on strategy profile generation, not sequential moves.
        # We'll use a simplified expansion: If this node is not terminal and has no children yet,
        # it's not fully expanded. If it has children, it's considered expanded for the purpose of
        # selecting an action for the *next player in sequence*.
        if len(self.game_state.current_strategy_profile) == len(game.get_players()):
            return True # This node represents a full strategy profile

        # Otherwise, if we're building a strategy profile by assigning actions sequentially
        # to players, we'd need to check if all actions for the *next player to assign*
        # have been explored as children.
        # For this prototype, let's assume expansion means creating a child for a random unassigned player's action.
        return len(self.children) > 0 # Simple heuristic for "expanded enough" for a single step

    def get_unexplored_actions(self, game: IGame, player_to_assign: IPlayer) -> List[IAction]:
        """Returns actions for player_to_assign that haven't been explored from this node."""
        all_actions = game.get_actions(player_to_assign)
        explored_actions = set(self.children.keys())
        return [action for action in all_actions if action not in explored_actions]

    def best_child(self, c_param: float = 1.414) -> Optional['MCTSNode']:
        """Selects the best child node using UCT (Upper Confidence Bound 1 applied to trees)."""
        if not self.children:
            return None
        
        log_total_visits = math.log(self.visits) if self.visits > 0 else 0
        
        best_uct = -float('inf')
        best_node = None

        for child_node in self.children.values():
            if child_node.visits == 0:
                # Prioritize unexplored nodes by giving them max UCT value
                uct_value = float('inf')
            else:
                uct_value = (child_node.total_reward / child_node.visits) + \
                            c_param * math.sqrt(log_total_visits / child_node.visits)
            
            if uct_value > best_uct:
                best_uct = uct_value
                best_node = child_node
        return best_node

import math

class MCTSSolver(IApproximator):
    """
    Implements a Monte Carlo Tree Search (MCTS) approach to approximate
    a robust equilibrium strategy profile.

    The 'reward' mechanism for MCTS is tailored to minimize the maximum gain
    achievable by any deviating coalition, guiding the search towards
    robust equilibrium strategies. A higher penalty (negative reward)
    is given for strategy profiles that allow large coalition deviations.
    """
    def __init__(self, game: IGame, coalition_manager: CoalitionManager,
                 iterations: int = 1000, exploration_constant: float = 1.414):
        self._game = game
        self._coalition_manager = coalition_manager
        self._iterations = iterations
        self._exploration_constant = exploration_constant
        logger.info(f"MCTSSolver initialized with {iterations} iterations.")

    def solve(self, game: IGame, coalition_rule: ICoalitionRule, **kwargs) -> Dict[IPlayer, IAction]:
        self._iterations = kwargs.get('iterations', self._iterations)
        self._exploration_constant = kwargs.get('exploration_constant', self._exploration_constant)

        root = MCTSNode(GameState(self._game.get_players()))

        for i in range(self._iterations):
            node = self._select(root)
            if not len(node.game_state.current_strategy_profile) == len(game.get_players()):
                node = self._expand(node, game)
            reward = self._simulate(node, game)
            self._backpropagate(node, reward)

            if (i + 1) % (self._iterations // 10 if self._iterations >= 10 else 1) == 0:
                logger.debug(f"MCTS iteration {i+1}/{self._iterations} complete.")

        # After MCTS, select the best action based on visit counts from the root's children.
        # The best 'final' strategy profile is usually derived from the child with most visits
        # or highest average reward, iterated down the tree.
        # For a normal form game, we want the final strategy profile, not just one move.
        # This simplification will find the most visited *full* strategy profile explored.
        
        # To get the best full strategy profile:
        # Traverse from root to a leaf node that represents a full strategy profile,
        # always picking the child with the highest visit count.
        current_node = root
        final_strategy_profile = {}

        while len(current_node.game_state.current_strategy_profile) < len(game.get_players()):
            if not current_node.children:
                # Should not happen if MCTS ran enough iterations to explore full paths
                break 
            
            best_child_for_profile = None
            max_visits = -1

            for child_action, child_node in current_node.children.items():
                if child_node.visits > max_visits:
                    max_visits = child_node.visits
                    best_child_for_profile = child_node
            
            if best_child_for_profile:
                current_node = best_child_for_profile
                final_strategy_profile = current_node.game_state.current_strategy_profile # Accumulate
            else:
                break # Fallback if no children were visited

        if not final_strategy_profile:
            logger.warning("MCTS failed to find a complete strategy profile. Returning a random one as fallback.")
            final_strategy_profile = self._get_random_strategy_profile(game)


        logger.info(f"MCTS completed. Best approximate strategy profile: {final_strategy_profile}")
        return final_strategy_profile

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selects a node to expand based on UCT."""
        while len(node.game_state.current_strategy_profile) == len(self._game.get_players()): # While full profile
            # If a full strategy profile has been reached, we can't expand further with actions.
            # We treat this as a terminal state for MCTS decision purposes for actions.
            # In our setup, we're building a *full* strategy profile in the expansion phase.
            # So, if a node represents a full profile, it's a "leaf" for selection.
            # If it's not a full profile but has no children, it's also a "leaf" to be expanded.
            
            # This needs to find the correct "player to move" in the context of building a profile.
            # Let's assume an ordered assignment of players.
            
            if not node.children: # If this node is not fully expanded or has no children
                break # This node needs expansion/simulation

            node = node.best_child(self._exploration_constant)
            if node is None: # Should not happen if children exist, but for safety
                break
        return node


    def _expand(self, node: MCTSNode, game: IGame) -> MCTSNode:
        """Expands the node by adding a new child for an unexplored action of an unassigned player."""
        players = game.get_players()
        assigned_players = set(node.game_state.current_strategy_profile.keys())
        unassigned_players = [p for p in players if p not in assigned_players]

        if not unassigned_players:
            # All players assigned, this node represents a full strategy profile, cannot expand further in this way.
            # This should ideally be handled by _select, by not calling expand on a full node.
            return node

        # For this prototype, pick the first unassigned player to assign an action
        player_to_assign = unassigned_players[0]
        unexplored_actions = node.get_unexplored_actions(game, player_to_assign)

        if not unexplored_actions:
            # All actions for this player have been explored from this node.
            # This means the node is "fully expanded" with respect to this player.
            # If no child was created, this implies we need to simulate from here or move up.
            return node # This node is expanded enough or cannot be expanded further for this player.

        # Pick a random unexplored action
        action = random.choice(unexplored_actions)
        
        new_state = node.game_state.update_strategy(player_to_assign, action)
        child_node = MCTSNode(new_state, parent=node, player_to_move=player_to_assign, action_taken=action)
        node.children[action] = child_node
        return child_node

    def _simulate(self, node: MCTSNode, game: IGame) -> float:
        """
        Simulates a rollout from the current node's state by randomly assigning
        strategies to remaining players, then evaluates the robustness.
        """
        current_profile_map = node.game_state.current_strategy_profile.copy()
        
        # Complete the strategy profile for any unassigned players
        all_players = game.get_players()
        for player in all_players:
            if player not in current_profile_map:
                actions = game.get_actions(player)
                if not actions:
                    logger.warning(f"Player {player.player_id} has no actions defined. Assigning None.")
                    current_profile_map[player] = None # Or handle error
                else:
                    current_profile_map[player] = random.choice(actions)

        final_profile = StrategyProfile(game, current_profile_map)

        # Evaluate the "robustness" of this final_profile
        # Reward is based on minimizing the maximum gain achievable by any deviating coalition.
        # So, a higher max gain means a lower (more negative) reward.
        max_coalition_deviation_gain = 0.0

        for coalition in self._coalition_manager.get_potential_coalitions():
            try:
                deviation_gain, _ = self._coalition_manager.evaluate_coalition_deviation(final_profile, coalition)
                max_coalition_deviation_gain = max(max_coalition_deviation_gain, deviation_gain)
            except Exception as e:
                logger.error(f"Error evaluating coalition deviation for {coalition}: {e}")
                # Penalize heavily if evaluation fails to avoid problematic profiles
                max_coalition_deviation_gain = float('inf')
                break

        # Reward formulation: A robust equilibrium minimizes this max_coalition_deviation_gain.
        # So, we want high rewards for low max_coalition_deviation_gain.
        # If max_coalition_deviation_gain is 0, it's a strong equilibrium (high reward).
        # If it's positive, we penalize it.
        reward = -max_coalition_deviation_gain # Make it a minimization problem for MCTS

        logger.debug(f"Simulated profile {final_profile}. Max deviation gain: {max_coalition_deviation_gain}, Reward: {reward}")
        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Updates visit counts and total rewards up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _get_random_strategy_profile(self, game: IGame) -> Dict[IPlayer, IAction]:
        """Helper to generate a random strategy profile as a fallback."""
        profile: Dict[IPlayer, IAction] = {}
        for player in game.get_players():
            actions = game.get_actions(player)
            if actions:
                profile[player] = random.choice(actions)
            else:
                profile[player] = BasicAction("NO_ACTION") # Placeholder for players with no actions
        return profile


# --- 9. robust_equilibrium_approximator/approximators/rl_agent_solver.py ---
# Mock RL agent solver - actual implementation would use a library like Stable Baselines3
class RLAgentSolver(IApproximator):
    def __init__(self, game: IGame, coalition_manager: CoalitionManager, episodes: int = 100):
        self._game = game
        self._coalition_manager = coalition_manager
        self._episodes = episodes
        logger.info(f"RLAgentSolver initialized for {episodes} episodes.")

    def solve(self, game: IGame, coalition_rule: ICoalitionRule, **kwargs) -> Dict[IPlayer, IAction]:
        self._episodes = kwargs.get('episodes', self._episodes)
        logger.info(f"Simulating RL agent training and solving for {self._episodes} episodes...")

        # In a real RL setup, this would involve defining an environment,
        # creating an agent (e.g., PPO, DQN), training it, and then extracting its policy.
        # For this mock, we'll simulate a learning process and return a "learned" strategy.

        best_profile_found: Dict[IPlayer, IAction] = {}
        min_worst_case_gain = float('inf')

        for i in range(self._episodes):
            # Simulate an RL agent proposing a strategy profile
            current_profile_map = self._generate_random_strategy_profile(game)
            current_profile = StrategyProfile(game, current_profile_map)

            worst_case_deviation_gain = 0.0
            for coalition in self._coalition_manager.get_potential_coalitions():
                try:
                    gain, _ = self._coalition_manager.evaluate_coalition_deviation(current_profile, coalition)
                    worst_case_deviation_gain = max(worst_case_deviation_gain, gain)
                except Exception as e:
                    logger.error(f"RL evaluation error for coalition {coalition}: {e}")
                    worst_case_deviation_gain = float('inf')
                    break
            
            # RL agent learns to minimize this worst_case_deviation_gain
            # The "reward" would be -worst_case_deviation_gain
            
            if worst_case_deviation_gain < min_worst_case_gain:
                min_worst_case_gain = worst_case_deviation_gain
                best_profile_found = current_profile_map
            
            if (i + 1) % (self._episodes // 10 if self._episodes >= 10 else 1) == 0:
                logger.debug(f"RL simulation {i+1}/{self._episodes} - Current min worst gain: {min_worst_case_gain}")

        if not best_profile_found:
            logger.warning("RL Agent Solver failed to find a valid profile. Returning a random one.")
            best_profile_found = self._generate_random_strategy_profile(game)

        logger.info(f"RLAgentSolver completed. Best approximate strategy profile: {best_profile_found}")
        return best_profile_found

    def _generate_random_strategy_profile(self, game: IGame) -> Dict[IPlayer, IAction]:
        """Helper to generate a random strategy profile."""
        profile: Dict[IPlayer, IAction] = {}
        for player in game.get_players():
            actions = game.get_actions(player)
            if actions:
                profile[player] = random.choice(actions)
            else:
                profile[player] = BasicAction("NO_ACTION")
        return profile

# --- 10. robust_equilibrium_approximator/approximators/hybrid_solver.py ---
class HybridSolver(IApproximator):
    """
    Combines MCTS and RL approaches.
    For this prototype, it will sequentially run one then the other.
    A more advanced version would have MCTS guide RL exploration or use RL policies in MCTS rollouts.
    """
    def __init__(self, game: IGame, coalition_manager: CoalitionManager,
                 mcts_iterations: int = 500, rl_episodes: int = 50):
        self._game = game
        self._coalition_manager = coalition_manager
        self._mcts_solver = MCTSSolver(game, coalition_manager, iterations=mcts_iterations)
        self._rl_solver = RLAgentSolver(game, coalition_manager, episodes=rl_episodes)
        logger.info("HybridSolver initialized (MCTS then RL approach).")

    def solve(self, game: IGame, coalition_rule: ICoalitionRule, **kwargs) -> Dict[IPlayer, IAction]:
        logger.info("HybridSolver: Starting MCTS phase...")
        mcts_result = self._mcts_solver.solve(game, coalition_rule, **kwargs)
        
        # In a real hybrid, MCTS output might inform RL's initial policy or exploration.
        # For this prototype, we'll just see which one gives a "better" result if we
        # run RL from scratch, or simply use the MCTS result.
        # Let's say MCTS finds a good starting point, and RL tries to refine it.
        # This implementation simplifies to MCTS then RL independently.

        # To show a hybrid, let's say RL is "seeded" with MCTS's best profile.
        # This is a conceptual seed, not a direct API call here.
        logger.info("HybridSolver: Starting RL refinement phase (conceptually building upon MCTS insight)...")
        rl_result = self._rl_solver.solve(game, coalition_rule, **kwargs)

        # For this prototype, we'll just return the RL result, assuming it "refined" MCTS.
        # A proper hybrid would compare or combine.
        logger.info("HybridSolver completed. Returning RL refinement result.")
        return rl_result


# --- 11. robust_equilibrium_approximator/utils/logging_config.py ---
# Already handled at the top of the file

# --- 12. robust_equilibrium_approximator/utils/metrics.py ---
class Metrics:
    """Provides utilities for evaluating approximation quality."""
    @staticmethod
    def calculate_worst_case_deviation_gain(game: IGame,
                                            strategy_profile_map: Dict[IPlayer, IAction],
                                            coalition_manager: CoalitionManager) -> float:
        """
        Calculates the maximum aggregate gain any coalition can achieve by deviating
        from the given strategy profile. This is the value we want to minimize.
        """
        profile = StrategyProfile(game, strategy_profile_map)
        worst_gain = 0.0
        for coalition in coalition_manager.get_potential_coalitions():
            try:
                gain, _ = coalition_manager.evaluate_coalition_deviation(profile, coalition)
                worst_gain = max(worst_gain, gain)
            except Exception as e:
                logger.error(f"Error in metrics for coalition {coalition}: {e}")
                return float('inf') # Indicate a problematic profile
        return worst_gain

    @staticmethod
    def is_strong_equilibrium(game: IGame,
                              strategy_profile_map: Dict[IPlayer, IAction],
                              coalition_manager: CoalitionManager,
                              epsilon: float = 1e-9) -> bool:
        """
        Checks if the given strategy profile is a strong equilibrium,
        meaning no coalition can strictly improve all its members' utilities by deviating.
        """
        # The evaluate_coalition_deviation already calculates the maximum *aggregate* gain
        # where *all members strictly improve*. If this max gain is <= epsilon (near zero),
        # then no such deviation exists.
        worst_gain = Metrics.calculate_worst_case_deviation_gain(game, strategy_profile_map, coalition_manager)
        return worst_gain <= epsilon
    
    @staticmethod
    def compare_profiles(game: IGame,
                         profile_a: Dict[IPlayer, IAction],
                         profile_b: Dict[IPlayer, IAction],
                         coalition_manager: CoalitionManager) -> Tuple[float, float]:
        """Compares two profiles based on their worst-case deviation gain."""
        gain_a = Metrics.calculate_worst_case_deviation_gain(game, profile_a, coalition_manager)
        gain_b = Metrics.calculate_worst_case_deviation_gain(game, profile_b, coalition_manager)
        return gain_a, gain_b

# --- 13. robust_equilibrium_approximator/utils/heuristics.py ---
class Heuristics:
    """
    Placeholder for general-purpose heuristic functions used in sampling or pruning strategies
    within the approximators and coalition manager.
    """
    @staticmethod
    def simple_coalition_pruning_heuristic(all_players: List[IPlayer], max_size: int = 2) -> List[Set[IPlayer]]:
        """
        A heuristic to prune the number of coalitions by only considering
        coalitions up to a certain maximum size.
        """
        pruned_coalitions: List[Set[IPlayer]] = []
        from itertools import combinations
        for k in range(1, max_size + 1):
            for combo in combinations(all_players, k):
                pruned_coalitions.append(set(combo))
        logger.debug(f"Generated {len(pruned_coalitions)} coalitions using pruning heuristic (max_size={max_size}).")
        return pruned_coalitions

# --- 14. robust_equilibrium_approximator/experiments/runner.py ---
class ExperimentRunner:
    """Orchestrates experiment execution."""
    def __init__(self, game: IGame, coalition_rule: ICoalitionRule):
        self._game = game
        self._coalition_rule = coalition_rule
        self._coalition_manager = CoalitionManager(game, coalition_rule)
        logger.info("ExperimentRunner initialized.")

    def run_experiment(self, approximator: IApproximator, label: str, **kwargs) -> Tuple[Dict[IPlayer, IAction], float]:
        """
        Runs a single experiment with a given approximator and returns
        the found strategy profile and its worst-case deviation gain.
        """
        logger.info(f"--- Running Experiment: {label} ---")
        start_time = time.time()
        
        try:
            approx_profile = approximator.solve(self._game, self._coalition_rule, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            worst_case_gain = Metrics.calculate_worst_case_deviation_gain(self._game, approx_profile, self._coalition_manager)
            
            logger.info(f"Experiment '{label}' completed in {duration:.2f} seconds.")
            logger.info(f"Approximate Profile: {approx_profile}")
            logger.info(f"Worst-Case Deviation Gain: {worst_case_gain:.4f}")
            logger.info(f"Is Strong Equilibrium (approx): {Metrics.is_strong_equilibrium(self._game, approx_profile, self._coalition_manager):.4f}")
            return approx_profile, worst_case_gain
        except Exception as e:
            logger.error(f"Error running experiment '{label}': {e}", exc_info=True)
            # Return empty profile and inf gain on error
            return {}, float('inf')

# --- 15. robust_equilibrium_approximator/experiments/results_analyzer.py ---
class ResultsAnalyzer:
    """Tools for analyzing and visualizing experiment outcomes."""
    def __init__(self):
        self.results: Dict[str, Tuple[Dict[IPlayer, IAction], float]] = {}
        logger.info("ResultsAnalyzer initialized.")

    def add_result(self, label: str, profile: Dict[IPlayer, IAction], worst_gain: float):
        self.results[label] = (profile, worst_gain)
        logger.info(f"Added result for '{label}': Worst-case gain={worst_gain:.4f}")

    def summarize_results(self):
        """Prints a summary of all collected results."""
        logger.info("\n--- Experiment Results Summary ---")
        if not self.results:
            logger.info("No results to display.")
            return

        sorted_results = sorted(self.results.items(), key=lambda item: item[1][1])

        for label, (profile, worst_gain) in sorted_results:
            logger.info(f"Approximator: {label}")
            logger.info(f"  Approximate Strategy: {profile}")
            logger.info(f"  Worst-Case Deviation Gain: {worst_gain:.4f}")
            if worst_gain == 0.0:
                logger.info(f"  -> Found an approximate Strong Equilibrium!")
            elif worst_gain < float('inf'):
                logger.info(f"  -> Robustness (lower is better): {worst_gain:.4f}")
            else:
                logger.info(f"  -> ERROR / UNSTABLE")
        logger.info("----------------------------------")

    # In a full implementation, this would also include plotting functions
    # (e.g., using matplotlib, seaborn) for visualization.


# --- Main execution block ---
if __name__ == "__main__":
    logger.info("Starting Robust Equilibrium Approximator prototype...")

    try:
        # 1. Define the game
        game = ExampleMinGainGame()
        players = game.get_players()
        actions_p1 = game.get_actions(players[0])
        actions_p2 = game.get_actions(players[1])
        logger.info(f"Game defined with players: {[p.player_id for p in players]}")
        logger.info(f"  P1 actions: {[a.action_id for a in actions_p1]}")
        logger.info(f"  P2 actions: {[a.action_id for a in actions_p2]}")

        # 2. Define coalition rules
        # Using a simple rule for demonstration
        coalition_rule = AllSubsetsCoalitionRule()
        
        # Optionally use a heuristic for pruning coalitions in very large games
        # pruned_coalitions_list = Heuristics.simple_coalition_pruning_heuristic(players, max_size=1)
        # class PrunedCoalitionRule(ICoalitionRule):
        #     def generate_coalitions(self, players: List[IPlayer]) -> List[Set[IPlayer]]:
        #         return pruned_coalitions_list
        # coalition_rule = PrunedCoalitionRule()


        # 3. Initialize Experiment Runner and Coalition Manager
        runner = ExperimentRunner(game, coalition_rule)
        results_analyzer = ResultsAnalyzer()

        # 4. Instantiate and run different approximators
        
        # MCTS Solver
        mcts_iterations = 5000 # Increased for better exploration in prototype
        mcts_solver = MCTSSolver(game, runner._coalition_manager, iterations=mcts_iterations)
        mcts_profile, mcts_gain = runner.run_experiment(mcts_solver, "MCTS Approximator", iterations=mcts_iterations)
        results_analyzer.add_result("MCTS Approximator", mcts_profile, mcts_gain)

        # RL Agent Solver
        rl_episodes = 2000 # Increased for better "learning" simulation
        rl_solver = RLAgentSolver(game, runner._coalition_manager, episodes=rl_episodes)
        rl_profile, rl_gain = runner.run_experiment(rl_solver, "RL Agent Approximator", episodes=rl_episodes)
        results_analyzer.add_result("RL Agent Approximator", rl_profile, rl_gain)

        # Hybrid Solver (MCTS then RL)
        hybrid_mcts_iter = 3000
        hybrid_rl_episodes = 1000
        hybrid_solver = HybridSolver(game, runner._coalition_manager,
                                     mcts_iterations=hybrid_mcts_iter, rl_episodes=hybrid_rl_episodes)
        hybrid_profile, hybrid_gain = runner.run_experiment(hybrid_solver, "Hybrid (MCTS+RL) Approximator",
                                                             iterations=hybrid_mcts_iter, episodes=hybrid_rl_episodes)
        results_analyzer.add_result("Hybrid (MCTS+RL) Approximator", hybrid_profile, hybrid_gain)

        # 5. Analyze and display results
        results_analyzer.summarize_results()

    except Exception as e:
        logger.critical(f"An unhandled error occurred during prototype execution: {e}", exc_info=True)
        # Exit with an error code to indicate failure
        exit(1)

    logger.info("Robust Equilibrium Approximator prototype finished.")