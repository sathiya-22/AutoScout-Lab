```python
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Type
import copy
import uuid
from datetime import datetime

# --- Placeholder Models (These would ideally come from framework.action_schemas) ---
# In a full implementation, you would import these from 'framework.action_schemas'.
# For this file's implementation, they are defined here to satisfy type hints.

class BaseAgentAction(BaseModel):
    """
    Base model for any agent action.
    This formalizes the structure of actions, crucial for semantic comparison
    and state reconciliation.
    """
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "unknown_agent"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action_type: str = "generic_action"
    payload: Dict[str, Any] = {} # Specific action details, e.g., {'prompt': '...', 'llm_output': '...'}

class WorkflowState(BaseModel):
    """
    Represents the canonical state of the entire system at a given point in time.
    All system-level data that agents operate on or contribute to should be
    formalized within this model.
    """
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 0  # Monotonically increasing version for state changes
    data: Dict[str, Any] = {} # The actual structured data representing the system's state

# --- Custom Exceptions (These would ideally come from framework.exceptions) ---
# In a full implementation, you would import these from 'framework.exceptions'.

class StateManagerError(Exception):
    """Base exception for StateManager operations."""
    pass

class TransactionError(StateManagerError):
    """Raised when a transaction operation fails (e.g., commit without active transaction)."""
    pass

class CheckpointError(StateManagerError):
    """Raised when a checkpoint operation fails (e.g., creating during transaction, ID conflict)."""
    pass

# --- State Manager Implementation ---
class StateManager:
    """
    The heart of the 'Deterministic Agentic Workflow Framework'.
    Manages the canonical, formalized representation of the system's state and
    a history of all agent actions. Supports atomic transactions, checkpointing,
    and rollback functionality to ensure state consistency and recoverability.
    """
    def __init__(self, initial_state: Optional[WorkflowState] = None):
        """
        Initializes the StateManager with an optional initial state.

        Args:
            initial_state (Optional[WorkflowState]): The starting state of the workflow.
                                                    If None, an empty WorkflowState is created.
        """
        if initial_state and not isinstance(initial_state, WorkflowState):
            raise TypeError("initial_state must be an instance of WorkflowState.")
        
        self._current_state: WorkflowState = initial_state if initial_state else WorkflowState()
        self._action_history: List[BaseAgentAction] = []
        
        # Stores checkpoints: checkpoint_id -> {'state': WorkflowState, 'action_history_len': int}
        self._checkpoints: Dict[str, Dict[str, Any]] = {} 
        
        # Transaction management:
        # Each item in the stack is a snapshot of the committed state and action history length
        # when a transaction began. This allows nested transactions.
        self._transaction_stack: List[Dict[str, Any]] = [] 
        
        # Staged changes for the current active (innermost) transaction
        self._pending_actions: List[BaseAgentAction] = []
        self._pending_state_update: Optional[WorkflowState] = None

    @property
    def current_state(self) -> WorkflowState:
        """
        Returns a deep copy of the current effective system state. If a transaction is active, 
        returns the staged state (`_pending_state_update`). Otherwise, returns the officially
        committed state (`_current_state`).
        This ensures external consumers get an immutable snapshot and cannot inadvertently
        modify the internal state.
        """
        return copy.deepcopy(self._pending_state_update) if self._transaction_stack and self._pending_state_update else copy.deepcopy(self._current_state)

    @property
    def action_history(self) -> List[BaseAgentAction]:
        """
        Returns a deep copy of the full action history, including staged actions if a transaction
        is active. This ensures external consumers get an immutable snapshot of the history.
        """
        effective_history = self._action_history + self._pending_actions if self._transaction_stack else self._action_history
        return [copy.deepcopy(action) for action in effective_history]

    def start_transaction(self):
        """
        Starts a new transaction. All state updates and action recordings
        will be staged until `commit_transaction` or `rollback_transaction` is called.
        Supports nested transactions: each call pushes a new level onto the transaction stack.
        """
        # Save a snapshot of the current committed state and history length
        # for potential rollback of this specific transaction level.
        self._transaction_stack.append({
            'state': copy.deepcopy(self._current_state),
            'action_history_len': len(self._action_history)
        })
        
        # Initialize pending changes for this new transaction level.
        # The base for the pending state is the current effective state (which might already
        # include pending changes from an outer transaction).
        self._pending_state_update = copy.deepcopy(self.current_state) 
        self._pending_actions = []

    def commit_transaction(self):
        """
        Commits the current (innermost) transaction, applying all staged changes to the
        actual committed state and action history. If nested, these changes are then
        promoted to become the new pending base for the outer transaction.

        Raises:
            TransactionError: If no active transaction is present to commit.
        """
        if not self._transaction_stack:
            raise TransactionError("No active transaction to commit.")
        
        # Pop the current transaction's snapshot from the stack.
        self._transaction_stack.pop()
        
        # Apply the pending state update and actions to the core committed state.
        if self._pending_state_update:
            self._current_state = self._pending_state_update
            self._current_state.version += 1 # Increment state version on successful commit
        self._action_history.extend(self._pending_actions)
        
        # After committing, if there are still active transactions (meaning this was a nested commit),
        # the _current_state (now updated by the just-committed transaction) becomes the
        # new pending base for the outer transaction. Otherwise, clear pending changes.
        if self._transaction_stack:
            self._pending_state_update = copy.deepcopy(self._current_state)
            self._pending_actions = [] # Clear pending actions for this level, as they are now committed.
        else:
            self._pending_state_update = None
            self._pending_actions = []

    def rollback_transaction(self):
        """
        Rolls back the current (innermost) transaction, discarding all staged changes
        and reverting the system's committed state and action history to their state
        before this transaction began.

        Raises:
            TransactionError: If no active transaction is present to roll back.
        """
        if not self._transaction_stack:
            raise TransactionError("No active transaction to rollback.")
        
        # Restore state and action history length from the snapshot taken at transaction start.
        snapshot = self._transaction_stack.pop()
        self._current_state = snapshot['state']
        self._action_history = self._action_history[:snapshot['action_history_len']]
        
        # After rolling back, if there are still active transactions,
        # the pending state/actions for the outer transaction are reset to reflect
        # the state before the rolled-back inner transaction occurred.
        if self._transaction_stack:
            self._pending_state_update = copy.deepcopy(self._current_state)
            self._pending_actions = [] # Clear pending actions for the rolled-back level.
        else:
            self._pending_state_update = None
            self._pending_actions = []

    def update_state(self, new_state: WorkflowState):
        """
        Updates the current system state. If a transaction is active, the update
        is applied to the staged state (`_pending_state_update`).
        Otherwise, it's applied immediately to `_current_state`.

        Args:
            new_state (WorkflowState): The new state to apply. Must be an instance of WorkflowState.
        
        Raises:
            TypeError: If `new_state` is not an instance of WorkflowState.
        """
        if not isinstance(new_state, WorkflowState):
            raise TypeError("new_state must be an instance of WorkflowState.")
        
        if self._transaction_stack:
            # Update the pending state for the current (innermost) transaction.
            self._pending_state_update = new_state
        else:
            # No transaction, apply immediately to the committed state.
            self._current_state = new_state
            self._current_state.version += 1 # Increment version for direct update

    def record_action(self, action: BaseAgentAction):
        """
        Records an agent action. If a transaction is active, the action is staged
        in `_pending_actions`. Otherwise, it's added immediately to `_action_history`.

        Args:
            action (BaseAgentAction): The action to record. Must be an instance of BaseAgentAction.
        
        Raises:
            TypeError: If `action` is not an instance of BaseAgentAction.
        """
        if not isinstance(action, BaseAgentAction):
            raise TypeError("action must be an instance of BaseAgentAction.")

        if self._transaction_stack:
            self._pending_actions.append(action)
        else:
            self._action_history.append(action)

    def create_checkpoint(self, checkpoint_id: Optional[str] = None) -> str:
        """
        Creates a snapshot of the current *committed* state and action history,
        associating it with a unique ID. Checkpoints cannot be created during
        an active transaction to ensure they represent a truly consistent, committed state.

        Args:
            checkpoint_id (Optional[str]): A unique identifier for the checkpoint.
                                           If None, a UUID will be generated.
        
        Returns:
            str: The ID of the created checkpoint.
            
        Raises:
            CheckpointError: If an active transaction is present, or if `checkpoint_id`
                             already exists.
        """
        if self._transaction_stack:
            raise CheckpointError(
                "Cannot create checkpoint during an active transaction. "
                "Commit or rollback all active transactions first to ensure a consistent snapshot."
            )

        checkpoint_id = checkpoint_id if checkpoint_id else str(uuid.uuid4())
        
        if checkpoint_id in self._checkpoints:
            raise CheckpointError(f"Checkpoint with ID '{checkpoint_id}' already exists.")
        
        # Store deep copies to ensure checkpointed state and history are immutable snapshots.
        self._checkpoints[checkpoint_id] = {
            'state': copy.deepcopy(self._current_state),
            'action_history_len': len(self._action_history)
        }
        return checkpoint_id

    def rollback_to_checkpoint(self, checkpoint_id: str):
        """
        Restores the system to a previously saved checkpoint.
        This operation discards all state and actions committed *after* the checkpoint
        was created, effectively reverting the system's history to that point.
        Rollbacks cannot be performed during an active transaction.

        Args:
            checkpoint_id (str): The ID of the checkpoint to roll back to.
            
        Raises:
            CheckpointError: If an active transaction is present, or if `checkpoint_id`
                             is not found.
        """
        if self._transaction_stack:
            raise CheckpointError(
                "Cannot rollback to checkpoint during an active transaction. "
                "Commit or rollback all active transactions first."
            )

        if checkpoint_id not in self._checkpoints:
            raise CheckpointError(f"Checkpoint with ID '{checkpoint_id}' not found.")
        
        checkpoint_data = self._checkpoints[checkpoint_id]
        
        # Restore the committed state and prune the action history to the checkpoint's point.
        self._current_state = copy.deepcopy(checkpoint_data['state'])
        # Deep copy actions up to the checkpoint to ensure integrity of restored history.
        self._action_history = [copy.deepcopy(action) for action in self._action_history[:checkpoint_data['action_history_len']]]
        
        # Increment version to signify a new state version, even if it's a rollback.
        # This helps in tracking changes, even if reverting.
        self._current_state.version += 1 

    def get_state_snapshot(self) -> WorkflowState:
        """
        Returns a deep copy of the current effective state (committed or pending).
        This is equivalent to accessing the `current_state` property and ensures
        an immutable snapshot.
        """
        return self.current_state # Property already returns a deep copy
    
    def get_action_history_snapshot(self) -> List[BaseAgentAction]:
        """
        Returns a deep copy of the current effective action history (committed + pending).
        This is equivalent to accessing the `action_history` property and ensures
        an immutable snapshot.
        """
        return self.action_history # Property already returns a deep copy

    def get_checkpoint_ids(self) -> List[str]:
        """
        Returns a list of all available checkpoint IDs.
        """
        return list(self._checkpoints.keys())
```