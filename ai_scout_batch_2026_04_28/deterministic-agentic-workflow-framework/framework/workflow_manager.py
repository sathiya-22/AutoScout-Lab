```python
import logging
from typing import Dict, Any, Type, Optional, List, Callable, Union
from uuid import uuid4

# Assume these classes and models exist as per architecture
# and are available in the project's framework/ and other directories.
from framework.state_manager import StateManager, StateIdentifier
from framework.action_schemas import AgentAction, AgentOutput, WorkflowState, WorkflowStep, WorkflowContext
from framework.exceptions import WorkflowError, AgentExecutionError, StateConsistencyError
from reconciliation.reconciliation_agent import ReconciliationAgent
from verification.verifier import Verifier
from config import Config
from utils.logging_config import setup_logging

# For type hinting reconciliation strategies without circular import or needing to define them here.
# In a full implementation, these would be imported from `reconciliation.strategies`.
class ReconciliationStrategyPayload(dict):
    """A generic dictionary type for reconciliation strategy payloads."""
    pass

class ReconciliationStrategy:
    """A minimal mock for ReconciliationStrategy for type hinting."""
    type: str
    payload: ReconciliationStrategyPayload


setup_logging()
logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Orchestrates the execution of agent workflows, interacting with the StateManager
    to record each step, request checkpoints, and initiate rollbacks. It manages
    the sequence and dependencies of agent tasks, and coordinates with Verifier
    and ReconciliationAgent for inconsistency handling, aiming to mitigate
    the 'semantic idempotency' problem.
    """
    def __init__(
        self,
        state_manager: StateManager,
        verifier: Verifier,
        reconciliation_agent: ReconciliationAgent,
        config: Config,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None
    ):
        """
        Initializes the WorkflowManager.

        Args:
            state_manager: The central StateManager instance.
            verifier: The Verifier instance for semantic comparison and invariant checking.
            reconciliation_agent: The ReconciliationAgent instance for resolving inconsistencies.
            config: The centralized configuration object.
            initial_context: Optional dictionary for initial workflow context.
            workflow_id: Optional ID to resume an existing workflow; if None, a new one is generated.
        """
        self._workflow_id = workflow_id if workflow_id else str(uuid4())
        self._state_manager = state_manager
        self._verifier = verifier
        self._reconciliation_agent = reconciliation_agent
        self._config = config
        self._current_workflow_state_id: Optional[StateIdentifier] = None

        # Initialize or load workflow state
        if not self._state_manager.has_workflow_state(self._workflow_id):
            initial_state = WorkflowState(
                workflow_id=self._workflow_id,
                current_step_id=None,
                history=[],
                context=WorkflowContext(data=initial_context if initial_context else {}),
                status="INITIALIZED"
            )
            self._current_workflow_state_id = self._state_manager.initialize_workflow(initial_state)
            logger.info(f"Initialized new workflow with ID: {self._workflow_id}. Initial state ID: {self._current_workflow_state_id}")
        else:
            self._current_workflow_state_id = self._state_manager.get_latest_state_id(self._workflow_id)
            if not self._current_workflow_state_id:
                raise WorkflowError(f"Workflow {self._workflow_id} exists but no latest state ID found.")
            logger.info(f"Resuming existing workflow with ID: {self._workflow_id} from state: {self._current_workflow_state_id}")

    def _get_current_workflow_state(self) -> WorkflowState:
        """
        Helper to retrieve the current complete workflow state from the StateManager.
        Raises an error if the state cannot be retrieved or is not set.
        """
        if not self._current_workflow_state_id:
            raise WorkflowError(f"Current workflow state ID is not set for workflow {self._workflow_id}.")
        state = self._state_manager.get_workflow_state(self._current_workflow_state_id)
        if not state:
            raise StateConsistencyError(
                f"Workflow state with ID {self._current_workflow_state_id} not found for workflow {self._workflow_id}. "
                "State might have been corrupted or rolled back unexpectedly."
            )
        return state

    def register_agent_action(
        self,
        agent_id: str,
        action_type: str,
        inputs: Dict[str, Any],
        expected_outputs: Optional[Dict[str, Any]] = None,
        parent_step_id: Optional[str] = None
    ) -> AgentAction:
        """
        Registers an agent's intended action with the workflow manager.
        This creates an AgentAction object that can then be passed to execute_agent_step.

        Args:
            agent_id: The identifier of the agent.
            action_type: The type of action being performed (e.g., 'data_processing', 'decision_making').
            inputs: Dictionary of inputs for the agent's action.
            expected_outputs: Optional dictionary of expected outputs for semantic comparison.
            parent_step_id: Optional ID of a parent step if this action is part of a nested workflow.

        Returns:
            An AgentAction object representing the registered action.
        """
        action = AgentAction(
            workflow_id=self._workflow_id,
            agent_id=agent_id,
            action_type=action_type,
            inputs=inputs,
            expected_outputs=expected_outputs,
            parent_step_id=parent_step_id
        )
        logger.debug(f"Agent {agent_id} registered action '{action_type}' for workflow {self._workflow_id}.")
        return action

    def execute_agent_step(
        self,
        agent_id: str,
        action: AgentAction,
        agent_execution_func: Callable[[Dict[str, Any], WorkflowState], AgentOutput],
        semantic_comparison_context: Optional[Dict[str, Any]] = None,
        retries: int = 0
    ) -> AgentOutput:
        """
        Executes an agent's action within the workflow, handling state updates,
        semantic comparison, and reconciliation. This is the core method for agent interaction.
        
        Args:
            agent_id: The identifier of the agent performing the action.
            action: The AgentAction object describing the action to be executed.
            agent_execution_func: A callable that takes agent inputs (dict) and the current workflow state (WorkflowState)
                                  and returns an AgentOutput. This function encapsulates the agent's logic.
            semantic_comparison_context: Contextual information for semantic comparison (e.g., specific task goals
                                         that help an LLM-as-comparator).
            retries: Number of retries allowed for this step in case of failure or inconsistency.

        Returns:
            The final AgentOutput from the successful and verified execution.

        Raises:
            AgentExecutionError: If the agent fails to complete its action after all retries.
            StateConsistencyError: If a semantic inconsistency is detected and cannot be reconciled within retries.
            WorkflowError: For other workflow-level issues, e.g., invariant violations detected by the Verifier.
        """
        step_id = str(uuid4())  # Unique ID for this execution step

        # Record the pending action in the workflow state history.
        # Create a new WorkflowStep to represent this execution attempt.
        pending_step = WorkflowStep(
            step_id=step_id,
            agent_id=agent_id,
            action=action,
            output=None,  # Output is not yet known
            status="PENDING",
            timestamp=self._state_manager.get_current_timestamp()
        )
        
        # Add the pending step to the workflow history and create a checkpoint.
        # This makes the state before execution recoverable for rollbacks.
        try:
            self._current_workflow_state_id = self._state_manager.add_workflow_step(
                workflow_id=self._workflow_id,
                step=pending_step,
                checkpoint=True  # Auto-checkpoint before execution
            )
            logger.info(
                f"Workflow {self._workflow_id}: Agent {agent_id} initiated action '{action.action_type}' "
                f"(Step {step_id}). Checkpoint: {self._current_workflow_state_id}"
            )
        except Exception as e:
            raise WorkflowError(f"Failed to record pending step {step_id} in state manager: {e}") from e

        attempts = 0
        while attempts <= retries:
            current_state = self._get_current_workflow_state()  # Always get the latest state for current attempt
            
            try:
                logger.debug(
                    f"Workflow {self._workflow_id}: Executing agent '{agent_id}' for step '{step_id}', "
                    f"attempt {attempts+1}/{retries+1}. Current state ID: {self._current_workflow_state_id}"
                )
                
                # Execute the agent's function with current inputs and workflow state
                agent_output = agent_execution_func(action.inputs, current_workflow_state=current_state)

                if not isinstance(agent_output, AgentOutput):
                    raise TypeError(f"Agent execution function must return an instance of AgentOutput, got {type(agent_output)}")

                # Update the step with the actual output and mark as completed
                updated_step = pending_step.copy(
                    update={"output": agent_output, "status": "COMPLETED", "timestamp": self._state_manager.get_current_timestamp()}
                )
                
                # Update the workflow step in the state manager. This also checkpoints the new state.
                self._current_workflow_state_id = self._state_manager.update_workflow_step(
                    workflow_id=self._workflow_id,
                    step_id=step_id,
                    updated_step=updated_step,
                    checkpoint=True
                )
                logger.info(
                    f"Workflow {self._workflow_id}: Agent {agent_id} action '{action.action_type}' (Step {step_id}) "
                    f"completed successfully. New state ID: {self._current_workflow_state_id}"
                )

                # --- Semantic Comparison ---
                # This block runs if it's a retry, or if `action.expected_outputs` were provided for the initial run.
                # The verifier determines if the actual output is semantically equivalent to an expected one.
                is_semantically_consistent = True
                if attempts > 0:  # If it's a retry, compare current output with previous valid output or initial expected_outputs
                    previous_output = self._state_manager.get_step_output_before_attempt(self._workflow_id, step_id, attempts)
                    if previous_output:
                        logger.debug(f"Performing semantic comparison for retry {attempts+1} against previous output.")
                        is_semantically_consistent = self._verifier.verify_semantic_equivalence(
                            workflow_id=self._workflow_id,
                            step_id=step_id,
                            output1=agent_output,
                            output2=previous_output,
                            context=semantic_comparison_context
                        )
                    elif action.expected_outputs:
                        logger.debug(f"Performing semantic comparison for retry {attempts+1} against expected_outputs (from action).")
                        is_semantically_consistent = self._verifier.verify_semantic_equivalence(
                            workflow_id=self._workflow_id,
                            step_id=step_id,
                            output1=agent_output,
                            output2=AgentOutput(output_data=action.expected_outputs), # Wrap expected_outputs in AgentOutput for consistent comparison
                            context=semantic_comparison_context
                        )
                elif action.expected_outputs:  # For initial run, if expected_outputs are defined
                    logger.debug(f"Performing semantic comparison for initial run against expected_outputs (from action).")
                    is_semantically_consistent = self._verifier.verify_semantic_equivalence(
                        workflow_id=self._workflow_id,
                        step_id=step_id,
                        output1=agent_output,
                        output2=AgentOutput(output_data=action.expected_outputs),
                        context=semantic_comparison_context
                    )

                if not is_semantically_consistent:
                    logger.warning(
                        f"Workflow {self._workflow_id}: Semantic inconsistency detected for step {step_id} "
                        f"after {attempts+1} attempts."
                    )
                    if attempts < retries:
                        # Attempt reconciliation if retries are available
                        logger.info(f"Workflow {self._workflow_id}: Attempting reconciliation for step {step_id}.")
                        
                        # The reconciliation agent needs context about the inconsistency
                        reconciliation_strategy = self._reconciliation_agent.propose_reconciliation(
                            workflow_id=self._workflow_id,
                            inconsistent_step_id=step_id,
                            current_state=self._get_current_workflow_state(),  # Pass current complete state
                            agent_action=action,
                            actual_output=agent_output,
                            expected_output=AgentOutput(output_data=action.expected_outputs) if action.expected_outputs else None
                        )

                        # Rollback to the state *before* this attempt for a clean retry
                        self._state_manager.rollback_to_last_checkpoint(self._workflow_id)
                        self._current_workflow_state_id = self._state_manager.get_latest_state_id(self._workflow_id)
                        logger.info(
                            f"Workflow {self._workflow_id}: Rolled back for reconciliation. "
                            f"Retrying with strategy: {reconciliation_strategy.type}. New current state ID: {self._current_workflow_state_id}"
                        )
                        # Apply strategy (e.g., modify action inputs for re-prompting)
                        action = self._apply_reconciliation_strategy(action, reconciliation_strategy)
                        attempts += 1
                        continue  # Retry the loop with modified action
                    else:
                        raise StateConsistencyError(
                            f"Workflow {self._workflow_id}: Semantic inconsistency after {retries+1} attempts for step {step_id}. "
                            "No more retries or reconciliation possible."
                        )

                # --- Invariant Verification ---
                # The verifier checks against formal rules and invariants for the new state.
                is_valid = self._verifier.verify_invariants(
                    workflow_id=self._workflow_id,
                    current_state=self._get_current_workflow_state(),  # Pass the most recent state (after agent_output)
                    new_step=updated_step  # The step that just completed
                )
                if not is_valid:
                    error_message = f"Workflow {self._workflow_id}: Rule/invariant violation detected for step {step_id}."
                    logger.error(error_message)
                    # For formal rule violations, typically an error is raised to prevent propagating invalid states.
                    raise WorkflowError(error_message)

                return agent_output  # Successfully completed and verified

            except Exception as e:
                error_type = type(e).__name__
                logger.error(
                    f"Workflow {self._workflow_id}: Agent {agent_id} action '{action.action_type}' (Step {step_id}) "
                    f"failed on attempt {attempts+1} ({error_type}): {e}"
                )
                
                if attempts == retries:
                    # If this was the last attempt, mark step as FAILED and re-raise
                    failed_step = pending_step.copy(
                        update={"output": None, "status": "FAILED", "error_message": str(e), "timestamp": self._state_manager.get_current_timestamp()}
                    )
                    self._current_workflow_state_id = self._state_manager.update_workflow_step(
                        workflow_id=self._workflow_id,
                        step_id=step_id,
                        updated_step=failed_step,
                        checkpoint=True
                    )
                    raise AgentExecutionError(
                        f"Agent {agent_id} failed to complete action {action.action_type} after {retries+1} attempts "
                        f"for workflow {self._workflow_id} (Step {step_id})."
                    ) from e
                
                # Rollback to the state *before* this failed attempt for the next retry
                self._state_manager.rollback_to_last_checkpoint(self._workflow_id)
                self._current_workflow_state_id = self._state_manager.get_latest_state_id(self._workflow_id)
                logger.warning(
                    f"Workflow {self._workflow_id}: Rolled back state for retry of step {step_id}. "
                    f"New current state ID: {self._current_workflow_state_id}"
                )
                attempts += 1
                # Continue the while loop for the next attempt

        # This line should ideally not be reached as an exception is raised if retries are exhausted.
        raise AgentExecutionError(f"Workflow {self._workflow_id}: Unexpected execution path for step {step_id}.")

    def _apply_reconciliation_strategy(self, action: AgentAction, strategy: ReconciliationStrategy) -> AgentAction:
        """
        Applies a reconciliation strategy to modify the agent's action for a retry.
        This method needs to be expanded as more strategies are defined in `reconciliation/strategies.py`.
        It creates a modified `AgentAction` object based on the strategy.
        """
        logger.info(f"Applying reconciliation strategy '{strategy.type}' to agent action for workflow {self._workflow_id}.")
        
        # Create a mutable copy of the action to modify
        modified_action = action.copy(deep=True) # Ensure deep copy for mutable fields like inputs

        if strategy.type == "CONTEXTUAL_REPROMPTING":
            new_prompt_context = strategy.payload.get("new_prompt_context", {})
            logger.debug(f"CONTEXTUAL_REPROMPTING: Adding/updating inputs with {new_prompt_context}")
            modified_action.inputs.update(new_prompt_context)
            
        elif strategy.type == "OUTPUT_REFINEMENT":
            refinement_guidance = strategy.payload.get("refinement_guidance", {})
            logger.debug(f"OUTPUT_REFINEMENT: Adding/updating inputs with refinement guidance {refinement_guidance}")
            # For output refinement, often the prompt inputs are modified to guide the LLM
            # to produce a more refined output in the next attempt.
            modified_action.inputs.update(refinement_guidance)
            
        elif strategy.type == "STATE_ADJUSTMENT":
            state_patch = strategy.payload.get("state_patch", {})
            logger.warning(
                f"STATE_ADJUSTMENT strategy received for modifying agent action. "
                f"This typically implies the ReconciliationAgent directly updated StateManager "
                f"before this method was called. No direct modification to AgentAction here, "
                f"but the agent should read the updated state when it re-executes."
            )
            # The agent_execution_func receives the latest current_workflow_state, so it will see the adjustment.
            
        elif strategy.type == "ROLLBACK_AND_RETRY":
            logger.debug("ROLLBACK_AND_RETRY strategy implies simply retrying after rollback, no action modification needed for the action itself.")
            # The workflow manager already handled the rollback and is about to retry.
            # No modification to the action itself is needed unless the strategy also
            # specified a modification alongside the rollback.

        else:
            logger.warning(f"Unknown reconciliation strategy type: {strategy.type}. No specific modification applied to action.")
        
        return modified_action

    def get_workflow_state(self) -> WorkflowState:
        """Retrieves the current complete workflow state."""
        return self._get_current_workflow_state()

    def get_workflow_history(self) -> List[WorkflowStep]:
        """Retrieves the history of workflow steps from the current state."""
        state = self._get_current_workflow_state()
        return state.history

    def checkpoint(self, description: Optional[str] = None) -> StateIdentifier:
        """
        Requests the state manager to create an explicit checkpoint of the current workflow state.
        This can be used to mark significant points in a workflow for later rollback.

        Args:
            description: An optional description for the checkpoint.

        Returns:
            The identifier of the newly created checkpoint.
        """
        logger.info(f"Workflow {self._workflow_id}: Requesting explicit checkpoint: '{description if description else 'No description'}'.")
        self._current_workflow_state_id = self._state_manager.create_checkpoint(self._workflow_id, description)
        logger.info(f"Workflow {self._workflow_id}: Created checkpoint with ID: {self._current_workflow_state_id}")
        return self._current_workflow_state_id

    def rollback_to_checkpoint(self, checkpoint_id: StateIdentifier) -> None:
        """
        Initiates a rollback of the workflow state to a specific, previously recorded checkpoint.
        All state changes after this checkpoint will be undone.

        Args:
            checkpoint_id: The identifier of the checkpoint to roll back to.

        Raises:
            WorkflowError: If the rollback operation fails.
        """
        logger.warning(f"Workflow {self._workflow_id}: Initiating rollback to checkpoint {checkpoint_id}.")
        try:
            self._state_manager.rollback_to_checkpoint(self._workflow_id, checkpoint_id)
            self._current_workflow_state_id = checkpoint_id  # Update current state ID to the rolled back one
            logger.info(f"Workflow {self._workflow_id}: Successfully rolled back to checkpoint {checkpoint_id}. "
                        f"Current state is now {self._current_workflow_state_id}.")
        except Exception as e:
            raise WorkflowError(f"Failed to rollback workflow {self._workflow_id} to checkpoint {checkpoint_id}: {e}") from e

    def get_workflow_id(self) -> str:
        """Returns the unique identifier of the workflow managed by this instance."""
        return self._workflow_id

```