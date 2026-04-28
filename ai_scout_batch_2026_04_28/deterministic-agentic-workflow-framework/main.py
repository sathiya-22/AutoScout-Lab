```python
# FILE NAME: main.py

import logging
import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import numpy as np
from abc import ABC, abstractmethod

# --- STUBS FOR PROJECT CONTEXT (simulate separate files/modules) ---

# config.py
class Config:
    LLM_MODEL = "gpt-4o-mini"
    LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Placeholder API key
    LOG_LEVEL = "INFO"
    SEMANTIC_SIMILARITY_THRESHOLD = 0.8
    STATE_STORAGE_PATH = "workflow_state.json"

config = Config()

# utils/logging_config.py
def setup_logging():
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).info("Logging setup complete.")

# utils/llm_connector.py
class LLMConnector:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.request_count = 0

    def chat_completion(self, prompt: str, temperature: float = 0.7) -> str:
        self.request_count += 1
        self.logger.info(f"LLM call {self.request_count} to {self.model_name} with prompt: {prompt[:80]}...")
        time.sleep(0.1)  # Simulate network latency

        # Simulate non-deterministic LLM output to demonstrate semantic idempotency problem
        if "process data" in prompt.lower():
            if self.request_count % 3 == 1:
                return "Processed data successfully. Key metrics: 123, Status: Completed. (Initial Attempt)"
            elif self.request_count % 3 == 2:
                # Semantically equivalent but structurally different
                return "Data processing finished. Results obtained: 123. Current State: Done. (Retry Attempt)"
            else:
                # Output after a 're-prompt' - hopefully closer to desired format or semantically correct
                return "Final Data Report: Key metrics are 123, and the overall status is Completed. (Reconciled Output)"
        elif "evaluate outputs" in prompt.lower():
            # LLM-as-Comparator
            if "key metrics: 123" in prompt.lower() and "results obtained: 123" in prompt.lower():
                return "YES. Both outputs convey the same core information about metrics and status being 123 and completed/done."
            return "NO. The outputs have significant differences in key information."
        return f"LLM output for '{prompt[:50]}...'"

# utils/serialization.py
def serialize_state(state: Dict[str, Any]) -> str:
    """Basic JSON serialization for demonstration."""
    return json.dumps(state, indent=2)

def deserialize_state(data: str) -> Dict[str, Any]:
    """Basic JSON deserialization."""
    return json.loads(data)

# framework/exceptions.py
class WorkflowException(Exception):
    """Base exception for workflow errors."""
    pass

class StateConsistencyError(WorkflowException):
    """Raised when an inconsistency in state or agent output is detected."""
    pass

class AgentExecutionError(WorkflowException):
    """Raised when an agent fails to complete its task."""
    pass

# framework/action_schemas.py
class BaseAgentInput(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual data for the agent's task.")

class BaseAgentOutput(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task.")
    raw_output: str = Field(..., description="Raw output from the LLM or agent process.")
    parsed_output: Optional[Dict[str, Any]] = Field(None, description="Structured/parsed output from the agent.")
    semantic_summary: Optional[str] = Field(None, description="A high-level semantic summary of the output.")

class TaskOutcome(BaseModel):
    status: Literal["PENDING", "COMPLETED", "FAILED", "RECONCILED", "ROLLED_BACK"]
    message: Optional[str] = None
    output: Optional[BaseAgentOutput] = None

class SemanticAction(BaseModel):
    action_id: str = Field(..., description="Unique ID for this specific action.")
    agent_id: str = Field(..., description="ID of the agent performing the action.")
    input_data: BaseAgentInput = Field(..., description="Input provided to the agent.")
    proposed_output: Optional[BaseAgentOutput] = Field(None, description="The output proposed by the agent.")
    actual_outcome: Optional[TaskOutcome] = Field(None, description="The final outcome of the action.")
    timestamp: float = Field(default_factory=time.time)
    retry_count: int = Field(0, description="Number of retries for this specific action.")

class AgentState(BaseModel):
    agent_id: str
    current_status: str
    last_action_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    workflow_id: str
    current_step: int = 0
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    action_history: List[SemanticAction] = Field(default_factory=list)
    checkpoints: Dict[int, Dict[str, Any]] = Field(default_factory=dict) # Stores serialized states

# framework/state_manager.py
class StateManager:
    def __init__(self, storage_path: str = config.STATE_STORAGE_PATH):
        self.storage_path = storage_path
        self._current_workflow_state: Optional[WorkflowState] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_state()

    def _load_state(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = deserialize_state(f.read())
                self._current_workflow_state = WorkflowState.parse_obj(data)
            self.logger.info(f"Loaded state from {self.storage_path}")
        except FileNotFoundError:
            self.logger.warning("No existing state file found. Starting fresh.")
            self._current_workflow_state = None
        except Exception as e:
            self.logger.error(f"Error loading state: {e}. Starting fresh.")
            self._current_workflow_state = None

    def _save_state(self):
        if self._current_workflow_state:
            with open(self.storage_path, 'w') as f:
                f.write(serialize_state(self._current_workflow_state.dict()))
            self.logger.debug(f"State saved to {self.storage_path}")

    def initialize_workflow(self, workflow_id: str) -> WorkflowState:
        if self._current_workflow_state and self._current_workflow_state.workflow_id == workflow_id:
            self.logger.info(f"Workflow {workflow_id} already initialized. Returning current state.")
            return self._current_workflow_state

        self._current_workflow_state = WorkflowState(workflow_id=workflow_id)
        self._save_state()
        self.logger.info(f"Initialized new workflow: {workflow_id}")
        return self._current_workflow_state

    def get_workflow_state(self) -> Optional[WorkflowState]:
        return self._current_workflow_state

    def update_agent_state(self, agent_id: str, status: str, context: Optional[Dict[str, Any]] = None, last_action_id: Optional[str] = None):
        if not self._current_workflow_state:
            raise WorkflowException("Workflow not initialized.")
        if agent_id not in self._current_workflow_state.agent_states:
            self._current_workflow_state.agent_states[agent_id] = AgentState(agent_id=agent_id, current_status=status, context=context or {})
        else:
            agent_state = self._current_workflow_state.agent_states[agent_id]
            agent_state.current_status = status
            if context:
                agent_state.context.update(context)
            if last_action_id:
                agent_state.last_action_id = last_action_id
        self._save_state()
        self.logger.debug(f"Agent '{agent_id}' state updated to '{status}'.")

    def record_action(self, action: SemanticAction):
        if not self._current_workflow_state:
            raise WorkflowException("Workflow not initialized.")
        # Find if this action is an update to an existing one (e.g., updating actual_outcome)
        for i, existing_action in enumerate(self._current_workflow_state.action_history):
            if existing_action.action_id == action.action_id:
                self._current_workflow_state.action_history[i] = action
                self.logger.debug(f"Action '{action.action_id}' updated for agent '{action.agent_id}'.")
                self._save_state()
                return

        self._current_workflow_state.action_history.append(action)
        self._save_state()
        self.logger.info(f"Action '{action.action_id}' recorded for agent '{action.agent_id}'.")

    def get_last_completed_action_for_agent_and_task(self, agent_id: str, task_id: str) -> Optional[SemanticAction]:
        if not self._current_workflow_state:
            return None
        for action in reversed(self._current_workflow_state.action_history):
            if action.agent_id == agent_id and action.input_data.task_id == task_id and action.actual_outcome and action.actual_outcome.status == "COMPLETED":
                return action
        return None

    def create_checkpoint(self, step: int):
        if not self._current_workflow_state:
            raise WorkflowException("Workflow not initialized.")
        self._current_workflow_state.checkpoints[step] = self._current_workflow_state.dict()
        self._save_state()
        self.logger.info(f"Checkpoint created at step {step}.")

    def rollback_to_checkpoint(self, step: int):
        if not self._current_workflow_state:
            raise WorkflowException("Workflow not initialized.")
        if step not in self._current_workflow_state.checkpoints:
            raise WorkflowException(f"Checkpoint at step {step} not found.")

        self._current_workflow_state = WorkflowState.parse_obj(self._current_workflow_state.checkpoints[step])
        self._save_state()
        self.logger.warning(f"Rolled back workflow to checkpoint at step {step}.")
        return self._current_workflow_state

# semantic_comparison/evaluation_metrics.py
class ComparisonResult(BaseModel):
    is_equivalent: bool
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)

# semantic_comparison/comparators.py
class SemanticComparator:
    def __init__(self, llm_connector: LLMConnector, threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD):
        self.llm_connector = llm_connector
        self.threshold = threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_embedding(self, text: str) -> List[float]:
        # Simulate embedding generation: returns a consistent random vector for the same text
        # In a real system, this would use a real embedding model (e.g., OpenAI, Sentence-BERT)
        seed = hash(text) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        return rng.rand(10).tolist() # Simple random vector

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def compare_semantic_equivalence(self, output1: BaseAgentOutput, output2: BaseAgentOutput) -> ComparisonResult:
        self.logger.info("Performing semantic comparison...")

        summary1 = output1.semantic_summary or output1.raw_output
        summary2 = output2.semantic_summary or output2.raw_output

        # Strategy 1: Embedding-based similarity
        vec1 = self._get_embedding(summary1)
        vec2 = self._get_embedding(summary2)
        embedding_score = self._cosine_similarity(vec1, vec2)
        is_equivalent_embedding = embedding_score >= self.threshold

        # Strategy 2: LLM-as-Comparator
        llm_prompt = (
            f"Given two outputs for the same task, evaluate if they are semantically equivalent or convey the same core information, "
            f"even if phrased differently. Respond with 'YES' or 'NO' followed by a brief explanation.\n\n"
            f"Output 1:\n'{summary1}'\n\n"
            f"Output 2:\n'{summary2}'"
        )
        llm_response = self.llm_connector.chat_completion(llm_prompt, temperature=0.1)
        is_equivalent_llm = "yes" in llm_response.lower()

        # Combine strategies
        is_equivalent = is_equivalent_embedding and is_equivalent_llm
        final_score = (embedding_score + (1.0 if is_equivalent_llm else 0.0)) / 2.0

        self.logger.info(f"Comparison Result: Equivalent={is_equivalent}, Score={final_score:.2f}")
        return ComparisonResult(
            is_equivalent=is_equivalent,
            score=final_score,
            details={
                "embedding_score": embedding_score,
                "llm_comparison_response": llm_response
            }
        )

# verification/rule_engine.py
class RuleEngine:
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_rule(self, rule_name: str, check_func):
        self.rules.append({"name": rule_name, "check": check_func})
        self.logger.info(f"Rule '{rule_name}' added.")

    def evaluate(self, agent_output: BaseAgentOutput) -> List[str]:
        violations = []
        parsed = agent_output.parsed_output
        if not parsed:
            violations.append("Output could not be parsed or parsed_output is empty for rule evaluation.")
            return violations

        for rule in self.rules:
            try:
                if not rule["check"](parsed):
                    violations.append(f"Rule '{rule['name']}' violated.")
            except Exception as e:
                violations.append(f"Error evaluating rule '{rule['name']}': {e}")
        self.logger.debug(f"Evaluated output against rules. Violations: {violations}")
        return violations

# verification/verifier.py
class Verifier:
    def __init__(self, rule_engine: RuleEngine, semantic_comparator: SemanticComparator):
        self.rule_engine = rule_engine
        self.semantic_comparator = semantic_comparator
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_output(self, current_output: BaseAgentOutput, previous_output: Optional[BaseAgentOutput] = None) -> List[str]:
        violations = []

        # 1. Rule-based validation
        rule_violations = self.rule_engine.evaluate(current_output)
        if rule_violations:
            violations.extend(rule_violations)
            self.logger.warning(f"Rule violations detected: {rule_violations}")

        # 2. Semantic comparison if a previous output exists (e.g., after a retry)
        if previous_output:
            self.logger.info(f"Comparing current output (task {current_output.task_id}) with previous output.")
            comparison_result = self.semantic_comparator.compare_semantic_equivalence(previous_output, current_output)
            if not comparison_result.is_equivalent:
                violations.append(f"Semantic divergence detected between current and previous output. Score: {comparison_result.score:.2f}, Details: {comparison_result.details.get('llm_comparison_response', '')}")
                self.logger.warning(f"Semantic divergence detected: {comparison_result.details}")

        return violations

# agents/base_agent.py
class BaseAgent(ABC):
    def __init__(self, agent_id: str, llm_connector: LLMConnector):
        self.agent_id = agent_id
        self.llm_connector = llm_connector
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, agent_input: BaseAgentInput) -> BaseAgentOutput:
        pass

# agents/example_data_processor_agent.py
class ExampleDataProcessorAgent(BaseAgent):
    def __init__(self, agent_id: str, llm_connector: LLMConnector):
        super().__init__(agent_id, llm_connector)
        self.logger.info(f"{self.agent_id} initialized.")

    def execute(self, agent_input: BaseAgentInput) -> BaseAgentOutput:
        self.logger.info(f"{self.agent_id} executing task {agent_input.task_id} with context: {agent_input.context}")
        
        data_to_process = agent_input.context.get('data', 'No data provided')
        prompt_modifier = agent_input.context.get('explicit_constraint', '')
        
        prompt = f"Given the following data: '{data_to_process}', process it and extract 'Key metrics' and 'Status'. {prompt_modifier} Output should clearly state 'Key metrics:' and 'Status:'."
        
        try:
            raw_output = self.llm_connector.chat_completion(prompt)
            
            # Simulate robust parsing based on expected keywords
            parsed_output = {}
            if "Key metrics:" in raw_output:
                metrics_part = raw_output.split("Key metrics:")[1].split(",")[0].split(".")[0].strip()
                parsed_output["metrics"] = metrics_part
            if "Status:" in raw_output:
                status_part = raw_output.split("Status:")[1].split(".")[0].strip()
                parsed_output["status"] = status_part
            elif "State:" in raw_output and "status" not in parsed_output: # Handle LLM's non-deterministic synonyms
                 status_part = raw_output.split("State:")[1].split(".")[0].strip()
                 parsed_output["status"] = status_part

            semantic_summary = f"Processed data. Status is {parsed_output.get('status', 'unknown')} and metrics are {parsed_output.get('metrics', 'N/A')}."

            return BaseAgentOutput(
                task_id=agent_input.task_id,
                raw_output=raw_output,
                parsed_output=parsed_output,
                semantic_summary=semantic_summary
            )
        except Exception as e:
            self.logger.error(f"Error in {self.agent_id} during execution: {e}")
            raise AgentExecutionError(f"Agent {self.agent_id} failed: {e}")

# reconciliation/strategies.py
class ReconciliationStrategy(ABC):
    @abstractmethod
    def apply(self, workflow_manager, inconsistency_details: Dict[str, Any]) -> TaskOutcome:
        pass

class RepromptStrategy(ReconciliationStrategy):
    def apply(self, workflow_manager, inconsistency_details: Dict[str, Any]) -> TaskOutcome:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.warning(f"Applying RepromptStrategy for inconsistency: {inconsistency_details.get('reason', 'N/A')}")
        
        agent_id = inconsistency_details['agent_id']
        task_id = inconsistency_details['task_id']
        last_action: SemanticAction = inconsistency_details['last_action']
        retry_count = inconsistency_details.get('retry_count', 0) + 1 # Increment retry count for this new attempt
        
        original_context = last_action.input_data.context.copy()
        
        # Create a new, refined prompt context with more explicit instructions
        refined_context = original_context
        refined_context['explicit_constraint'] = (
            f"Ensure the output format clearly separates 'Key metrics:' and 'Status:'. "
            f"Be precise with these exact keywords. Do not use synonyms like 'Results' or 'State'. "
            f"The previous attempt failed due to: {inconsistency_details.get('reason', 'unspecified parsing issue')}."
        )
        
        new_agent_input = BaseAgentInput(
            task_id=task_id,
            context=refined_context
        )
        
        self.logger.info(f"Re-prompting agent {agent_id} for task {task_id} with refined context (retry {retry_count}).")
        try:
            # Recursively call execute_agent_action with new input and incremented retry_count
            new_output = workflow_manager.execute_agent_action(
                agent_id=agent_id,
                agent_input=new_agent_input,
                is_retry=True,
                retry_count=retry_count
            )
            return TaskOutcome(status="RECONCILED", message="Re-prompting successful", output=new_output)
        except AgentExecutionError as e:
            self.logger.error(f"Re-prompting failed for agent {agent_id}: {e}")
            return TaskOutcome(status="FAILED", message=f"Re-prompting failed: {e}")

class RollbackAndRetryStrategy(ReconciliationStrategy):
    def apply(self, workflow_manager, inconsistency_details: Dict[str, Any]) -> TaskOutcome:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.warning(f"Applying RollbackAndRetryStrategy for inconsistency: {inconsistency_details.get('reason', 'N/A')}")
        
        agent_id = inconsistency_details['agent_id']
        task_id = inconsistency_details['task_id']
        
        current_workflow_step = workflow_manager.state_manager.get_workflow_state().current_step
        rollback_step = current_workflow_step # Rollback to the previous logical step
        if current_workflow_step > 0:
             rollback_step = current_workflow_step - 1
        
        self.logger.info(f"Rolling back workflow to checkpoint at step {rollback_step} and re-attempting agent {agent_id} for task {task_id}.")
        
        try:
            workflow_manager.state_manager.rollback_to_checkpoint(rollback_step)
            # Re-fetch the original input from the rolled-back state or from the inconsistency_details
            original_input = inconsistency_details['last_action'].input_data
            
            # Reset retry count upon rollback as it's a fresh attempt from a stable state
            new_output = workflow_manager.execute_agent_action(
                agent_id=agent_id,
                agent_input=original_input,
                is_retry=True,
                retry_count=0 
            )
            return TaskOutcome(status="RECONCILED", message="Rollback and retry successful", output=new_output)
        except Exception as e:
            self.logger.error(f"Rollback and retry failed: {e}")
            return TaskOutcome(status="FAILED", message=f"Rollback and retry failed: {e}")

# reconciliation/reconciliation_agent.py
class ReconciliationAgent:
    def __init__(self, workflow_manager, verifier, comparators):
        self.workflow_manager = workflow_manager
        self.verifier = verifier
        self.comparators = comparators # Not directly used here, but could be for deeper analysis
        self.strategies = {
            "reprompt": RepromptStrategy(),
            "rollback_and_retry": RollbackAndRetryStrategy()
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    def reconcile_inconsistency(self, inconsistency_details: Dict[str, Any]) -> TaskOutcome:
        self.logger.warning(f"ReconciliationAgent initiated for inconsistency: {inconsistency_details.get('reason', 'N/A')}")
        
        strategy_name = "reprompt"
        current_retry_count = inconsistency_details.get('retry_count', 0)
        
        # Simple strategy selection for prototype:
        # Try reprompting first. If that already happened, try rollback.
        if "Semantic divergence" in inconsistency_details.get('reason', '') or "Rule" in inconsistency_details.get('reason', ''):
            if current_retry_count == 0:
                strategy_name = "reprompt"
            else: # If first reprompt failed or didn't fully resolve, try rollback
                strategy_name = "rollback_and_retry"
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            self.logger.error(f"No reconciliation strategy found for '{strategy_name}'. Failing reconciliation.")
            return TaskOutcome(status="FAILED", message=f"No strategy for '{strategy_name}'.")

        self.logger.info(f"Applying reconciliation strategy: {strategy_name}")
        return strategy.apply(self.workflow_manager, inconsistency_details)

# framework/workflow_manager.py
class WorkflowManager:
    def __init__(self, state_manager: StateManager, verifier: Verifier, llm_connector: LLMConnector):
        self.state_manager = state_manager
        self.verifier = verifier
        self.llm_connector = llm_connector
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reconciliation_agent: Optional[ReconciliationAgent] = None

    def set_reconciliation_agent(self, reconciliation_agent: ReconciliationAgent):
        self.reconciliation_agent = reconciliation_agent

    def register_agent(self, agent: BaseAgent):
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent '{agent.agent_id}' already registered. Overwriting.")
        self.agents[agent.agent_id] = agent
        self.state_manager.update_agent_state(agent.agent_id, "REGISTERED")
        self.logger.info(f"Agent '{agent.agent_id}' registered.")

    def start_workflow(self, workflow_id: str):
        self.state_manager.initialize_workflow(workflow_id)
        self.state_manager.create_checkpoint(0) # Initial checkpoint
        self.logger.info(f"Workflow '{workflow_id}' started.")

    def execute_agent_action(self, agent_id: str, agent_input: BaseAgentInput, is_retry: bool = False, retry_count: int = 0, max_reconciliation_attempts: int = 2) -> BaseAgentOutput:
        if agent_id not in self.agents:
            raise WorkflowException(f"Agent '{agent_id}' not registered.")

        agent = self.agents[agent_id]
        current_workflow_state = self.state_manager.get_workflow_state()
        if not current_workflow_state:
            raise WorkflowException("Workflow not initialized. Call start_workflow first.")

        action_id = f"{agent_id}-{agent_input.task_id}-{uuid.uuid4().hex[:8]}"
        self.state_manager.update_agent_state(agent_id, "RUNNING", last_action_id=action_id)

        proposed_output = None
        try:
            proposed_output = agent.execute(agent_input)
            
            semantic_action = SemanticAction(
                action_id=action_id,
                agent_id=agent_id,
                input_data=agent_input,
                proposed_output=proposed_output,
                actual_outcome=TaskOutcome(status="PENDING", message="Output generated, awaiting verification"),
                retry_count=retry_count
            )
            self.state_manager.record_action(semantic_action)

            previous_output = None
            if is_retry:
                # For comparison, get the last *successfully completed* action's output for this agent and task.
                # If no completed action, perhaps use the last PENDING/FAILED action's proposed_output for comparison.
                last_successful_action = self.state_manager.get_last_completed_action_for_agent_and_task(agent_id, agent_input.task_id)
                if last_successful_action:
                    previous_output = last_successful_action.proposed_output
                else: # Fallback: if no successful history, use the immediate prior failed attempt for comparison
                    for action in reversed(self.state_manager.get_workflow_state().action_history):
                        if action.agent_id == agent_id and action.input_data.task_id == agent_input.task_id and action.action_id != action_id:
                            previous_output = action.proposed_output
                            break
            
            violations = self.verifier.validate_output(proposed_output, previous_output)

            if violations:
                self.logger.warning(f"Agent '{agent_id}' output for task '{agent_input.task_id}' failed verification. Violations: {violations}")
                
                # Update the recorded action's outcome to FAILED
                semantic_action.actual_outcome = TaskOutcome(status="FAILED", message="Verification failed", output=proposed_output)
                self.state_manager.record_action(semantic_action) # Re-record to update outcome

                if self.reconciliation_agent and retry_count < max_reconciliation_attempts:
                    self.logger.info(f"Attempting reconciliation for agent '{agent_id}' (Reconciliation attempt {retry_count + 1}/{max_reconciliation_attempts}).")
                    inconsistency_details = {
                        "agent_id": agent_id,
                        "task_id": agent_input.task_id,
                        "reason": ", ".join(violations),
                        "current_output": proposed_output,
                        "previous_output": previous_output,
                        "last_action": semantic_action,
                        "retry_count": retry_count
                    }
                    reconciliation_outcome = self.reconciliation_agent.reconcile_inconsistency(inconsistency_details)
                    
                    if reconciliation_outcome.status == "RECONCILED" and reconciliation_outcome.output:
                        self.state_manager.update_agent_state(agent_id, "COMPLETED", last_action_id=action_id)
                        # No increment of current_step here; reconciliation resolves the *current* step's failure.
                        self.logger.info(f"Agent '{agent_id}' task '{agent_input.task_id}' successfully reconciled.")
                        return reconciliation_outcome.output
                    else:
                        self.logger.error(f"Reconciliation failed or did not yield a new output for '{agent_id}'. Reconciliation Outcome: {reconciliation_outcome.message}")
                        raise StateConsistencyError(f"Agent '{agent_id}' output inconsistent and reconciliation failed after {retry_count+1} attempts: {reconciliation_outcome.message}")
                else:
                    raise StateConsistencyError(f"Agent '{agent_id}' output inconsistent and max reconciliation attempts ({max_reconciliation_attempts}) reached: {', '.join(violations)}")
            else:
                self.state_manager.update_agent_state(agent_id, "COMPLETED", last_action_id=action_id)
                semantic_action.actual_outcome = TaskOutcome(status="COMPLETED", message="Verification successful", output=proposed_output)
                self.state_manager.record_action(semantic_action) # Re-record to update outcome
                current_workflow_state.current_step += 1 # Increment workflow step ONLY on successful, verified completion
                self.state_manager.create_checkpoint(current_workflow_state.current_step)
                self.logger.info(f"Agent '{agent_id}' task '{agent_input.task_id}' completed and verified, advancing workflow to step {current_workflow_state.current_step}.")
                return proposed_output

        except (AgentExecutionError, StateConsistencyError) as e:
            self.state_manager.update_agent_state(agent_id, "FAILED", last_action_id=action_id)
            semantic_action = SemanticAction(
                action_id=action_id,
                agent_id=agent_id,
                input_data=agent_input,
                actual_outcome=TaskOutcome(status="FAILED", message=str(e)),
                proposed_output=proposed_output, # Include proposed output even on failure
                retry_count=retry_count
            )
            self.state_manager.record_action(semantic_action)
            self.logger.error(f"Agent '{agent_id}' execution failed for task '{agent_input.task_id}': {e}")
            raise
        except Exception as e:
            self.state_manager.update_agent_state(agent_id, "FAILED", last_action_id=action_id)
            semantic_action = SemanticAction(
                action_id=action_id,
                agent_id=agent_id,
                input_data=agent_input,
                actual_outcome=TaskOutcome(status="FAILED", message=f"Unexpected error: {e}"),
                proposed_output=proposed_output,
                retry_count=retry_count
            )
            self.state_manager.record_action(semantic_action)
            self.logger.critical(f"Unhandled error in workflow for agent '{agent_id}': {e}", exc_info=True)
            raise WorkflowException(f"Unhandled error during agent execution: {e}")

# --- END OF STUBS ---

# main.py content starts here
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Starting Deterministic Agentic Workflow Framework Prototype...")

    workflow_id = "data_processing_workflow_001"
    
    # 1. Initialize core components
    llm_connector = LLMConnector(api_key=config.LLM_API_KEY, model_name=config.LLM_MODEL)
    state_manager = StateManager()
    semantic_comparator = SemanticComparator(llm_connector=llm_connector)
    
    rule_engine = RuleEngine()
    # Example rule: Parsed output must contain 'metrics' and 'status' (required fields)
    rule_engine.add_rule("HasMetricsAndStatus", lambda parsed_output: "metrics" in parsed_output and "status" in parsed_output)
    # Example rule: Metrics must be numeric
    rule_engine.add_rule("MetricsIsNumeric", lambda parsed_output: parsed_output.get("metrics", "").replace('.', '', 1).isdigit())
    
    verifier = Verifier(rule_engine=rule_engine, semantic_comparator=semantic_comparator)
    workflow_manager = WorkflowManager(state_manager=state_manager, verifier=verifier, llm_connector=llm_connector)
    
    reconciliation_agent = ReconciliationAgent(
        workflow_manager=workflow_manager,
        verifier=verifier,
        comparators=semantic_comparator
    )
    workflow_manager.set_reconciliation_agent(reconciliation_agent)

    # 2. Register Agents
    data_processor_agent = ExampleDataProcessorAgent(agent_id="DataProcessorAgent", llm_connector=llm_connector)
    workflow_manager.register_agent(data_processor_agent)

    # 3. Start Workflow
    try:
        workflow_manager.start_workflow(workflow_id)

        # 4. Agent performs initial task (First run: LLMConnector.request_count % 3 == 1)
        logger.info("\n--- STEP 1: Initial Agent Execution ---")
        initial_input = BaseAgentInput(
            task_id="process_sales_data_Q1",
            context={"data": "Sales figures for Q1: Revenue 1.2M, Expenses 0.8M. Focus on net profit calculation and status.", "data_prompt": "Process sales data and extract key metrics and status."}
        )
        first_output = workflow_manager.execute_agent_action(
            agent_id="DataProcessorAgent",
            agent_input=initial_input
        )
        logger.info(f"Initial Agent Output: {first_output.raw_output}")
        logger.info(f"Initial Parsed Output: {first_output.parsed_output}")
        logger.info(f"Initial Semantic Summary: {first_output.semantic_summary}")
        
        # 5. Simulate a re-execution/retry for the SAME logical task
        # This will trigger LLMConnector.request_count % 3 == 2, providing a semantically similar but structurally different output.
        # This simulates the "semantic idempotency" problem.
        logger.info("\n--- STEP 2: Simulating Re-execution/Retry of the Same Task (triggering inconsistency) ---")
        
        try:
            # We call execute_agent_action with is_retry=True, and retry_count=0 (for this logical attempt cycle)
            second_attempt_output = workflow_manager.execute_agent_action(
                agent_id="DataProcessorAgent",
                agent_input=initial_input, # Use the same input to represent a re-try of the same task
                is_retry=True,
                retry_count=0 # First attempt at reconciliation for this failure
            )
            logger.info(f"Second (reconciled) Agent Output: {second_attempt_output.raw_output}")
            logger.info(f"Second (reconciled) Parsed Output: {second_attempt_output.parsed_output}")
            logger.info(f"Second (reconciled) Semantic Summary: {second_attempt_output.semantic_summary}")

        except StateConsistencyError as e:
            logger.error(f"Workflow ended due to unresolvable inconsistency: {e}")
        
        # 6. Display final state
        logger.info("\n--- Final Workflow State ---")
        final_state = state_manager.get_workflow_state()
        logger.info(f"Current Step: {final_state.current_step}")
        logger.info(f"Agent States: {final_state.agent_states}")
        logger.info(f"Action History Length: {len(final_state.action_history)}")
        logger.info(f"Checkpoints: {list(final_state.checkpoints.keys())}")

    except WorkflowException as e:
        logger.error(f"Workflow encountered a critical error: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during workflow execution: {e}", exc_info=True)

    logger.info("Deterministic Agentic Workflow Framework Prototype finished.")

```