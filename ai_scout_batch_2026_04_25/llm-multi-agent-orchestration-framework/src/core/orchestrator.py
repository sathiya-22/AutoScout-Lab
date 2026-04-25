```python
import asyncio
import uuid
from enum import Enum, auto
from typing import Dict, Any, Optional, List

# Assuming relative imports within the src directory
# These are placeholders for actual implementations in their respective files.
# For this file to be runnable, minimal definitions are provided below the main class.
from src.core.agent import BaseAgent
from src.core.task_manager import TaskManager, Task
from src.core.resource_manager import ResourceManager
from src.core.error_handling import ErrorHandling, OrchestrationError
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

class AgentState(Enum):
    """Represents the current lifecycle state of an agent within the orchestrator."""
    REGISTERED = auto()  # Agent has been registered but not yet started
    IDLE = auto()        # Agent is active but not currently processing a task
    RUNNING = auto()     # Agent is active and potentially processing tasks
    PAUSED = auto()      # Agent is temporarily halted
    STOPPED = auto()     # Agent has been gracefully shut down
    ERROR = auto()       # Agent encountered an unrecoverable error

class Orchestrator:
    """
    The central brain for managing multi-agent systems.
    Responsible for agent registration, lifecycle management (start, stop, pause),
    dynamic scheduling of agent actions, and overall system state management
    across a distributed environment.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        resource_manager: ResourceManager,
        error_handler: ErrorHandling
    ):
        """
        Initializes the Orchestrator with necessary core dependencies.

        Args:
            task_manager (TaskManager): Manages task creation, routing, and assignment.
            resource_manager (ResourceManager): Tracks and allocates resources to agents.
            error_handler (ErrorHandling): Provides robust fault tolerance mechanisms.
        """
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_states: Dict[str, AgentState] = {}
        self._running_orchestrator_tasks: Dict[str, asyncio.Task] = {} # For internal orchestrator tasks like monitoring

        self.task_manager = task_manager
        self.resource_manager = resource_manager
        self.error_handler = error_handler

        logger.info("Orchestrator initialized and ready to manage agents.")

    async def register_agent(self, agent: BaseAgent) -> None:
        """
        Registers an agent with the orchestrator.
        Assigns a unique ID if the agent doesn't have one and sets its initial state.

        Args:
            agent (BaseAgent): The agent instance to register.
        
        Raises:
            OrchestrationError: If an agent with the same ID is already registered.
        """
        if not agent.agent_id:
            agent.agent_id = str(uuid.uuid4())
            logger.debug(f"Assigned new unique ID '{agent.agent_id}' to agent '{agent.name}'.")

        if agent.agent_id in self._agents:
            self.error_handler.handle_error(
                OrchestrationError(f"Attempted to register agent '{agent.name}' with existing ID '{agent.agent_id}'."),
                context={"agent_id": agent.agent_id, "agent_name": agent.name}
            )
            raise OrchestrationError(f"Agent with ID '{agent.agent_id}' is already registered.")

        self._agents[agent.agent_id] = agent
        self._agent_states[agent.agent_id] = AgentState.REGISTERED
        logger.info(f"Agent '{agent.name}' (ID: {agent.agent_id}) registered successfully.")

    async def deregister_agent(self, agent_id: str) -> None:
        """
        Deregisters an agent from the orchestrator.
        Attempts to stop the agent gracefully if it's running or paused.

        Args:
            agent_id (str): The ID of the agent to deregister.
        
        Raises:
            OrchestrationError: If the agent is not found.
        """
        if agent_id not in self._agents:
            self.error_handler.handle_error(
                OrchestrationError(f"Attempted to deregister non-existent agent '{agent_id}'."),
                context={"agent_id": agent_id}
            )
            raise OrchestrationError(f"Agent '{agent_id}' not found for deregistration.")

        agent_name = self._agents[agent_id].name
        current_state = self._agent_states.get(agent_id)

        try:
            if current_state in [AgentState.RUNNING, AgentState.PAUSED, AgentState.IDLE]:
                logger.info(f"Attempting to stop agent '{agent_name}' ({agent_id}) before deregistration.")
                await self.stop_agent(agent_id)
        except OrchestrationError as e:
            logger.warning(f"Error stopping agent '{agent_name}' ({agent_id}) during deregistration. Proceeding anyway: {e}")
            # Continue deregistration even if stopping fails, but log the issue.

        del self._agents[agent_id]
        if agent_id in self._agent_states: # Ensure state is removed even if stop_agent failed or modified it.
            del self._agent_states[agent_id]
        
        logger.info(f"Agent '{agent_name}' (ID: {agent_id}) deregistered successfully.")

    async def start_agent(self, agent_id: str) -> None:
        """
        Starts a registered agent, allowing it to begin processing tasks.

        Args:
            agent_id (str): The ID of the agent to start.
        
        Raises:
            OrchestrationError: If the agent is not found or cannot be started.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            self.error_handler.handle_error(
                OrchestrationError(f"Agent '{agent_id}' not found."),
                context={"agent_id": agent_id}
            )
            raise OrchestrationError(f"Agent '{agent_id}' not found.")

        current_state = self._agent_states.get(agent_id)
        if current_state in [AgentState.RUNNING, AgentState.IDLE]:
            logger.debug(f"Agent '{agent.name}' (ID: {agent_id}) is already active ({current_state.name}). No action needed.")
            return
        if current_state == AgentState.ERROR:
            logger.warning(f"Agent '{agent.name}' (ID: {agent_id}) is in ERROR state. Attempting restart.")

        try:
            await agent.start()  # Assumes BaseAgent has an async start method
            self._agent_states[agent_id] = AgentState.IDLE # Agent starts as IDLE, ready to take tasks
            logger.info(f"Agent '{agent.name}' (ID: {agent_id}) started successfully and is now {AgentState.IDLE.name}.")
        except Exception as e:
            self.error_handler.handle_error(
                OrchestrationError(f"Failed to start agent '{agent.name}' (ID: {agent_id}): {e}"),
                context={"agent_id": agent_id, "agent_name": agent.name, "error_type": type(e).__name__, "error_msg": str(e)}
            )
            self._agent_states[agent_id] = AgentState.ERROR
            raise OrchestrationError(f"Failed to start agent '{agent.name}' (ID: {agent_id}).") from e

    async def stop_agent(self, agent_id: str) -> None:
        """
        Stops a running or paused agent, preventing it from processing further tasks.

        Args:
            agent_id (str): The ID of the agent to stop.
        
        Raises:
            OrchestrationError: If the agent is not found or cannot be stopped.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            self.error_handler.handle_error(
                OrchestrationError(f"Agent '{agent_id}' not found."),
                context={"agent_id": agent_id}
            )
            raise OrchestrationError(f"Agent '{agent_id}' not found.")

        current_state = self._agent_states.get(agent_id)
        if current_state in [AgentState.STOPPED, AgentState.REGISTERED]:
            logger.debug(f"Agent '{agent.name}' (ID: {agent_id}) is already {current_state.name}. No action needed.")
            return

        try:
            await agent.stop()  # Assumes BaseAgent has an async stop method
            self._agent_states[agent_id] = AgentState.STOPPED
            logger.info(f"Agent '{agent.name}' (ID: {agent_id}) stopped successfully.")
        except Exception as e:
            self.error_handler.handle_error(
                OrchestrationError(f"Failed to stop agent '{agent.name}' (ID: {agent_id}): {e}"),
                context={"agent_id": agent_id, "agent_name": agent.name, "error_type": type(e).__name__, "error_msg": str(e)}
            )
            self._agent_states[agent_id] = AgentState.ERROR # Mark as error if stop fails
            raise OrchestrationError(f"Failed to stop agent '{agent.name}' (ID: {agent_id}).") from e

    async def pause_agent(self, agent_id: str) -> None:
        """
        Pauses a running agent, temporarily halting its task processing.

        Args:
            agent_id (str): The ID of the agent to pause.
        
        Raises:
            OrchestrationError: If the agent is not found or cannot be paused.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            self.error_handler.handle_error(
                OrchestrationError(f"Agent '{agent_id}' not found."),
                context={"agent_id": agent_id}
            )
            raise OrchestrationError(f"Agent '{agent_id}' not found.")

        current_state = self._agent_states.get(agent_id)
        if current_state == AgentState.PAUSED:
            logger.debug(f"Agent '{agent.name}' (ID: {agent_id}) is already paused. No action needed.")
            return
        if current_state != AgentState.RUNNING and current_state != AgentState.IDLE:
            logger.warning(f"Agent '{agent.name}' (ID: {agent_id}) is not in a pausable state (current: {current_state.name}).")
            return

        try:
            await agent.pause()  # Assumes BaseAgent has an async pause method
            self._agent_states[agent_id] = AgentState.PAUSED
            logger.info(f"Agent '{agent.name}' (ID: {agent_id}) paused successfully.")
        except Exception as e:
            self.error_handler.handle_error(
                OrchestrationError(f"Failed to pause agent '{agent.name}' (ID: {agent_id}): {e}"),
                context={"agent_id": agent_id, "agent_name": agent.name, "error_type": type(e).__name__, "error_msg": str(e)}
            )
            self._agent_states[agent_id] = AgentState.ERROR # Mark as error if pause fails
            raise OrchestrationError(f"Failed to pause agent '{agent.name}' (ID: {agent_id}).") from e

    def get_agent_status(self, agent_id: str) -> Optional[AgentState]:
        """
        Retrieves the current state of a specific agent.

        Args:
            agent_id (str): The ID of the agent.

        Returns:
            Optional[AgentState]: The state of the agent, or None if not found.
        """
        return self._agent_states.get(agent_id)

    def get_all_agent_statuses(self) -> Dict[str, AgentState]:
        """
        Retrieves the current states of all registered agents.

        Returns:
            Dict[str, AgentState]: A dictionary mapping agent IDs to their states.
        """
        return self._agent_states.copy()

    async def dispatch_task(self, task_payload: Any, preferred_agent_id: Optional[str] = None) -> Task:
        """
        Creates, routes, allocates resources for, and assigns a task to an appropriate agent.

        Args:
            task_payload (Any): The data or instructions for the task.
            preferred_agent_id (Optional[str]): An optional ID of a preferred agent for this task.

        Returns:
            Task: The created and dispatched Task object.

        Raises:
            OrchestrationError: If no suitable agent is found, resources are insufficient, or task dispatch fails.
        """
        try:
            # 1. Create the task
            task = self.task_manager.create_task(task_payload)
            logger.debug(f"Task '{task.task_id}' created with payload: {task_payload}")

            # 2. Identify and filter suitable agents
            candidate_agents: List[BaseAgent] = []
            if preferred_agent_id:
                if preferred_agent_id not in self._agents:
                    raise OrchestrationError(f"Preferred agent '{preferred_agent_id}' is not registered.")
                if self._agent_states.get(preferred_agent_id) not in [AgentState.RUNNING, AgentState.IDLE]:
                    raise OrchestrationError(f"Preferred agent '{preferred_agent_id}' is not available (current state: {self._agent_states.get(preferred_agent_id).name}).")
                candidate_agents.append(self._agents[preferred_agent_id])
            else:
                for agent_id, state in self._agent_states.items():
                    if state in [AgentState.RUNNING, AgentState.IDLE]:
                        candidate_agents.append(self._agents[agent_id])
            
            if not candidate_agents:
                raise OrchestrationError("No agents are currently running or idle to accept tasks.")

            # 3. Route the task to the best agent from candidates
            assigned_agent_id = await self.task_manager.route_task(task, candidate_agents)

            if not assigned_agent_id:
                raise OrchestrationError(f"TaskManager could not find a suitable agent for task '{task.task_id}'.")
            
            assigned_agent = self._agents[assigned_agent_id]
            
            # 4. Resource allocation check and commit
            # This is a critical step to ensure agents don't exceed their limits.
            if not await self.resource_manager.check_and_allocate(assigned_agent_id, task.estimated_resources):
                raise OrchestrationError(f"Insufficient resources for agent '{assigned_agent_id}' to execute task '{task.task_id}'.")

            # 5. Assign and execute the task (delegating execution to TaskManager/Agent)
            # The TaskManager is responsible for communicating the task to the agent and tracking its lifecycle.
            # It might push to an agent's queue or invoke an agent's method directly.
            self._agent_states[assigned_agent_id] = AgentState.RUNNING # Mark agent as running while processing task
            await self.task_manager.assign_task(task, assigned_agent)
            
            logger.info(f"Task '{task.task_id}' dispatched to agent '{assigned_agent.name}' (ID: {assigned_agent_id}).")
            return task
        except Exception as e:
            # Clean up allocated resources if task dispatch fails after allocation
            if 'assigned_agent_id' in locals() and 'task' in locals():
                try:
                    await self.resource_manager.deallocate(assigned_agent_id, task.estimated_resources)
                    logger.warning(f"Deallocated resources for failed task '{task.task_id}' on agent '{assigned_agent_id}'.")
                except Exception as dealloc_e:
                    logger.error(f"Failed to deallocate resources for agent '{assigned_agent_id}' after task '{task.task_id}' dispatch failure: {dealloc_e}")
            
            self.error_handler.handle_error(
                OrchestrationError(f"Failed to dispatch task with payload {task_payload}: {e}"),
                context={"task_payload": task_payload, "preferred_agent_id": preferred_agent_id, "error_type": type(e).__name__, "error_msg": str(e)}
            )
            raise OrchestrationError(f"Failed to dispatch task: {e}") from e

    async def _monitor_agents(self, interval: int = 10) -> None:
        """
        Internal periodic background task to monitor the health and status of registered agents.
        This could involve checking heartbeats, resource usage, or responsiveness.
        """
        logger.info(f"Orchestrator monitoring agents every {interval} seconds.")
        while True:
            await asyncio.sleep(interval)
            monitored_agents_count = 0
            for agent_id, state in self._agent_states.items():
                # Only monitor active or potentially problematic agents
                if state not in [AgentState.STOPPED, AgentState.REGISTERED]:
                    monitored_agents_count += 1
                    agent = self._agents[agent_id]
                    logger.debug(f"Monitoring agent '{agent.name}' (ID: {agent_id}) - Current State: {state.name}")
                    
                    try:
                        # Example: Check resource usage for active agents
                        current_usage = await self.resource_manager.get_current_usage(agent_id)
                        if current_usage:
                            logger.debug(f"Agent '{agent.name}' (ID: {agent_id}) resource usage: {current_usage}")
                            # Example: Trigger warning if usage exceeds a threshold
                            if current_usage.get("cpu_percent", 0) > 90:
                                logger.warning(f"Agent '{agent.name}' (ID: {agent_id}) showing high CPU usage: {current_usage.get('cpu_percent')}%")
                            if current_usage.get("memory_usage_mb", 0) > 500: # Example threshold
                                logger.warning(f"Agent '{agent.name}' (ID: {agent_id}) showing high memory usage: {current_usage.get('memory_usage_mb')}MB")
                        
                        # Add more sophisticated health checks here (e.g., agent heartbeat, queue depth, error rates)
                        # If a critical issue is detected, change agent state to ERROR and potentially restart.
                    except Exception as e:
                        logger.error(f"Error during monitoring of agent '{agent.name}' (ID: {agent_id}): {e}")
                        if self._agent_states[agent_id] != AgentState.ERROR:
                            self._agent_states[agent_id] = AgentState.ERROR
                            self.error_handler.handle_error(
                                OrchestrationError(f"Agent '{agent.name}' ({agent_id}) unresponsive during monitoring."),
                                context={"agent_id": agent_id, "agent_name": agent.name, "monitoring_error": str(e)}
                            )
            
            if monitored_agents_count > 0:
                logger.debug(f"Completed monitoring of {monitored_agents_count} agents.")
            else:
                logger.debug("No active agents found to monitor.")

    async def start(self) -> None:
        """
        Starts the orchestrator's internal background tasks, such as monitoring.
        This method should be called once to bring the orchestrator online.
        """
        if "monitor" not in self._running_orchestrator_tasks or self._running_orchestrator_tasks["monitor"].done():
            self._running_orchestrator_tasks["monitor"] = asyncio.create_task(self._monitor_agents())
            logger.info("Orchestrator background monitoring task started.")
        else:
            logger.warning("Orchestrator monitoring is already running or being started.")
        logger.info("Orchestrator is fully operational.")

    async def stop(self) -> None:
        """
        Stops all registered agents gracefully and cleans up orchestrator's internal tasks.
        This method should be called during system shutdown.
        """
        logger.info("Orchestrator initiating shutdown process...")
        
        # 1. Stop all active agents
        agent_ids_to_stop = list(self._agent_states.keys()) # Get a copy of agent IDs
        stop_tasks = []
        for agent_id in agent_ids_to_stop:
            state = self._agent_states.get(agent_id)
            if state in [AgentState.RUNNING, AgentState.IDLE, AgentState.PAUSED, AgentState.ERROR]:
                stop_tasks.append(self.stop_agent(agent_id))
        
        if stop_tasks:
            logger.info(f"Attempting to stop {len(stop_tasks)} active agents.")
            # Run stop tasks concurrently, but gather results (even if they raise exceptions)
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            logger.info("Finished attempting to stop all active agents.")
        else:
            logger.info("No active agents found to stop.")
        
        # 2. Cancel any running internal orchestrator tasks
        for task_name, task in self._running_orchestrator_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Orchestrator internal task '{task_name}' cancellation requested.")
                try:
                    await task # Await to ensure cancellation is processed
                except asyncio.CancelledError:
                    logger.info(f"Orchestrator internal task '{task_name}' successfully cancelled.")
                except Exception as e:
                    logger.error(f"Error while cancelling orchestrator internal task '{task_name}': {e}")
        self._running_orchestrator_tasks.clear()
        
        logger.info("Orchestrator shutdown complete.")

# --- Placeholder definitions for assumed dependencies (for standalone execution/type checking) ---
# In a real project, these would be proper imports from their respective files.

# src/core/agent.py (minimal definition)
class BaseAgent:
    def __init__(self, agent_id: Optional[str] = None, name: str = "UnnamedAgent"):
        self.agent_id = agent_id
        self.name = name
        self.context: Dict[str, Any] = {}
        self._is_started = False

    async def start(self):
        if not self._is_started:
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) starting...")
            await asyncio.sleep(0.1) # Simulate async startup
            self._is_started = True
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) started.")
        else:
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) already started.")

    async def stop(self):
        if self._is_started:
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) stopping...")
            await asyncio.sleep(0.1) # Simulate async shutdown
            self._is_started = False
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) stopped.")
        else:
            logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) already stopped.")

    async def pause(self):
        logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) pausing...")
        await asyncio.sleep(0.05)
        logger.debug(f"Agent '{self.name}' (ID: {self.agent_id}) paused.")

    async def execute_task(self, task: 'Task') -> Any:
        """Abstract method for agents to implement task processing."""
        logger.info(f"Agent '{self.name}' (ID: {self.agent_id}) processing task '{task.task_id}' with payload: {task.payload}")
        await asyncio.sleep(1) # Simulate work
        return {"result": f"Task '{task.task_id}' completed by agent '{self.name}'"}


# src/core/task_manager.py (minimal definition)
class Task:
    def __init__(self, task_id: str, payload: Any, estimated_resources: Dict[str, Any]):
        self.task_id = task_id
        self.payload = payload
        self.status = "CREATED"
        self.assigned_agent_id: Optional[str] = None
        self.result: Any = None
        self.estimated_resources = estimated_resources
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        logger.info("TaskManager initialized.")

    def create_task(self, payload: Any) -> Task:
        task_id = str(uuid.uuid4())
        # A real implementation would parse payload to estimate resources more accurately
        estimated_resources = {"cpu": 1.0, "memory_mb": 100, "llm_tokens": 1000} 
        task = Task(task_id, payload, estimated_resources)
        self._tasks[task_id] = task
        logger.debug(f"Task '{task_id}' created by TaskManager.")
        return task

    async def route_task(self, task: Task, suitable_agents: List[BaseAgent]) -> Optional[str]:
        """
        Selects the best agent for a given task from a list of suitable candidates.
        This simplified version just picks the first one.
        """
        if suitable_agents:
            selected_agent = suitable_agents[0]
            logger.debug(f"TaskManager routed task '{task.task_id}' to agent '{selected_agent.name}' (ID: {selected_agent.agent_id}).")
            return selected_agent.agent_id
        logger.warning(f"TaskManager found no suitable agents for task '{task.task_id}'.")
        return None

    async def assign_task(self, task: Task, agent: BaseAgent) -> None:
        """
        Assigns a task to a specific agent and initiates its execution.
        """
        task.assigned_agent_id = agent.agent_id
        task.status = "ASSIGNED"
        task.start_time = asyncio.get_event_loop().time()
        logger.info(f"Task '{task.task_id}' assigned to agent '{agent.name}' (ID: {agent.agent_id}). Initiating execution.")

        try:
            # Direct call to agent's execute_task for simplicity.
            # In a distributed setup, this would be an RPC call or message queue publish.
            task.result = await agent.execute_task(task)
            task.status = "COMPLETED"
            task.end_time = asyncio.get_event_loop().time()
            logger.info(f"Task '{task.task_id}' completed by agent '{agent.name}'. Duration: {task.end_time - task.start_time:.2f}s.")
        except Exception as e:
            task.status = "FAILED"
            task.end_time = asyncio.get_event_loop().time()
            logger.error(f"Task '{task.task_id}' failed on agent '{agent.name}': {e}. Duration: {task.end_time - task.start_time:.2f}s.")
            raise # Re-raise to allow orchestrator's error handler to catch it

# src/core/resource_manager.py (minimal definition)
class ResourceManager:
    def __init__(self):
        self._allocated_resources: Dict[str, Dict[str, Any]] = {} # agent_id -> {resource: value}
        self._current_usage: Dict[str, Dict[str, Any]] = {} # agent_id -> {metric: value}
        self._resource_limits: Dict[str, Dict[str, Any]] = {} # agent_id -> {resource: max_value}
        logger.info("ResourceManager initialized.")

    async def check_and_allocate(self, agent_id: str, required_resources: Dict[str, Any]) -> bool:
        """
        Simulates checking and allocating resources for an agent.
        Returns True if resources can be allocated, False otherwise.
        """
        logger.debug(f"Checking/allocating resources for agent '{agent_id}': {required_resources}")
        # Simple check: always allow for this placeholder. A real implementation would check against limits.
        
        current_allocations = self._allocated_resources.setdefault(agent_id, {})
        for res_name, res_value in required_resources.items():
            current_allocations[res_name] = current_allocations.get(res_name, 0) + res_value
        
        await asyncio.sleep(0.01) # Simulate async operation
        return True

    async def deallocate(self, agent_id: str, resources: Dict[str, Any]) -> None:
        """
        Simulates deallocating resources for an agent.
        """
        logger.debug(f"Deallocating resources for agent '{agent_id}': {resources}")
        if agent_id in self._allocated_resources:
            for res_name, res_value in resources.items():
                self._allocated_resources[agent_id][res_name] = max(0, self._allocated_resources[agent_id].get(res_name, 0) - res_value)
        await asyncio.sleep(0.01)

    async def get_current_usage(self, agent_id: str) -> Dict[str, Any]:
        """
        Simulates getting current resource usage metrics for an agent.
        In a real system, this would query monitoring systems or the agent directly.
        """
        await asyncio.sleep(0.01)
        # Return dummy data for illustration
        return {
            "cpu_percent": 50, # Example: 50% CPU usage
            "memory_usage_mb": 150, # Example: 150MB memory usage
            "llm_tokens_used": 1200 # Example: 1200 LLM tokens used
        }

# src/core/error_handling.py (minimal definition)
class OrchestrationError(Exception):
    """Custom exception for orchestration-related errors."""
    pass

class ErrorHandling:
    def __init__(self):
        logger.info("ErrorHandling initialized.")

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs an error and potentially triggers fault tolerance mechanisms.
        """
        error_msg = f"Orchestration Fault Detected: {error.__class__.__name__}: {error}"
        if context:
            error_msg += f" Context: {context}"
        logger.error(error_msg, exc_info=True) # exc_info=True logs stack trace
        
        # In a real system, this method would implement robust fault tolerance:
        # - Publish to a dead-letter queue for failed tasks.
        # - Increment metrics for error rates.
        # - Trigger circuit breakers if a specific agent or service repeatedly fails.
        # - Send alerts (e.g., to Sentry, PagerDuty).
        # - Log specific actions based on error type (e.g., initiate agent restart).


# src/monitoring/logger.py (minimal definition)
import logging

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger instance.
    Ensures basic console logging is set up if no handlers exist.
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.DEBUG) # Default to DEBUG level

    # Prevent adding duplicate handlers if get_logger is called multiple times
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.propagate = False # Prevent messages from being passed to the root logger
    return logger_instance
```