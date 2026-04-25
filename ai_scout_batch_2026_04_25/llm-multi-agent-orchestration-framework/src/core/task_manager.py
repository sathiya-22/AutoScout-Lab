```python
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable

class TaskStatus(Enum):
    """
    Represents the current status of a task in the multi-agent system.
    """
    PENDING = "PENDING"      # Task has been created but not yet assigned.
    ASSIGNED = "ASSIGNED"    # Task has been assigned to an agent.
    IN_PROGRESS = "IN_PROGRESS" # Agent is actively working on the task.
    COMPLETED = "COMPLETED"  # Task has been successfully completed.
    FAILED = "FAILED"        # Task failed during execution.
    CANCELLED = "CANCELLED"  # Task was explicitly cancelled.

class Task:
    """
    Represents a single task to be processed by an agent within the system.
    Tasks are immutable once created, with status and assignment being updated
    through the TaskManager.
    """
    def __init__(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5, # 1 (highest) to 10 (lowest)
        status: TaskStatus = TaskStatus.PENDING,
        created_at: Optional[datetime] = None,
        assigned_agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not task_id:
            raise ValueError("Task ID cannot be empty.")
        if not task_type:
            raise ValueError("Task type cannot be empty.")
        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dictionary.")
        if not isinstance(priority, int) or not (1 <= priority <= 10):
            raise ValueError("Priority must be an integer between 1 and 10.")
        if not isinstance(status, TaskStatus):
            raise TypeError("Status must be a valid TaskStatus enum member.")

        self.task_id = task_id
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.status = status
        self.created_at = created_at if created_at is not None else datetime.utcnow()
        self.assigned_agent_id = assigned_agent_id
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        return (f"Task(id='{self.task_id}', type='{self.task_type}', "
                f"status={self.status.value}, priority={self.priority}, "
                f"assigned_agent='{self.assigned_agent_id or 'None'}')")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Task object to a dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "assigned_agent_id": self.assigned_agent_id,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Task':
        """Creates a Task object from a dictionary."""
        return Task(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data["payload"],
            priority=data.get("priority", 5),
            status=TaskStatus[data.get("status", "PENDING")],
            created_at=datetime.fromisoformat(data["created_at"]),
            assigned_agent_id=data.get("assigned_agent_id"),
            metadata=data.get("metadata", {}),
        )

# Type hint for a callable that resolves an agent's capabilities given its ID.
# This function is expected to return a list of strings representing the agent's capabilities (e.g., "code_generation", "data_analysis").
AgentCapabilityResolver = Callable[[str], List[str]]

class TaskManager:
    """
    Manages the lifecycle of tasks within the multi-agent system.
    Handles task creation, prioritization, routing suggestions, and status updates.

    It relies on an external `agent_capability_resolver` (typically provided by the
    Orchestrator) to determine what tasks an agent can perform.
    """
    def __init__(self, agent_capability_resolver: Optional[AgentCapabilityResolver] = None):
        # Stores all tasks by their ID
        self._tasks: Dict[str, Task] = {}
        # Stores IDs of tasks that are currently PENDING, sorted by priority (lowest number = highest priority)
        # and then by creation time (FIFO for same priority).
        # In a production system, this would be a persistent, distributed priority queue.
        self._pending_tasks_ids: List[str] = []
        
        # The resolver should ideally be provided by the Orchestrator, which has access to the AgentRegistry.
        self.agent_capability_resolver = agent_capability_resolver or self._default_agent_capability_resolver

    def _default_agent_capability_resolver(self, agent_id: str) -> List[str]:
        """
        A default placeholder resolver for agent capabilities.
        In a real system, this would query the Orchestrator or AgentRegistry for actual agent capabilities.
        If no specific resolver is provided, this method will be used, implying no specific capabilities
        unless explicitly handled by downstream logic.
        """
        # print(f"Warning: Using default agent capability resolver for agent '{agent_id}'. "
        #       "Integrate with Agent Abstraction/Orchestrator for real capabilities.")
        return []

    async def create_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5, # 1 (highest) to 10 (lowest)
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Creates a new task and adds it to the system. The task is initially in PENDING status.

        Args:
            task_type: A string identifying the category of task (e.g., "code_generation", "data_analysis").
            payload: A dictionary containing the specific data or instructions for the task.
            priority: An integer from 1 (highest) to 10 (lowest), dictating allocation order.
            metadata: Optional dictionary for additional context or debugging information.

        Returns:
            The newly created Task object.
        Raises:
            ValueError, TypeError: If input parameters are invalid.
        """
        task_id = str(uuid.uuid4())
        new_task = Task(task_id, task_type, payload, priority, metadata=metadata)
        self._tasks[task_id] = new_task
        await self._add_to_pending_queue(task_id)
        return new_task

    async def _add_to_pending_queue(self, task_id: str):
        """
        Adds a task ID to the internal pending queue, maintaining priority order.
        Higher priority (lower integer value) tasks come first. For tasks with the same priority,
        the older task (earlier created_at) comes first (FIFO).
        """
        task = self._tasks[task_id]
        if task.status != TaskStatus.PENDING:
            # Only add pending tasks to the queue
            return

        # Simple O(N) insertion to maintain sorted order for a prototype.
        # For large task queues, consider `heapq` or a persistent, distributed priority queue.
        inserted = False
        for i, tid in enumerate(self._pending_tasks_ids):
            other_task = self._tasks[tid]
            if task.priority < other_task.priority:
                self._pending_tasks_ids.insert(i, task_id)
                inserted = True
                break
            elif task.priority == other_task.priority and task.created_at < other_task.created_at:
                 self._pending_tasks_ids.insert(i, task_id)
                 inserted = True
                 break
        if not inserted:
            self._pending_tasks_ids.append(task_id)

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieves a task by its unique identifier.

        Args:
            task_id: The unique identifier of the task.

        Returns:
            The Task object if found, otherwise None.
        """
        return self._tasks.get(task_id)

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        assigned_agent_id: Optional[str] = None,
    ) -> bool:
        """
        Updates the status of an existing task.

        Args:
            task_id: The ID of the task to update.
            status: The new status for the task.
            assigned_agent_id: Optional ID of the agent currently assigned to the task.
                               This field is updated only if provided.

        Returns:
            True if the task was found and updated, False otherwise (e.g., task not found).
        """
        if not isinstance(status, TaskStatus):
            raise TypeError("Status must be a valid TaskStatus enum member.")

        task = self._tasks.get(task_id)
        if not task:
            # Consider logging this as a warning in a real system.
            return False

        task.status = status
        if assigned_agent_id is not None: # Allow clearing assignment by passing None explicitly
            task.assigned_agent_id = assigned_agent_id

        # Manage task in the pending queue based on its new status
        if status == TaskStatus.PENDING and task_id not in self._pending_tasks_ids:
            await self._add_to_pending_queue(task_id)
        elif status != TaskStatus.PENDING and task_id in self._pending_tasks_ids:
            self._pending_tasks_ids.remove(task_id)

        return True

    async def get_pending_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """
        Retrieves a list of tasks that are currently pending, ordered by priority
        (highest priority first) and then by creation time (FIFO).

        Args:
            limit: Optional maximum number of pending tasks to return.

        Returns:
            A list of pending Task objects.
        """
        # Ensure _pending_tasks_ids only contains valid, PENDING tasks
        # This loop also serves as a cleanup mechanism if tasks were updated directly
        # or removed without going through _remove_from_pending_queue.
        current_pending_tasks_ids = []
        pending_tasks_objects = []
        for task_id in self._pending_tasks_ids:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                current_pending_tasks_ids.append(task_id)
                pending_tasks_objects.append(task)
        self._pending_tasks_ids = current_pending_tasks_ids # Resync the internal list

        return pending_tasks_objects[:limit] if limit is not None else pending_tasks_objects

    async def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """
        Assigns a specific task to an agent and updates its status to ASSIGNED.
        If the task was pending, it is removed from the pending queue.

        Args:
            task_id: The ID of the task to assign.
            agent_id: The ID of the agent to assign the task to.

        Returns:
            True if the assignment was successful, False otherwise (e.g., task not found,
            or task is already in a finalized state like COMPLETED/FAILED/CANCELLED).
        """
        task = self._tasks.get(task_id)
        if not task:
            # print(f"Warning: Attempted to assign non-existent task {task_id} to agent {agent_id}.")
            return False

        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            # print(f"Warning: Attempted to assign a finalized task {task_id} to agent {agent_id}. Current status: {task.status.value}")
            return False

        return await self.update_task_status(task_id, TaskStatus.ASSIGNED, assigned_agent_id=agent_id)

    async def find_suitable_agents(
        self,
        task: Task,
        available_agent_ids: List[str],
    ) -> List[str]:
        """
        Identifies agents capable of handling a given task based on their declared capabilities.
        This method primarily checks for capability match using the `agent_capability_resolver`.
        It does NOT consider current agent load, availability, or resource constraints;
        these considerations are typically handled by the Orchestrator/ResourceManager
        when making the final task allocation decision.

        Args:
            task: The Task object for which to find suitable agents.
            available_agent_ids: A list of IDs of agents that are currently active and potentially available.

        Returns:
            A list of agent IDs that are suitable for the given task.
        """
        suitable_agent_ids = []
        # Assumption: `task.task_type` directly corresponds to a required agent capability.
        # In a more complex system, `task_type` could map to a set of required capabilities,
        # or `task.metadata` could specify requirements.
        required_capability = task.task_type

        for agent_id in available_agent_ids:
            try:
                agent_capabilities = self.agent_capability_resolver(agent_id)
                if required_capability in agent_capabilities:
                    suitable_agent_ids.append(agent_id)
            except Exception as e:
                # Log this error instead of printing in a production system.
                print(f"Error resolving capabilities for agent {agent_id}: {e}")
                # Decide if an agent with resolver error should be excluded or retried.
                # For now, we simply skip it.

        return suitable_agent_ids

    async def get_all_tasks(self) -> List[Task]:
        """
        Retrieves all tasks known to the TaskManager, regardless of their status.

        Returns:
            A list of all Task objects.
        """
        return list(self._tasks.values())

    async def delete_task(self, task_id: str) -> bool:
        """
        Deletes a task from the TaskManager. This operation is typically permanent.
        Use with caution, as it removes all historical data for the task.

        Args:
            task_id: The ID of the task to delete.

        Returns:
            True if the task was found and deleted, False otherwise.
        """
        if task_id in self._tasks:
            if task_id in self._pending_tasks_ids:
                self._pending_tasks_ids.remove(task_id)
            del self._tasks[task_id]
            return True
        return False
```