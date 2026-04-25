```python
import asyncio
import time
import functools
import logging
from collections import deque
from enum import Enum

# Setup basic logging for the error handling module
logger = logging.getLogger(__name__)
# Prevent adding multiple handlers if the module is reloaded or imported multiple times
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Custom Exceptions ---
class OrchestrationError(Exception):
    """Base exception for all orchestration-related errors."""
    pass

class AgentCommunicationError(OrchestrationError):
    """Raised when there's an issue communicating with an agent."""
    pass

class AgentTaskFailedError(OrchestrationError):
    """Raised when an agent reports a task failure."""
    def __init__(self, message: str, task_id: str = None, agent_id: str = None, original_exception: Exception = None):
        super().__init__(message)
        self.task_id = task_id
        self.agent_id = agent_id
        self.original_exception = original_exception
        logger.error(f"AgentTaskFailedError: {message} | Task ID: {task_id}, Agent ID: {agent_id}")

class AgentUnavailableError(OrchestrationError):
    """Raised when an agent cannot be reached or is considered offline."""
    pass

class ResourceAllocationError(OrchestrationError):
    """Raised when resources cannot be allocated or are insufficient."""
    pass

class CircuitBreakerOpenError(OrchestrationError):
    """Raised when a circuit breaker is open, preventing further calls."""
    def __init__(self, service_name: str = "unknown"):
        super().__init__(f"Circuit breaker for '{service_name}' is open. Request prevented.")
        self.service_name = service_name
        logger.warning(f"CircuitBreakerOpenError for service: {service_name}")

class MaxRetriesExceededError(OrchestrationError):
    """Raised when an operation fails after exhausting all retry attempts."""
    def __init__(self, message: str, last_exception: Exception = None):
        super().__init__(message)
        self.last_exception = last_exception
        logger.error(f"MaxRetriesExceededError: {message} | Last exception: {type(last_exception).__name__}")


# --- Retry Mechanism ---

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_catch: tuple = (Exception,),
    log_level: int = logging.WARNING
):
    """
    A decorator for retrying an asynchronous or synchronous function upon specified exceptions.

    Args:
        max_attempts (int): The maximum number of times to attempt the function call. Must be >= 1.
        delay (float): The initial delay in seconds between retries. Must be >= 0.
        backoff_factor (float): Factor by which the delay will increase each attempt. Must be >= 1.
        exceptions_to_catch (tuple): A tuple of exception types to catch and retry on.
        log_level (int): The logging level for retry attempts (e.g., logging.INFO, logging.WARNING).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be at least 1")

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions_to_catch as e:
                    logger.log(
                        log_level,
                        f"Attempt {attempt}/{max_attempts} for '{func.__name__}' failed: {type(e).__name__}: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    if attempt == max_attempts:
                        logger.error(f"Function '{func.__name__}' failed after {max_attempts} attempts.")
                        raise MaxRetriesExceededError(
                            f"Operation '{func.__name__}' failed after {max_attempts} attempts.",
                            last_exception=e
                        ) from e
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    logger.log(
                        log_level,
                        f"Attempt {attempt}/{max_attempts} for '{func.__name__}' failed: {type(e).__name__}: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    if attempt == max_attempts:
                        logger.error(f"Function '{func.__name__}' failed after {max_attempts} attempts.")
                        raise MaxRetriesExceededError(
                            f"Operation '{func.__name__}' failed after {max_attempts} attempts.",
                            last_exception=e
                        ) from e
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# --- Circuit Breaker ---

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"

class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern to prevent repeated calls to a failing service.
    Works for both synchronous and asynchronous functions.
    """
    def __init__(self,
                 service_name: str,
                 failure_threshold: int = 3,
                 reset_timeout_seconds: int = 30,
                 recovery_success_threshold: int = 1):
        """
        Args:
            service_name (str): The name of the service/function this breaker protects.
            failure_threshold (int): Number of consecutive failures before the circuit opens. Must be >= 1.
            reset_timeout_seconds (int): Time in seconds before the circuit transitions from OPEN to HALF_OPEN. Must be >= 0.
            recovery_success_threshold (int): Number of successful calls in HALF_OPEN state to CLOSE the circuit. Must be >= 1.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if reset_timeout_seconds < 0:
            raise ValueError("reset_timeout_seconds must be non-negative")
        if recovery_success_threshold < 1:
            raise ValueError("recovery_success_threshold must be at least 1")

        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.recovery_success_threshold = recovery_success_threshold

        self._state = CircuitBreakerState.CLOSED
        self._last_failure_time: float = 0.0 # epoch time of last failure
        self._current_failures: int = 0
        self._current_successes_in_half_open: int = 0
        self._lock = asyncio.Lock() # For async state protection

        logger.info(f"Circuit Breaker '{self.service_name}' initialized. "
                    f"Threshold: {failure_threshold}, Reset Timeout: {reset_timeout_seconds}s, "
                    f"Recovery Successes: {recovery_success_threshold}")

    @property
    def state(self) -> CircuitBreakerState:
        """Returns the current state of the circuit breaker."""
        # For simplicity and to avoid blocking sync calls, we access _state directly
        # for a property. State changes are handled by async methods with locks.
        return self._state

    async def _change_state(self, new_state: CircuitBreakerState):
        """Changes the circuit breaker state and resets relevant counters."""
        async with self._lock:
            if self._state != new_state:
                logger.debug(f"CB '{self.service_name}' transitioning from {self._state.value} to {new_state.value}")
                self._state = new_state
                if new_state == CircuitBreakerState.CLOSED:
                    self._current_failures = 0
                    self._last_failure_time = 0.0
                    self._current_successes_in_half_open = 0
                elif new_state == CircuitBreakerState.OPEN:
                    self._last_failure_time = time.time()
                    self._current_failures = 0 # failures start fresh from 0 in OPEN state
                    self._current_successes_in_half_open = 0
                elif new_state == CircuitBreakerState.HALF_OPEN:
                    self._current_successes_in_half_open = 0
                    self._current_failures = 0 # failures start fresh from 0 in HALF_OPEN

    async def _record_failure(self):
        """Records a failure and updates the circuit breaker state."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                # If a failure occurs in HALF_OPEN, immediately open the circuit again
                await self._change_state(CircuitBreakerState.OPEN)
                logger.warning(f"CB '{self.service_name}' failed in HALF_OPEN, immediately returning to OPEN.")
            elif self._state == CircuitBreakerState.CLOSED:
                self._current_failures += 1
                logger.warning(f"CB '{self.service_name}' recorded failure {self._current_failures}/{self.failure_threshold}.")
                if self._current_failures >= self.failure_threshold:
                    await self._change_state(CircuitBreakerState.OPEN)
                    logger.error(f"CB '{self.service_name}' opened due to {self._current_failures} failures.")
            # If already OPEN, do nothing (no new failure count needed)

    async def _record_success(self):
        """Records a success and updates the circuit breaker state."""
        async with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._current_successes_in_half_open += 1
                logger.info(f"CB '{self.service_name}' recorded success in HALF_OPEN: {self._current_successes_in_half_open}/{self.recovery_success_threshold}.")
                if self._current_successes_in_half_open >= self.recovery_success_threshold:
                    await self._change_state(CircuitBreakerState.CLOSED)
                    logger.info(f"CB '{self.service_name}' closed after successful recovery in HALF_OPEN.")
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success if currently closed, defensively
                if self._current_failures > 0:
                    self._current_failures = 0

    async def _check_and_transition_state(self):
        """Checks and potentially transitions the circuit breaker state based on time."""
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                elapsed_time = time.time() - self._last_failure_time
                if elapsed_time > self.reset_timeout_seconds:
                    await self._change_state(CircuitBreakerState.HALF_OPEN)
                    logger.info(f"CB '{self.service_name}' moved to HALF_OPEN state after timeout ({self.reset_timeout_seconds}s).")
            return self._state

    def __call__(self, func):
        """Decorator to apply the circuit breaker to a function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_state = await self._check_and_transition_state()
            if current_state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(self.service_name)

            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result
            except Exception as e:
                await self._record_failure()
                raise e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Synchronous decorator usage for async CircuitBreaker methods needs careful handling.
            # We must use asyncio.run or loop.run_until_complete if we want to call async methods.
            # This can be problematic if there's already an active event loop.
            # For a mixed sync/async system, it's generally better to ensure that
            # calls to a CB are consistent (either always async or always sync).
            # For this prototype, we'll try to get the current loop or create a new one.

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError: # No running loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                # Note: Creating a new loop for each sync call is inefficient.
                # In production, use a dedicated executor or redesign sync components.

            current_state = loop.run_until_complete(self._check_and_transition_state())
            
            if current_state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(self.service_name)

            try:
                result = func(*args, **kwargs)
                loop.run_until_complete(self._record_success())
                return result
            except Exception as e:
                loop.run_until_complete(self._record_failure())
                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

# --- Dead-Letter Queue (DLQ) Handler ---

class DeadLetterQueueHandler:
    """
    A simple in-memory Dead-Letter Queue handler for failed items (messages, tasks).
    In a production system, this would interact with an external message queue
    like Redis Streams, Kafka, or RabbitMQ, or a dedicated database table.
    """
    def __init__(self, max_dlq_size: int = 1000):
        if max_dlq_size < 0:
            raise ValueError("max_dlq_size cannot be negative")
        self._failed_items: deque[dict] = deque(maxlen=max_dlq_size)
        self.max_dlq_size = max_dlq_size
        logger.info(f"DeadLetterQueueHandler initialized with max size {max_dlq_size}.")

    def handle_failed_item(self, item: dict, reason: str = "unknown", original_error: Exception = None):
        """
        Adds a failed item to the dead-letter queue.
        
        Args:
            item (dict): The item that failed (e.g., a message payload, task details).
            reason (str): A description of why the item failed.
            original_error (Exception): The exception that caused the failure.
        """
        failed_entry = {
            "timestamp": time.time(),
            "item": item,
            "reason": reason,
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None,
            "stack_trace": traceback.format_exc() if original_error else None # Capture stack trace
        }
        self._failed_items.append(failed_entry)
        logger.error(f"Item sent to DLQ: Reason='{reason}', Item='{item}'. "
                     f"DLQ size: {len(self._failed_items)}/{self.max_dlq_size}")
        # In a real system, here you'd publish to a dedicated DLQ topic/queue.

    def get_failed_items(self) -> list[dict]:
        """Retrieves all items currently in the DLQ."""
        return list(self._failed_items)

    def clear_dlq(self):
        """Clears all items from the DLQ."""
        self._failed_items.clear()
        logger.info("DeadLetterQueueHandler cleared.")

    def __len__(self):
        return len(self._failed_items)

import traceback # Imported here to be used in DLQ handler

# Global instance for common usage if desired, or instantiate as needed.
dlq_handler = DeadLetterQueueHandler()
```