```python
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, TYPE_CHECKING

# To avoid circular imports, StateManager is not imported directly at the top.
# Type checking can use a forward reference or ignore the import for runtime.
if TYPE_CHECKING:
    from monitor.state_manager import StateManager # Only for type hints

_F = TypeVar('_F', bound=Callable[..., Any])

class Instrumentation:
    """
    Provides decorators to instrument AI agent methods, capturing internal states
    and external interactions, and forwarding these observations to a StateManager.

    This class uses a class-level attribute to store the StateManager instance,
    allowing decorators to access it without needing an explicit instance of
    Instrumentation in every decorated agent method. The MonitorCore is
    responsible for setting the StateManager instance.
    """
    _state_manager_instance: Optional["StateManager"] = None

    @classmethod
    def set_state_manager(cls, state_manager: "StateManager") -> None:
        """
        Sets the global StateManager instance that all decorators will use.
        This must be called once by the MonitorCore during initialization.
        """
        if cls._state_manager_instance is not None and cls._state_manager_instance is not state_manager:
            # In a production system, one might log a warning here if the StateManager
            # is being reset, as it usually indicates a misconfiguration.
            pass
        cls._state_manager_instance = state_manager

    @classmethod
    def _get_state_manager(cls) -> "StateManager":
        """
        Retrieves the globally set StateManager instance.
        Raises a RuntimeError if the StateManager has not been set.
        """
        if cls._state_manager_instance is None:
            raise RuntimeError(
                "StateManager has not been set for Instrumentation. "
                "Call Instrumentation.set_state_manager(state_manager) first, "
                "typically in monitor/core.py."
            )
        return cls._state_manager_instance

    @staticmethod
    def instrument_thought(func: _F) -> _F:
        """
        Decorator to capture an agent's generated thought.
        Assumes the decorated function returns the thought string.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state_manager = Instrumentation._get_state_manager()
            try:
                thought = func(*args, **kwargs)
                state_manager.record_observation("thought_generated", {"thought": thought, "timestamp": time.time()})
                return thought
            except Exception as e:
                state_manager.record_observation("thought_generation_error", {"error": str(e), "timestamp": time.time()})
                raise # Re-raise the original exception
        return wrapper # type: ignore

    @staticmethod
    def instrument_tool_call(func: _F) -> _F:
        """
        Decorator to capture an agent's tool call initiation and its result.
        Assumes the decorated function takes (self, tool_name, tool_args, ...) as arguments
        and returns the tool's result. Handles both success and failure.
        """
        @wraps(func)
        def wrapper(agent_instance: Any, tool_name: str, tool_args: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            state_manager = Instrumentation._get_state_manager()
            tool_result = None
            status = "success"
            error: Optional[str] = None
            start_time = time.time()

            try:
                tool_result = func(agent_instance, tool_name, tool_args, *args, **kwargs)
            except Exception as e:
                status = "failure"
                error = str(e)
                raise # Re-raise the exception after logging
            finally:
                end_time = time.time()
                duration = end_time - start_time
                state_manager.record_observation("tool_called", {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result, # Will be None on failure
                    "status": status,
                    "error": error,
                    "duration": duration,
                    "timestamp": time.time()
                })
            return tool_result
        return wrapper # type: ignore

    @staticmethod
    def instrument_output(func: _F) -> _F:
        """
        Decorator to capture an agent's final produced output.
        Assumes the decorated function returns the final output.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state_manager = Instrumentation._get_state_manager()
            try:
                final_output = func(*args, **kwargs)
                state_manager.record_observation("final_output_produced", {"output": final_output, "timestamp": time.time()})
                return final_output
            except Exception as e:
                state_manager.record_observation("output_production_error", {"error": str(e), "timestamp": time.time()})
                raise # Re-raise the original exception
        return wrapper # type: ignore

    @classmethod
    def instrument_agent_intervention_receiver(cls, intervention_type: str) -> Callable[[_F], _F]:
        """
        Decorator factory to capture when an agent receives an intervention.
        The `intervention_type` should be a descriptive string (e.g., "replan", "receive_hint").
        """
        def decorator(func: _F) -> _F:
            @wraps(func)
            def wrapper(agent_instance: Any, *args: Any, **kwargs: Any) -> Any:
                state_manager = Instrumentation._get_state_manager()
                
                # Capture arguments passed to the intervention method, excluding 'self' (agent_instance)
                # Ensure args and kwargs are JSON-serializable for the state manager's storage.
                intervention_payload = {
                    "method": func.__name__,
                    "intervention_type": intervention_type,
                    "args": list(args), # Convert tuple to list for potential modification/serialization
                    "kwargs": kwargs,
                    "timestamp": time.time()
                }
                
                state_manager.record_observation(f"agent_received_intervention", intervention_payload)
                return func(agent_instance, *args, **kwargs)
            return wrapper # type: ignore
        return decorator
```