import asyncio
import logging
from collections import defaultdict
from typing import Callable, Any, Dict, List

# It's assumed that logger_config.py will set up the logging handlers and format.
# For standalone testing or if logger_config.py isn't loaded yet,
# a basic configuration can prevent "No handlers could be found for logger" warnings.
try:
    from monitoring_debugging.logger_config import get_logger
    logger = get_logger("EventBus")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("EventBus")
    logger.warning("Could not import get_logger from monitoring_debugging.logger_config. Using default logging.")


class EventBus:
    """
    A central asynchronous message passing system for all inter-agent communication,
    system events, and monitoring hooks within the Agent Orchestration Control Framework.
    Implemented as a singleton to ensure a single, consistent event channel.
    """
    _instance: 'EventBus' = None
    _lock: asyncio.Lock
    _subscribers: Dict[str, List[Callable[..., Any]]]
    _logger: logging.Logger

    def __new__(cls):
        """
        Ensures that only one instance of EventBus is created (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            # Initialize instance-specific attributes only once
            cls._instance._subscribers = defaultdict(list)
            cls._instance._lock = asyncio.Lock()
            cls._instance._logger = logger # Use the module-level logger
            cls._instance._logger.info("EventBus initialized (singleton instance created).")
        return cls._instance

    async def subscribe(self, event_type: str, callback: Callable[..., Any]):
        """
        Subscribes a callback function to a specific event type.
        The callback will be called with the event data when the event is published.

        Args:
            event_type (str): The unique identifier for the event type.
            callback (Callable): The asynchronous or synchronous function to call
                                  when the event is published. It should accept
                                  the event data as arguments (*args, **kwargs).
        """
        async with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                self._logger.debug(f"Subscriber registered for '{event_type}': {callback.__name__}")
            else:
                self._logger.warning(f"Callback '{callback.__name__}' is already subscribed to event type '{event_type}'.")

    async def unsubscribe(self, event_type: str, callback: Callable[..., Any]):
        """
        Unsubscribes a specific callback function from an event type.

        Args:
            event_type (str): The unique identifier for the event type.
            callback (Callable): The function to remove from subscriptions.
        """
        async with self._lock:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                self._logger.debug(f"Subscriber '{callback.__name__}' unregistered from '{event_type}'.")
            else:
                self._logger.warning(f"Callback '{callback.__name__}' not found for event type '{event_type}'. Cannot unsubscribe.")

    async def publish(self, event_type: str, *args, **kwargs):
        """
        Publishes an event to all subscribed callbacks for the given event type.
        Each subscriber's callback is executed as an independent asyncio task,
        ensuring non-blocking and concurrent processing.

        Args:
            event_type (str): The unique identifier for the event type being published.
            *args: Positional arguments to pass to the subscriber callbacks.
            **kwargs: Keyword arguments to pass to the subscriber callbacks.
                      These typically represent the event data/payload.
        """
        self._logger.info(f"Publishing event '{event_type}' with data: args={args}, kwargs={kwargs}")

        async with self._lock:
            # Get a copy of the subscribers list to prevent issues if subscribers
            # modify their own subscriptions during event processing.
            subscribers_for_event = list(self._subscribers.get(event_type, []))

        if not subscribers_for_event:
            self._logger.debug(f"No subscribers found for event type '{event_type}'. Event not processed by any handler.")
            return

        # Create tasks for each subscriber to run concurrently
        tasks = []
        for callback in subscribers_for_event:
            task = asyncio.create_task(self._run_subscriber_safely(callback, event_type, *args, **kwargs))
            tasks.append(task)
        
        # Note: We do not await these tasks here. This makes `publish` a "fire-and-forget"
        # operation, allowing the publisher to continue immediately without waiting
        # for all subscribers to process the event. For scenarios where the publisher
        # needs to know if all handlers completed or needs results, different
        # patterns (e.g., returning futures or explicit awaiting) would be needed.

    async def _run_subscriber_safely(self, callback: Callable[..., Any], event_type: str, *args, **kwargs):
        """
        Internal method to safely execute a subscriber's callback.
        Handles exceptions in individual callbacks to prevent cascading failures
        and logs the outcome.
        """
        try:
            # Check if the callback is an awaitable coroutine function
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                # If it's a regular function, execute it directly.
                # In an asyncio context, it's generally better to `run_in_executor`
                # for CPU-bound synchronous tasks, but for simple callbacks, direct
                # execution is often acceptable if they are not blocking.
                callback(*args, **kwargs)
            self._logger.debug(f"Successfully executed subscriber '{callback.__name__}' for event '{event_type}'.")
        except Exception as e:
            self._logger.error(
                f"Error executing subscriber '{callback.__name__}' for event '{event_type}': {e}",
                exc_info=True # Include traceback for debugging
            )


def get_event_bus() -> EventBus:
    """
    Convenience function to get the singleton instance of the EventBus.
    """
    return EventBus()