import multiprocessing
import time
from typing import Callable, Any, Dict, Tuple, Optional

# Define a custom exception for clarity
class SandboxExecutionError(Exception):
    """Custom exception for errors during sandboxed execution."""
    pass

class SandboxExecutor:
    """
    Executes a given function within a separate process to provide isolation
    and enforce timeouts. This helps mitigate risks associated with LLM-driven
    tool execution by isolating critical or potentially risky actions.
    """
    def __init__(self, timeout: int = 60):
        """
        Initializes the SandboxExecutor.

        Args:
            timeout (int): The maximum time in seconds to allow the sandboxed
                           function to run before terminating it. Defaults to 60 seconds.
        """
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        self.timeout = timeout

    @staticmethod
    def _run_target_function(target_func: Callable[..., Any], args: Tuple, kwargs: Dict[str, Any], result_queue: multiprocessing.Queue):
        """
        Internal static method executed by the child process.
        It runs the target function and places its result or an error message
        into the provided multiprocessing.Queue.
        """
        try:
            result = target_func(*args, **kwargs)
            result_queue.put({'success': True, 'result': result})
        except Exception as e:
            # Capture any exception from the target function
            result_queue.put({'success': False, 'error': str(e), 'exception_type': type(e).__name__})

    def execute(self, target_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Executes a callable function in a sandboxed process with a defined timeout.

        Args:
            target_func (Callable[..., Any]): The function to execute. This function
                                               and its arguments must be picklable.
            *args: Positional arguments to pass to the target function.
            **kwargs: Keyword arguments to pass to the target function.

        Returns:
            Any: The return value of the target function.

        Raises:
            SandboxExecutionError: If the execution fails due to a timeout,
                                   an internal error in the sandboxed function,
                                   or unexpected process termination.
            ValueError: If the target_func is not callable.
        """
        if not callable(target_func):
            raise ValueError("target_func must be a callable function.")

        # Create a queue for inter-process communication
        result_queue = multiprocessing.Queue()
        
        # Create and start the child process
        process = multiprocessing.Process(
            target=SandboxExecutor._run_target_function,
            args=(target_func, args, kwargs, result_queue)
        )
        process.start()

        # Wait for the process to complete, with a timeout
        process.join(timeout=self.timeout)

        if process.is_alive():
            # If the process is still alive after join, it means it timed out
            process.terminate() # Send SIGTERM
            process.join(timeout=5) # Give it a little more time to clean up
            if process.is_alive():
                process.kill() # If still alive, send SIGKILL
                process.join()
            raise SandboxExecutionError(
                f"Sandbox execution timed out after {self.timeout} seconds. Process was terminated."
            )

        # Check the result queue for output from the child process
        if not result_queue.empty():
            result_data = result_queue.get()
            if result_data.get('success'):
                return result_data.get('result')
            else:
                # Re-raise the exception caught in the child process
                error_msg = result_data.get('error', 'An unknown error occurred in the sandboxed function.')
                exception_type = result_data.get('exception_type', 'Exception')
                raise SandboxExecutionError(
                    f"Sandboxed function failed: [{exception_type}] {error_msg}"
                )
        else:
            # This branch is reached if the process terminated but nothing was put in the queue.
            # This could indicate a crash before any result could be sent.
            exit_code_info = f" with exit code {process.exitcode}" if process.exitcode is not None else ""
            raise SandboxExecutionError(
                f"Sandboxed process terminated unexpectedly without returning a result or error{exit_code_info}. "
                "This may indicate a crash or an unhandled internal process error."
            )