import datetime
import time
import os

class ToolError(Exception):
    """Custom exception for errors encountered by tools."""
    pass

def search_web(query: str) -> str:
    """
    Simulates a web search for the given query.
    Returns a mock search result string.
    """
    # print(f"Tool Call: search_web(query='{query}')")
    time.sleep(0.5) # Simulate network latency
    if "current events" in query.lower():
        return f"Mock search result for '{query}': 'Global AI summit concludes with focus on ethical development and sustainability.'"
    elif "programming languages" in query.lower():
        return f"Mock search result for '{query}': 'Python, JavaScript, and Rust continue to be top choices for developers.'"
    elif "weather in new york" in query.lower():
        return f"Mock search result for '{query}': 'New York City weather: Partly cloudy, 68°F (20°C).'"
    elif "non_existent_topic" in query.lower():
        raise ToolError(f"Simulated error: Could not find any relevant results for '{query}'.")
    else:
        return f"Mock search result for '{query}': 'Information regarding {query} found from various online sources.'"

def read_file(filename: str) -> str:
    """
    Simulates reading content from a file.
    In a real system, this would interact with the file system.
    For this prototype, it returns mock content or raises an error.
    """
    # print(f"Tool Call: read_file(filename='{filename}')")
    time.sleep(0.2) # Simulate file system access latency
    mock_files = {
        "report.txt": "This is a mock report content. It details the initial progress on the meta-monitoring project, highlighting phase 1 completion.",
        "config.json": '{"api_key_status": "active", "monitor_interval_seconds": 10, "max_log_entries": 100}',
        "notes.md": "# Project Notes\n- Initial brainstorming complete\n- Setup core architecture components\n- Next steps: Implement detectors and interventions.",
        "empty.txt": ""
    }
    
    if filename in mock_files:
        return mock_files[filename]
    elif filename.startswith("non_existent_"):
        raise ToolError(f"Error: File '{filename}' not found on the simulated file system.")
    else:
        # Simulate dynamically created file content for other filenames
        return f"Mock content for file '{filename}'. This file was supposedly created by an agent action."

def write_file(filename: str, content: str) -> bool:
    """
    Simulates writing content to a file.
    Returns True on success, raises ToolError on simulated failure.
    """
    # print(f"Tool Call: write_file(filename='{filename}', content_length={len(content)})")
    time.sleep(0.3) # Simulate file system write latency
    
    if "permission_denied" in filename.lower():
        raise ToolError(f"Error: Permission denied for file '{filename}'. Cannot write content.")
    elif len(content) > 2000: # Simulate a maximum content size limit
        raise ToolError(f"Error: Content too large for file '{filename}'. Max 2000 characters allowed.")
    elif "critical_failure" in content.lower():
        raise ToolError(f"Simulated critical failure during write operation for '{filename}'.")
    else:
        # In a real system, you would actually write to a file here.
        # For the prototype, we just acknowledge the simulated write.
        # print(f"Tool Info: Successfully simulated writing to '{filename}'.")
        return True

def get_current_time() -> str:
    """
    Returns the current date and time as a string.
    """
    # print("Tool Call: get_current_time()")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """
    Simulates a calculator tool that evaluates a mathematical expression.
    Handles basic arithmetic.
    """
    # print(f"Tool Call: calculate(expression='{expression}')")
    time.sleep(0.1) # Simulate calculation time
    try:
        # Using eval is generally unsafe for arbitrary untrusted input in production.
        # For a controlled prototype demonstrating basic functionality, it's used here.
        # In a real system, a safer math expression parser would be preferred.
        result = eval(expression, {"__builtins__": None}, {}) # Limited scope for safety
        return str(result)
    except (SyntaxError, TypeError, NameError, ZeroDivisionError) as e:
        raise ToolError(f"Error calculating '{expression}': Invalid expression or operation. Details: {e}")
    except Exception as e:
        raise ToolError(f"An unexpected error occurred during calculation of '{expression}': {e}")


# A dictionary to easily access tools by name for agent use
TOOL_REGISTRY = {
    "search_web": search_web,
    "read_file": read_file,
    "write_file": write_file,
    "get_current_time": get_current_time,
    "calculate": calculate,
}

# Example of how an agent might describe available tools (for LLM context)
TOOL_DESCRIPTIONS = {
    "search_web": "Searches the internet for information based on a query. Returns relevant text snippets.",
    "read_file": "Reads the content of a specified file. Returns the file's content as a string.",
    "write_file": "Writes the given content to a specified file. Returns True on success, raises an error on failure (e.g., permission denied).",
    "get_current_time": "Retrieves the current date and time. Returns a string in 'YYYY-MM-DD HH:MM:SS' format.",
    "calculate": "Evaluates a mathematical expression and returns the result as a string. Supports basic arithmetic operations.",
}