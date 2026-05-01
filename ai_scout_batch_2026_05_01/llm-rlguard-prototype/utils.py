import uuid
import datetime
import functools

def generate_unique_id() -> str:
    """Generates a unique identifier string."""
    return str(uuid.uuid4())

def get_timestamp(fmt: str = '%Y-%m-%d %H:%M:%S.%f') -> str:
    """
    Returns the current UTC timestamp as a formatted string.

    Args:
        fmt (str): The format string for datetime.strftime.
                   Defaults to '%Y-%m-%d %H:%M:%S.%f'.

    Returns:
        str: The formatted current UTC timestamp.
    """
    return datetime.datetime.utcnow().strftime(fmt)

def deep_get(obj: dict, keys: list, default=None):
    """
    Safely retrieves a value from a nested dictionary using a list of keys.

    Args:
        obj (dict): The dictionary to search within.
        keys (list): A list of keys representing the path to the desired value.
        default: The default value to return if the key path is not found.

    Returns:
        The value found at the specified path, or the default value if not found.
    """
    try:
        return functools.reduce(lambda d, key: d[key], keys, obj)
    except (KeyError, TypeError, IndexError):
        return default

def average_list(data: list) -> float:
    """
    Calculates the average of a list of numerical data.

    Args:
        data (list): A list of numbers (integers or floats).

    Returns:
        float: The average of the list, or 0.0 if the list is empty.
    """
    if not data:
        return 0.0
    if not all(isinstance(x, (int, float)) for x in data):
        raise TypeError("All elements in the list must be numbers (int or float).")
    return sum(data) / len(data)

def clamp(value, min_value, max_value):
    """
    Clamps a value within a specified range.

    Args:
        value: The value to clamp.
        min_value: The minimum allowed value.
        max_value: The maximum allowed value.

    Returns:
        The clamped value.
    """
    return max(min_value, min(value, max_value))<ctrl63>