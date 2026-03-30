import datetime

def add_numbers(num1: float, num2: float) -> float:
    """
    Adds two numbers and returns their sum.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        The sum of num1 and num2.
    """
    try:
        return num1 + num2
    except TypeError as e:
        raise ValueError(f"Invalid input types for add_numbers: {e}. Both arguments must be numbers.")

def get_current_utc_date() -> str:
    """
    Returns the current date in YYYY-MM-DD format (UTC).

    Returns:
        A string representing the current UTC date.
    """
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def reverse_string(text: str) -> str:
    """
    Reverses a given string.

    Args:
        text: The string to be reversed.

    Returns:
        The reversed string.
    """
    if not isinstance(text, str):
        raise TypeError("Input for reverse_string must be a string.")
    return text[::-1]

def get_string_length(text: str) -> int:
    """
    Returns the length of a given string.

    Args:
        text: The string for which to determine the length.

    Returns:
        An integer representing the length of the string.
    """
    if not isinstance(text, str):
        raise TypeError("Input for get_string_length must be a string.")
    return len(text)