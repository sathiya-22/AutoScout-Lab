```python
import abc
import re
import json
from typing import Any, Callable, List, Optional, Tuple, Type, Union

# Attempt to import Pydantic components.
# This makes PydanticSchemaValidator optional, allowing the module to be imported
# even if Pydantic is not installed, but the Pydantic validator will not function.
try:
    from pydantic import BaseModel, ValidationError, Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

    # Dummy classes if Pydantic is not installed, to allow type hinting and module import.
    class BaseModel:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Pydantic is not installed. This BaseModel is a dummy.")

        @classmethod
        def model_validate(cls, data: dict):
            raise NotImplementedError("Pydantic is not installed.")
            
        @classmethod
        def parse_obj(cls, data: dict):
            raise NotImplementedError("Pydantic is not installed.")

        def dict(self, *args, **kwargs):
            raise NotImplementedError("Pydantic is not installed.")

    class ValidationError(Exception):
        pass # Dummy ValidationError for type hinting

    class Field:
        def __init__(self, *args, **kwargs):
            pass


class Validator(abc.ABC):
    """Abstract base class for all LLM output validators."""

    @abc.abstractmethod
    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Validates the given LLM output string.

        Args:
            output: The LLM output string to validate.

        Returns:
            A tuple containing:
            - bool: True if the output is valid, False otherwise.
            - Optional[str]: An error message if invalid, None otherwise.
            - Optional[Any]: The parsed/validated data if successful (e.g., Pydantic model
                             instance for PydanticSchemaValidator), None otherwise.
                             This allows the correction layer to use the validated structure directly.
        """
        pass


class PydanticSchemaValidator(Validator):
    """
    Validates an LLM output against a Pydantic schema.
    Assumes the output is a JSON string.
    Supports Pydantic V1 and V2 by checking for 'model_validate' (V2) then 'parse_obj' (V1).
    """
    def __init__(self, schema_model: Type[BaseModel]):
        if not _HAS_PYDANTIC:
            raise ImportError(
                "Pydantic is not installed. PydanticSchemaValidator cannot be used. "
                "Please install Pydantic (e.g., `pip install pydantic`)."
            )
        if not issubclass(schema_model, BaseModel):
            raise TypeError("schema_model must be a Pydantic BaseModel subclass.")
        self.schema_model = schema_model

    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        if not _HAS_PYDANTIC:
            return False, "Pydantic is not installed. PydanticSchemaValidator cannot validate.", None

        try:
            parsed_data = json.loads(output)
            
            # Attempt Pydantic V2 validation first, then V1
            if hasattr(self.schema_model, 'model_validate'): # Pydantic V2
                validated_output = self.schema_model.model_validate(parsed_data)
            elif hasattr(self.schema_model, 'parse_obj'): # Pydantic V1
                validated_output = self.schema_model.parse_obj(parsed_data)
            else:
                return False, (
                    f"Pydantic schema model {self.schema_model.__name__} does not have "
                    f"'model_validate' (Pydantic V2) or 'parse_obj' (Pydantic V1) method."
                ), None

            return True, None, validated_output
        except json.JSONDecodeError as e:
            return False, f"Output is not valid JSON: {e}", None
        except ValidationError as e:
            return False, f"Output does not conform to Pydantic schema {self.schema_model.__name__}: {e}", None
        except Exception as e:
            # Catch other potential errors during parsing or validation (e.g., incompatible types before ValidationError)
            return False, (
                f"An unexpected error occurred during Pydantic validation for "
                f"{self.schema_model.__name__}: {type(e).__name__} - {e}"
            ), None


class RegexValidator(Validator):
    """
    Validates an LLM output against a regular expression pattern.
    Checks if the entire output string fully matches the pattern using `re.fullmatch`.
    """
    def __init__(self, pattern: str, flags: Union[int, re.RegexFlag] = 0):
        self.pattern = pattern
        try:
            self.regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern provided: {e}") from e

    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        if self.regex.fullmatch(output):
            return True, None, output
        else:
            return False, f"Output does not fully match regex pattern: '{self.pattern}'", None


class LengthValidator(Validator):
    """
    Validates the length of an LLM output string.
    """
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be provided.")
        if min_length is not None and min_length < 0:
            raise ValueError("min_length cannot be negative.")
        if max_length is not None and max_length < 0:
            raise ValueError("max_length cannot be negative.")
        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length.")

        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        output_len = len(output)
        if self.min_length is not None and output_len < self.min_length:
            return False, (
                f"Output length ({output_len}) is less than minimum required length "
                f"({self.min_length}). Expected >= {self.min_length} characters."
            ), None
        if self.max_length is not None and output_len > self.max_length:
            return False, (
                f"Output length ({output_len}) exceeds maximum allowed length "
                f"({self.max_length}). Expected <= {self.max_length} characters."
            ), None
        return True, None, output


class KeywordPresenceValidator(Validator):
    """
    Validates an LLM output based on the presence or absence of specific keywords.
    Keywords can be checked case-sensitively or case-insensitively.
    """
    def __init__(self, required_keywords: Optional[List[str]] = None, forbidden_keywords: Optional[List[str]] = None, case_sensitive: bool = False):
        if not required_keywords and not forbidden_keywords:
            raise ValueError("At least one of required_keywords or forbidden_keywords must be provided.")
        self.required_keywords = required_keywords or []
        self.forbidden_keywords = forbidden_keywords or []
        self.case_sensitive = case_sensitive

    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        processed_output = output if self.case_sensitive else output.lower()
        
        for keyword in self.required_keywords:
            processed_keyword = keyword if self.case_sensitive else keyword.lower()
            if processed_keyword not in processed_output:
                return False, f"Required keyword '{keyword}' not found in output.", None

        for keyword in self.forbidden_keywords:
            processed_keyword = keyword if self.case_sensitive else keyword.lower()
            if processed_keyword in processed_output:
                return False, f"Forbidden keyword '{keyword}' found in output.", None
        
        return True, None, output


class CustomFunctionValidator(Validator):
    """
    Validates an LLM output using a custom Python function.
    The function should take the output string as input and return a boolean.
    """
    def __init__(self, validation_function: Callable[[str], bool], error_message: str = "Custom validation failed."):
        if not callable(validation_function):
            raise TypeError("validation_function must be a callable.")
        self.validation_function = validation_function
        self.error_message = error_message

    def validate(self, output: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        try:
            is_valid = self.validation_function(output)
            if is_valid:
                return True, None, output
            else:
                return False, self.error_message, None
        except Exception as e:
            return False, f"Custom validation function raised an error: {type(e).__name__} - {e}", None

```