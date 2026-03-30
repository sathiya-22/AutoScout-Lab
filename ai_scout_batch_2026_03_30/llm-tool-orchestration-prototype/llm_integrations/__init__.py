# Expose key components from sub-modules for easier import.
# This makes it possible to do:
# from llm_integrations import LLMClient, ResponseParser, PromptManager, get_prompt_template

# Initialize components to None; they will be populated if imports are successful.
LLMClient = None
ResponseParser = None
parse_llm_response = None
PromptManager = None
get_prompt_template = None

# Attempt to import core components.
# If an import fails, the corresponding variable remains None.
# This approach provides basic error handling by allowing the package to be imported
# even if a sub-module has issues, while critical dependencies will cause downstream
# failures when the `None` object is attempted to be used.

try:
    from .llm_client import LLMClient
except ImportError:
    pass # LLMClient will remain None if the import fails
except Exception:
    pass # Catch other potential errors during import

try:
    from .response_parser import ResponseParser, parse_llm_response
except ImportError:
    pass # ResponseParser/parse_llm_response will remain None
except Exception:
    pass

try:
    from .prompt_templates import PromptManager, get_prompt_template
except ImportError:
    pass # PromptManager/get_prompt_template will remain None
except Exception:
    pass

# Define what is exposed when `from llm_integrations import *` is used.
# Only include components that were successfully imported (i.e., are not None).
__all__ = [
    name for name, obj in locals().items()
    if not name.startswith('_') and obj is not None
]