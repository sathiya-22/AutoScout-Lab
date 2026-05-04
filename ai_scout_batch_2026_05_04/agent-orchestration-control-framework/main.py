```python
import argparse
import logging
import os
import sys
import yaml

# Add the project root to the Python path to allow absolute imports
# This assumes main.py is in the project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Core Framework Imports
from core.event_bus import EventBus
from core.state_manager import StateManager
from agents.tool_registry import ToolRegistry
from core.agent_lifecycle import AgentLifecycle
from core.orchestrator import Orchestrator

# Specification & Control Imports
from specifications.dsl_parser import DSLParser
from specifications.constraint_engine import ConstraintEngine
from specifications.interaction_spec_validator import InteractionSpecValidator

# Configuration Imports
from config.settings import AppSettings
from monitoring_debugging.logger_config import setup_logging

# Example Imports
from examples import simple_workflow
from examples import finance_approval_process

# Define the root directory of the project for easier path management
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_application_settings() -> AppSettings:
    """
    Loads application settings from config/settings.py.
    """
    try:
        settings = AppSettings()
        return settings
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load application settings. Ensure config/settings.py is valid and environment variables are correctly set if used. Details: {e}")
        sys.exit(1)

def load_agent_configurations(settings: AppSettings) -> dict:
    """
    Loads agent configurations from the YAML file specified in settings.
    """
    config_path = os.path.join(PROJECT_ROOT, settings.AGENT_CONFIG_PATH)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Agent configuration file not found at '{config_path}'. Please ensure it exists.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"CRITICAL ERROR: Failed to parse agent configuration file '{config_path}'. Please check YAML syntax. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred while loading agent configurations from '{config_path}'. Details: {e}")
        sys.exit(1)

def load_formal_specification(settings: AppSettings, spec_filename: str) -> tuple[dict | None, str | None]:
    """
    Loads and parses a formal specification file using the DSLParser.
    Returns the parsed specification and its raw content, or (None, None) if not found/parsed.
    """
    if not spec_filename:
        logging.getLogger(__name__).info("No formal specification filename provided. Skipping formal spec loading.")
        return None, None

    spec_path = os.path.join(PROJECT_ROOT, settings.SPECIFICATIONS_DIR, spec_filename)
    if not os.path.exists(spec_path):
        logging.getLogger(__name__).warning(f"Formal specification file not found at '{spec_path}'. Constraints based on this spec will not be enforced.")
        return None, None
    
    try:
        with open(spec_path, 'r') as f:
            dsl_content = f.read()
        
        parser = DSLParser()
        parsed_spec = parser.parse(dsl_content)
        logging.getLogger(__name__).info(f"Formal specification '{spec_filename}' loaded and parsed successfully.")
        return parsed_spec, dsl_content
    except FileNotFoundError: # Redundant due to os.path.exists check, but good for robustness
        logging.getLogger(__name__).error(f"Formal specification file not found at '{spec_path}'. This should have been caught earlier.")
        return None, None
    except Exception as e:
        logging.getLogger(__name__).error(f"Error parsing formal specification '{spec_path}'. Check DSL syntax. Details: {e}", exc_info=True)
        return None, None

def initialize_framework_components(
    settings: AppSettings,
    agent_configs: dict,
    formal_spec_parsed: dict | None
) -> dict:
    """
    Initializes all core components of the orchestration framework.
    Returns a dictionary of initialized components.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing Agent Orchestration Control Framework components...")

    try:
        # Core communication and state
        event_bus = EventBus()
        state_manager = StateManager(event_bus=event_bus)
        tool_registry = ToolRegistry()

        # Agent lifecycle management
        agent_lifecycle = AgentLifecycle(
            event_bus=event_bus,
            state_manager=state_manager,
            tool_registry=tool_registry,
            agent_configs=agent_configs # Pass agent configurations for agent creation
        )

        # Specification enforcement components
        constraint_engine = None
        interaction_spec_validator = None
        if formal_spec_parsed:
            constraint_engine = ConstraintEngine(formal_spec_parsed)
            interaction_spec_validator = InteractionSpecValidator(formal_spec_parsed)
            logger.info("Formal specification engines (ConstraintEngine, InteractionSpecValidator) initialized.")
        else:
            logger.info("No formal specification loaded. Constraint and interaction validation will be skipped.")

        # The central orchestrator
        orchestrator = Orchestrator(
            event_bus=event_bus,
            state_manager=state_manager,
            agent_lifecycle=agent_lifecycle,
            constraint_engine=constraint_engine,
            interaction_spec_validator=interaction_spec_validator,
            settings=settings
        )

        logger.info("Framework components initialized successfully.")
        return {
            "event_bus": event_bus,
            "state_manager": state_manager,
            "tool_registry": tool_registry,
            "agent_lifecycle": agent_lifecycle,
            "orchestrator": orchestrator,
            "constraint_engine": constraint_engine,
            "interaction_spec_validator": interaction_spec_validator,
            "settings": settings,
            "agent_configs": agent_configs # Also pass configs for examples if they need raw access
        }
    except Exception as e:
        logger.critical(f"Failed to initialize framework components due to an unexpected error: {e}", exc_info=True)
        raise # Re-raise to be caught by main's error handling

def run_selected_example(example_name: str, framework_components: dict):
    """
    Runs the specified example workflow using the initialized framework components.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to run example: '{example_name}'")

    try:
        if example_name == "simple_workflow":
            simple_workflow.run(framework_components)
        elif example_name == "finance_approval_process":
            # The finance example might explicitly require a spec,
            # so we ensure it's handled or log a warning if not present.
            if framework_components["constraint_engine"] is None:
                logger.warning("Running 'finance_approval_process' without a loaded formal specification. "
                               "Constraint enforcement will be skipped.")
            finance_approval_process.run(framework_components)
        else:
            logger.error(f"Unknown example specified: '{example_name}'.")
            print(f"Error: Unknown example '{example_name}'. Available examples: 'simple_workflow', 'finance_approval_process'.")
            sys.exit(1)
        
        logger.info(f"Example '{example_name}' completed successfully.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during example '{example_name}' execution: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: A problem occurred while running example '{example_name}'. Please check the logs for detailed information.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="""
        Agent Orchestration Control Framework Entry Point.
        This script initializes the core framework components and runs a selected example workflow.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "example",
        type=str,
        help="""Specify which example to run:
  simple_workflow         - A basic demonstration of agent interaction.
  finance_approval_process - A more elaborate example showcasing formal specifications and meta-agent control."""
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="example_spec_finance.fsl", # Default to a finance spec as it's common for complex examples
        help="""Optional: Specify a formal specification file (e.g., 'example_spec_finance.fsl').
  The file should be located in the 'specifications/' directory.
  If not provided or file not found, constraint and interaction validation will be skipped."""
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for console and file output (default: INFO)."
    )

    args = parser.parse_args()

    # --- 1. Setup Logging ---
    main_logger = setup_logging(log_level=args.log_level)
    main_logger.info("Starting Agent Orchestration Control Framework...")

    # --- 2. Load Application Settings ---
    settings = load_application_settings()
    main_logger.debug(f"Application settings loaded: {settings.dict()}")

    # --- 3. Load Agent Configurations ---
    agent_configs = load_agent_configurations(settings)
    main_logger.debug(f"Agent configurations loaded for agents: {list(agent_configs.keys())}")

    # --- 4. Load Formal Specification (if requested) ---
    formal_spec_parsed = None
    if args.spec:
        formal_spec_parsed, _ = load_formal_specification(settings, args.spec)
    else:
        main_logger.info("No formal specification file specified via --spec. Continuing without formal constraints.")

    # --- 5. Initialize Framework Components ---
    try:
        framework_components = initialize_framework_components(settings, agent_configs, formal_spec_parsed)
    except Exception:
        # initialize_framework_components already logs critical error and re-raises
        print("\nFramework failed to initialize. Please check the logs for detailed errors.")
        sys.exit(1)

    # --- 6. Run the specified example ---
    run_selected_example(args.example, framework_components)

    main_logger.info("Agent Orchestration Control Framework finished execution.")
    print("\nOrchestration run completed. Check 'logs/framework.log' for details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any unexpected top-level errors not handled elsewhere
        print(f"\nAN UNEXPECTED FATAL ERROR OCCURRED: {e}")
        print("Please check the log files (logs/framework.log) for more details.")
        sys.exit(1)
```