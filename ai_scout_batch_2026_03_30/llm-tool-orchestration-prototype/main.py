import logging
import os
from typing import Dict, Any

# Assume relative imports based on the described file structure
from config.config import Config, load_config
from llm_integrations.llm_client import LLMClient
from llm_integrations.prompt_templates import PromptTemplates
from tools.tool_registry import ToolRegistry
from tools.sandbox_executor import SandboxExecutor
from validation.schema_validator import SchemaValidator
from validation.guardrails import Guardrails
from agents.verifier_agent import VerifierAgent
from agents.tool_executor_agent import ToolExecutorAgent
from agents.primary_orchestrator import PrimaryOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_system() -> Dict[str, Any]:
    """Initializes all components of the hierarchical agentic system."""
    logger.info("Initializing system components...")
    
    try:
        # 1. Configuration
        config: Config = load_config(os.getenv("CONFIG_PATH", "config/default_config.yaml"))
        
        # 2. LLM Integration Layer
        llm_client = LLMClient(config=config)
        
        # 3. Advanced Prompt Engineering
        prompt_templates = PromptTemplates(config=config) # Pass config if prompt paths/settings are there
        
        # 4. Tool Definitions and Registry
        tool_registry = ToolRegistry(tool_definitions_path=config.get_tool_definitions_path())
        
        # 5. Schema Validation
        schema_validator = SchemaValidator(tool_registry=tool_registry)
        
        # 6. Guardrails
        guardrails = Guardrails(config=config) # Pass config for guardrail rules/thresholds
        
        # 7. Sandboxed Execution
        sandbox_executor = SandboxExecutor(config=config)
        
        # 8. Verifier Agent
        verifier_agent = VerifierAgent(
            llm_client=llm_client,
            guardrails=guardrails,
            prompt_templates=prompt_templates,
            config=config
        )
        
        # 9. Tool Executor Agent
        tool_executor_agent = ToolExecutorAgent(
            llm_client=llm_client,
            verifier_agent=verifier_agent,
            tool_registry=tool_registry,
            schema_validator=schema_validator,
            sandbox_executor=sandbox_executor,
            prompt_templates=prompt_templates,
            config=config
        )
        
        # 10. Orchestrator Agent
        primary_orchestrator = PrimaryOrchestrator(
            llm_client=llm_client,
            tool_executor_agent=tool_executor_agent,
            verifier_agent=verifier_agent, # Orchestrator might also directly use verifier for high-level checks
            prompt_templates=prompt_templates,
            config=config
        )
        
        logger.info("System initialization complete.")
        return {
            "primary_orchestrator": primary_orchestrator,
            "config": config
        }
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise

def main():
    """Main function to run the Hierarchical Agentic System."""
    system_components = {}
    try:
        system_components = initialize_system()
        orchestrator: PrimaryOrchestrator = system_components["primary_orchestrator"]
        
        print("\n--- Hierarchical Agentic System ---")
        print("Enter 'exit' to quit.")
        
        while True:
            user_query = input("\nUser Query: ")
            if user_query.lower() == 'exit':
                logger.info("Exiting application.")
                break
            if not user_query.strip():
                print("Please enter a query.")
                continue

            try:
                logger.info(f"Processing user query: '{user_query}'")
                final_result = orchestrator.run(user_query)
                print(f"\nOrchestrator Final Result:\n{final_result}")
                logger.info("Query processing complete.")
            except ValueError as ve:
                print(f"\nError processing query: {ve}")
                logger.warning(f"Validation or processing error for query '{user_query}': {ve}")
            except Exception as e:
                print(f"\nAn unexpected error occurred during query processing: {e}")
                logger.error(f"Unexpected error during query '{user_query}': {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Application failed to start or encountered a critical error: {e}", exc_info=True)
        print(f"\nCritical system error: {e}. Please check logs for details.")

if __name__ == "__main__":
    main()