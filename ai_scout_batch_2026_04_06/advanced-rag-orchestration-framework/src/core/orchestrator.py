import logging
from typing import List, Dict, Any, Type, Callable, Optional, Union

# Assume these are defined in src/core/interfaces.py
from src.core.interfaces import (
    OrchestrationModule,
    QueryProcessor,
    Retriever,
    Reranker,
    EvidenceCompressor,
    Generator,
    AgenticModule,
)

# Assume these are defined in src/core/models.py
from src.core.models import (
    Query,
    Document,
    RetrievalResult,  # This might become part of Context directly or for internal use
    Context,
    PipelineResult,
    GeneratedResponse,
)

logger = logging.getLogger(__name__)

# Basic logging configuration if not already set up by main.py
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PipelineStep:
    """
    Encapsulates a single stage in the RAG pipeline, holding an
    OrchestrationModule instance and defining which method to call.
    """
    def __init__(self, name: str, module: OrchestrationModule, method_name: Optional[str] = None):
        self.name = name
        self.module = module
        self.method_name = method_name or self._infer_method_name(module)

        if not hasattr(self.module, self.method_name):
            raise AttributeError(
                f"Module '{self.name}' (type: {type(self.module).__name__}) "
                f"does not have the expected method '{self.method_name}'."
            )
        logger.debug(f"PipelineStep '{self.name}' initialized with module {type(module).__name__} "
                     f"and method '{self.method_name}'.")

    def _infer_method_name(self, module: OrchestrationModule) -> str:
        """
        Infers the primary execution method name based on the module's type.
        This assumes a standardized naming convention for module interfaces.
        """
        if isinstance(module, QueryProcessor):
            return "process_query"
        elif isinstance(module, Retriever):
            return "retrieve"
        elif isinstance(module, Reranker):
            return "rerank"
        elif isinstance(module, EvidenceCompressor):
            return "compress_evidence"
        elif isinstance(module, Generator):
            return "generate"
        elif isinstance(module, AgenticModule):
            # Agentic modules might have a more complex 'execute' or 'run_agent' method
            # that might involve internal loops or tool calls.
            return "execute_agentic_loop"
        else:
            raise ValueError(f"Cannot infer method name for unknown module type: {type(module).__name__}. "
                             f"Please provide `method_name` explicitly.")

    def execute(self, current_context: Context) -> Context:
        """
        Executes the designated method of the encapsulated module.
        Each module's method is expected to take a Context object and return
        an updated Context object.
        """
        logger.info(f"Executing step '{self.name}' with method '{self.method_name}'...")
        try:
            method = getattr(self.module, self.method_name)
            updated_context = method(current_context)

            if not isinstance(updated_context, Context):
                raise TypeError(
                    f"Module '{self.name}' method '{self.method_name}' "
                    f"returned type {type(updated_context).__name__}, expected a Context object."
                )
            logger.debug(f"Step '{self.name}' completed. Context updated.")
            return updated_context
        except Exception as e:
            logger.error(f"Error during execution of step '{self.name}' (method: '{self.method_name}'): {e}", exc_info=True)
            raise # Re-raise to allow the orchestrator to catch and handle.


class RAGOrchestrator:
    """
    The central control plane for defining, orchestrating, and executing
    multi-stage RAG pipelines. It manages the flow of data (via the Context object)
    between pluggable modules and provides a programmatic interface for pipeline construction.
    """
    def __init__(self, pipeline_name: str = "Default RAG Pipeline"):
        self.pipeline_name = pipeline_name
        self._pipeline_steps: List[PipelineStep] = []
        logger.info(f"RAG Orchestrator '{pipeline_name}' initialized.")

    def add_step(self, name: str, module: OrchestrationModule, method_name: Optional[str] = None) -> 'RAGOrchestrator':
        """
        Adds a module as a named step to the RAG pipeline.

        Args:
            name (str): A unique identifier for this pipeline step (e.g., "query_translation", "vector_retrieval").
            module (OrchestrationModule): An instance of a module conforming to one of the
                                          OrchestrationModule interfaces (e.g., QueryProcessor, Retriever).
            method_name (Optional[str]): The specific method on the module instance to call for this step.
                                         If None, the method is inferred based on the module's type.

        Returns:
            RAGOrchestrator: The orchestrator instance, allowing for fluent chaining of add_step calls.
        """
        step = PipelineStep(name, module, method_name)
        self._pipeline_steps.append(step)
        logger.info(f"Added step '{name}' ({type(module).__name__}) to pipeline '{self.pipeline_name}'.")
        return self

    def run(self, initial_query: Query, initial_context: Optional[Context] = None) -> PipelineResult:
        """
        Executes the defined multi-stage RAG pipeline with the given initial query.

        The orchestrator iterates through the defined pipeline steps, passing a
        mutable Context object that accumulates results and state from each stage.

        Args:
            initial_query (Query): The user's original query.
            initial_context (Optional[Context]): An optional initial Context object
                                                 to pre-populate pipeline state before
                                                 the first step. If None, a new Context
                                                 is created.

        Returns:
            PipelineResult: A comprehensive object containing the final query,
                            generated response, and the complete execution context.
        """
        if not self._pipeline_steps:
            logger.warning("Pipeline is empty. No steps to execute.")
            # Return an empty result with the initial query.
            return PipelineResult(
                query=initial_query,
                generated_response=GeneratedResponse(text="No pipeline steps defined or executed."),
                context=initial_context or Context(query=initial_query, initial_query=initial_query),
                success=False,
                error_message="Pipeline is empty."
            )

        # Initialize the context for the pipeline run
        current_context = initial_context if initial_context else Context(query=initial_query, initial_query=initial_query)
        # Ensure the current query in context is the initial one at the start of the run
        current_context.query = initial_query
        current_context.pipeline_history = [] # Reset history for a new run

        logger.info(f"Starting RAG pipeline '{self.pipeline_name}' for query: '{initial_query.text}'")

        try:
            for i, step in enumerate(self._pipeline_steps):
                current_context.pipeline_history.append(f"Executing step {i+1}: {step.name}")
                current_context = step.execute(current_context) # Each step updates the context

                # Log intermediate context state (optional, can be verbose)
                logger.debug(f"Context state after '{step.name}': Query={current_context.query.text}, "
                             f"Retrieved Docs={len(current_context.retrieval_results)}, "
                             f"Reranked Docs={len(current_context.reranked_results)}, "
                             f"Response_exists={current_context.generated_response is not None}")

                # Agentic loop considerations:
                # If an AgenticModule intends to dynamically alter the pipeline flow
                # (e.g., re-run a previous stage, add new steps), it would need to
                # communicate this via the Context or a specific return object that
                # the Orchestrator can interpret. For this linear orchestrator,
                # such complex re-routing logic would need to be handled by the
                # AgenticModule itself (e.g., the agent might internally call
                # retrieve multiple times), or by a higher-level "super-orchestrator"
                # that controls multiple RAGOrchestrator runs based on agentic feedback.
                # For this implementation, the flow is strictly sequential through added steps.

            final_response = current_context.generated_response
            if not final_response:
                # If the last step wasn't a generator or it failed to produce a response
                logger.warning(
                    "Pipeline finished without a generated response. "
                    "Ensure a Generator module is included and executed successfully."
                )
                final_response = GeneratedResponse(
                    text="No response generated. Pipeline might not include a Generator stage or it failed.",
                    citations=[],
                    metadata={"reason": "no_generator_output"}
                )

            final_result = PipelineResult(
                query=initial_query,
                generated_response=final_response,
                context=current_context,
                success=True,
                error_message=None
            )
            logger.info(f"RAG pipeline '{self.pipeline_name}' completed successfully.")
            return final_result

        except Exception as e:
            logger.exception(f"RAG pipeline '{self.pipeline_name}' failed during execution.")
            # On error, return a PipelineResult indicating failure, with partial context if available.
            return PipelineResult(
                query=initial_query,
                generated_response=GeneratedResponse(text=f"Pipeline failed: {e}"),
                context=current_context,
                success=False,
                error_message=str(e)
            )