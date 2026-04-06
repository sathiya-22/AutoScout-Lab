from abc import ABC, abstractmethod
from typing import List, Any

# Assuming models.py is in the same directory as interfaces.py
from .models import Query, Document, RetrievalResult, Context, PipelineResult, PipelineState


class BaseModule(ABC):
    """
    Abstract Base Class for all pluggable modules in the RAG orchestration framework.
    Provides a common interface for module initialization and configuration.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the module with a configuration dictionary.

        Args:
            config (dict, optional): Configuration settings specific to the module.
                                     Defaults to None, allowing modules to define
                                     their own default configurations.
        """
        self.config = config if config is not None else {}
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """
        Abstract method to be implemented by concrete modules for validating their
        specific configuration settings. This ensures that essential parameters
        are present and correctly typed.
        """
        pass


class QueryProcessor(BaseModule):
    """
    Abstract Base Class for modules responsible for processing and enhancing user queries.
    Examples: intent recognition, query translation, hypothetical document generation.
    """

    @abstractmethod
    def process(self, query: Query) -> Query:
        """
        Processes the input query, potentially transforming it, extracting entities,
        or generating expanded forms.

        Args:
            query (Query): The original or current query object.

        Returns:
            Query: The processed, enhanced, or transformed query object.
        """
        pass


class Retriever(BaseModule):
    """
    Abstract Base Class for modules responsible for retrieving relevant documents
    or passages based on a query. This can encompass vector search, keyword search,
    graph search, or hybrid approaches.
    """

    @abstractmethod
    def retrieve(self, query: Query, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """
        Retrieves a list of potential documents or passages relevant to the query.

        Args:
            query (Query): The processed query object.
            top_k (int): The maximum number of retrieval results to return.
            **kwargs: Additional parameters for specific retrieval implementations
                      (e.g., filters, specific index names).

        Returns:
            List[RetrievalResult]: A list of retrieved documents with their scores and metadata.
        """
        pass


class Reranker(BaseModule):
    """
    Abstract Base Class for modules that re-rank an initial set of retrieved documents
    to improve relevance, diversity, or other criteria.
    """

    @abstractmethod
    def rerank(self, query: Query, results: List[RetrievalResult], top_n: int = 5, **kwargs) -> List[RetrievalResult]:
        """
        Reranks a list of retrieval results based on the query, aiming to improve
        the order or select the most relevant subset.

        Args:
            query (Query): The processed query object.
            results (List[RetrievalResult]): The initial list of retrieval results.
            top_n (int): The number of top results to return after reranking.
            **kwargs: Additional parameters for specific reranking implementations
                      (e.g., diversity thresholds, contextual signals).

        Returns:
            List[RetrievalResult]: The reranked and potentially truncated list of retrieval results.
        """
        pass


class EvidenceCompressor(BaseModule):
    """
    Abstract Base Class for modules that process and compress retrieved evidence
    into a concise context suitable for LLM generation.
    Examples: summarization, extractive QA, prompt optimization.
    """

    @abstractmethod
    def compress(self, query: Query, results: List[RetrievalResult], **kwargs) -> Context:
        """
        Compresses and optimizes the retrieved evidence to create a coherent
        context for the Language Model.

        Args:
            query (Query): The processed query object.
            results (List[RetrievalResult]): The reranked or filtered retrieval results.
            **kwargs: Additional parameters for specific compression implementations
                      (e.g., summary length, compression strategy).

        Returns:
            Context: An object containing the optimized prompt context for the LLM.
        """
        pass


class Generator(BaseModule):
    """
    Abstract Base Class for modules responsible for generating the final response
    using a Language Model based on the query and compressed context.
    """

    @abstractmethod
    def generate(self, query: Query, context: Context, **kwargs) -> PipelineResult:
        """
        Generates a response using an LLM based on the query and the provided context.

        Args:
            query (Query): The original or processed query.
            context (Context): The optimized context derived from retrieved evidence.
            **kwargs: Additional parameters for LLM interaction (e.g., temperature,
                      max_tokens, specific model configurations).

        Returns:
            PipelineResult: The final result object containing the generated answer
                            and potentially other metadata.
        """
        pass


class AgenticModule(BaseModule):
    """
    Abstract Base Class for agentic modules that provide intelligent control,
    iterative refinement, and critique capabilities to the RAG pipeline.
    This module can decide to re-run pipeline stages, call tools, or evaluate results.
    """

    @abstractmethod
    def supervise(self, initial_state: PipelineState, **kwargs) -> PipelineState:
        """
        Supervises and potentially refines the RAG pipeline's output through
        iterative steps, critique, or tool use. It operates on the full pipeline state.

        Args:
            initial_state (PipelineState): The current state of the RAG pipeline,
                                           including the query, retrieved results,
                                           generated answer, etc.
            **kwargs: Additional parameters for the agentic process (e.g., number
                      of refinement steps, access to external tools).

        Returns:
            PipelineState: The potentially refined or final state of the pipeline
                           after the agentic module's intervention.
        """
        pass<ctrl63>