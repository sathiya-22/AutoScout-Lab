```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.core.models import Query, RetrievalResult
from src.core.interfaces import Retriever as RetrieverInterface


class BaseRetriever(RetrieverInterface, ABC):
    """
    Abstract Base Class for all Retrieval modules within the Advanced RAG Orchestration Framework.

    This ABC implements the `RetrieverInterface` and provides a foundational structure
    and common properties expected from all concrete retriever implementations (e.g.,
    vector, keyword, graph) in this module. It ensures a standardized contract for
    integration into the multi-stage RAG pipeline.

    Concrete implementations must inherit from this class and provide specific
    implementations for the abstract methods and properties defined here.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the base retriever with configuration.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary for the retriever.
                                               This dictionary can contain various settings
                                               like API keys, model paths, thresholds, or
                                               specific index configurations.
                                               Defaults to an empty dictionary.
        """
        self.config = config if config is not None else {}
        # Subclasses should call super().__init__(config) if they override __init__
        # to ensure base configuration is handled.

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the unique name of the retriever implementation.

        This property must be implemented by concrete classes to identify
        the specific retriever (e.g., "VectorRetriever", "KeywordRetriever").
        """
        pass

    @abstractmethod
    async def retrieve(self, query: Query, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """
        Asynchronously retrieves relevant documents or passages based on the input query.

        This method encapsulates the core logic for searching one or more indices
        (e.g., vector databases, keyword search engines, graph databases) and returning
        a list of retrieval results.

        Args:
            query (Query): The processed query object, which may contain the original
                           user query, expanded queries, recognized intent, or other
                           information for targeted retrieval.
            top_k (int): The maximum number of retrieval results to return.
                         The actual number might be less if fewer relevant items are found.
            **kwargs: Additional parameters specific to the retriever implementation,
                      such as specific index names, filters, query modifiers, or
                      search strategies (e.g., 'search_type', 'filter_conditions').

        Returns:
            List[RetrievalResult]: A list of `RetrievalResult` objects, each containing
                                   a document and its associated score, metadata, and source.
                                   The list should be ordered by relevance or score.

        Raises:
            RetrievalError: A custom exception (expected to be defined in `src/core/exceptions.py`
                            or `src/utils.py`) indicating a failure during the retrieval process.
                            This could include issues like connectivity problems with external
                            services, invalid index configurations, or upstream search failures.
                            Concrete implementations should catch specific exceptions and
                            re-raise them as a `RetrievalError` for consistent error handling.
        """
        pass

    async def aclose(self):
        """
        Asynchronously closes any open connections or releases resources held by the retriever.

        This method provides a hook for graceful shutdown and resource management.
        Concrete implementations should override this if they manage external resources
        such as database connections, client sessions, or file handles.
        The default implementation performs no operation.
        """
        pass
```