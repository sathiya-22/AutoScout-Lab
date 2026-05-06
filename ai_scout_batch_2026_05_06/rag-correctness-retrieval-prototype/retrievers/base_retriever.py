from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval mechanisms.

    Defines the interface that all specific retriever implementations must adhere to,
    ensuring consistency across different retrieval strategies.
    """

    def __init__(self):
        """
        Initializes the base retriever.
        Subclasses should extend this to set up specific configurations
        (e.g., LLM clients, vector stores).
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> list[str]:
        """
        Abstract method to retrieve relevant context based on a given query.

        This method must be implemented by all concrete retriever subclasses.

        Args:
            query (str): The user's original or rewritten query.
            **kwargs: Arbitrary keyword arguments that specific retriever implementations
                      might need (e.g., `k` for number of results, `score_threshold`).

        Returns:
            list[str]: A list of strings, where each string represents a retrieved
                       document chunk or piece of context. The order of chunks
                       might imply relevance or be arbitrary, depending on the
                       implementation.
        """
        raise NotImplementedError("The 'retrieve' method must be implemented by subclasses.")