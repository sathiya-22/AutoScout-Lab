from typing import List, Dict, Any, Optional
import logging

# Assuming utils/logger.py exists and provides a setup_logger function
try:
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    # Fallback to basic logging if utils.logger is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Could not import setup_logger from utils.logger. Using basic logging configuration.")


# Assuming storage/vector_store_interface.py defines a VectorStoreInterface
# This interface should provide methods for semantic search.
try:
    from storage.vector_store_interface import VectorStoreInterface
except ImportError:
    logger.error("Could not import VectorStoreInterface from storage. Please ensure storage/vector_store_interface.py exists.")
    # Define a mock interface for development purposes if the actual one is missing
    class VectorStoreInterface:
        def similarity_search(self, query: str, k: int, score_threshold: Optional[float] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            """
            MOCK: Simulates a semantic similarity search.
            In a real implementation, this would interact with a vector database.
            """
            logger.warning("Using MOCK VectorStoreInterface. No actual vector search will be performed.")
            if "example" in query.lower():
                return [
                    {"text": "This is an example of a past interaction about task setup.", "score": 0.95, "metadata": {"type": "interaction", "timestamp": "2023-01-01T10:00:00"}},
                    {"text": "Documentation snippet regarding API authentication.", "score": 0.88, "metadata": {"type": "document", "source": "docs.api"}},
                ]
            return []


class SemanticContextRetriever:
    """
    Retrieves semantically relevant context from a vector store using embeddings.
    This module acts as an interface to the vector database, fetching information
    based on the semantic similarity of a query.
    """

    def __init__(self, vector_store: VectorStoreInterface):
        """
        Initializes the SemanticContextRetriever with a VectorStoreInterface instance.

        Args:
            vector_store (VectorStoreInterface): An instance of the vector store
                                                 interface for performing similarity searches.
        """
        if not isinstance(vector_store, VectorStoreInterface):
            raise TypeError("vector_store must be an instance of VectorStoreInterface")
        self._vector_store: VectorStoreInterface = vector_store
        logger.info("SemanticContextRetriever initialized.")

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves semantically relevant context snippets from the underlying vector store.

        Args:
            query (str): The query string representing the current context or request
                         for which relevant information is sought.
            k (int): The maximum number of top-k results to retrieve. Must be positive.
            score_threshold (Optional[float]): An optional minimum similarity score.
                                                Only results with a score equal to or
                                                above this threshold will be returned.
            metadata_filters (Optional[Dict[str, Any]]): Optional metadata filters
                                                          to apply during the search.
                                                          (e.g., {"type": "user_interaction"}).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
                                  a retrieved context item. Each item typically includes
                                  'text', 'score', and 'metadata'. Returns an empty list
                                  if no relevant context is found or an error occurs.
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided for semantic context retrieval. Query must be a non-empty string.")
            return []

        if not isinstance(k, int) or k <= 0:
            logger.warning(f"Invalid value for 'k': {k}. 'k' must be a positive integer. Defaulting to 1.")
            k = 1

        if score_threshold is not None and (not isinstance(score_threshold, (int, float)) or not (0.0 <= score_threshold <= 1.0)):
            logger.warning(f"Invalid value for 'score_threshold': {score_threshold}. It must be a float between 0.0 and 1.0, or None. Ignoring threshold.")
            score_threshold = None

        if metadata_filters is not None and not isinstance(metadata_filters, dict):
            logger.warning(f"Invalid type for 'metadata_filters': {type(metadata_filters)}. Expected dict or None. Ignoring filters.")
            metadata_filters = None

        try:
            logger.debug(f"Attempting semantic retrieval for query (first 100 chars): '{query[:100]}...' with k={k}, score_threshold={score_threshold}, filters={metadata_filters}")
            results = self._vector_store.similarity_search(
                query=query,
                k=k,
                score_threshold=score_threshold,
                metadata_filters=metadata_filters
            )
            logger.info(f"Successfully retrieved {len(results)} semantic context items for query.")
            return results
        except Exception as e:
            logger.error(f"Error during semantic context retrieval for query '{query[:100]}...': {e}", exc_info=True)
            return []

# Example of how it might be used (for testing/demonstration, not part of the required output)
# if __name__ == "__main__":
#     # Mock VectorStoreInterface for demonstration
#     class MockVectorStore(VectorStoreInterface):
#         def similarity_search(self, query: str, k: int, score_threshold: Optional[float] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
#             print(f"Mock search received: '{query}', k={k}, threshold={score_threshold}, filters={metadata_filters}")
#             mock_data = [
#                 {"text": "The user asked to add a new feature for data processing.", "score": 0.98, "metadata": {"type": "user_query"}},
#                 {"text": "Previous task involved refactoring the database access layer.", "score": 0.85, "metadata": {"type": "task_history"}},
#                 {"text": "A common utility function for string manipulation exists in `utils.py`.", "score": 0.75, "metadata": {"type": "code_snippet"}},
#                 {"text": "The project requires Python 3.9 or higher.", "score": 0.60, "metadata": {"type": "project_info"}},
#                 {"text": "Customer feedback mentioned issues with login flow.", "score": 0.90, "metadata": {"type": "customer_feedback"}},
#             ]
#             results = []
#             # Simple heuristic for mock similarity based on query content
#             for item in mock_data:
#                 if query.lower() in item['text'].lower() or \
#                    ("feature" in query.lower() and "feature" in item['text'].lower()) or \
#                    ("database" in query.lower() and "database" in item['text'].lower()) or \
#                    ("login" in query.lower() and "login" in item['text'].lower()):
#                     if score_threshold is None or item['score'] >= score_threshold:
#                         if metadata_filters is None or all(item['metadata'].get(k) == v for k, v in metadata_filters.items()):
#                             results.append(item)
#             results.sort(key=lambda x: x['score'], reverse=True)
#             return results[:k]
#
#     mock_vector_store = MockVectorStore()
#     retriever = SemanticContextRetriever(mock_vector_store)
#
#     print("\n--- Test 1: Basic retrieval ---")
#     context = retriever.retrieve_context(query="user wants a new feature", k=2)
#     for item in context:
#         print(f"  Score: {item['score']:.2f}, Text: {item['text']}")
#
#     print("\n--- Test 2: Retrieval with score threshold ---")
#     context = retriever.retrieve_context(query="database issue", k=3, score_threshold=0.8)
#     for item in context:
#         print(f"  Score: {item['score']:.2f}, Text: {item['text']}")
#
#     print("\n--- Test 3: Retrieval with metadata filter ---")
#     context = retriever.retrieve_context(query="customer comments", k=5, metadata_filters={"type": "customer_feedback"})
#     for item in context:
#         print(f"  Score: {item['score']:.2f}, Text: {item['text']}")
#
#     print("\n--- Test 4: Invalid query ---")
#     context = retriever.retrieve_context(query=123, k=2)
#     print(f"Result for invalid query: {context}")
#
#     print("\n--- Test 5: Invalid k ---")
#     context = retriever.retrieve_context(query="test", k=0)
#     print(f"Result for invalid k: {context}")
#
#     print("\n--- Test 6: No results ---")
#     context = retriever.retrieve_context(query="unrelated topic", k=1)
#     print(f"Result for unrelated topic: {context}")
#
#     print("\n--- Test 7: Threshold too high ---")
#     context = retriever.retrieve_context(query="previous task", k=5, score_threshold=0.99)
#     for item in context:
#         print(f"  Score: {item['score']:.2f}, Text: {item['text']}")