import logging
from typing import List, Optional

from src.core.models import Query, Document, RetrievalResult
from src.modules.retrieval.base import AbstractRetriever
from src.providers.vector_db_provider import BaseVectorDBProvider

logger = logging.getLogger(__name__)

class VectorRetriever(AbstractRetriever):
    """
    A retriever implementation that performs vector similarity search against a
    configured vector database.
    """

    def __init__(self, vector_db_provider: BaseVectorDBProvider, default_index_name: str, default_top_k: int = 5):
        """
        Initializes the VectorRetriever.

        Args:
            vector_db_provider: An instance of a vector database provider.
            default_index_name: The default index to search within the vector database.
            default_top_k: The default number of top results to retrieve.
        """
        if not isinstance(vector_db_provider, BaseVectorDBProvider):
            raise TypeError("vector_db_provider must be an instance of BaseVectorDBProvider")
        if not default_index_name:
            raise ValueError("default_index_name cannot be empty")

        self.vector_db_provider = vector_db_provider
        self.default_index_name = default_index_name
        self.default_top_k = default_top_k
        logger.info(f"VectorRetriever initialized with index '{default_index_name}' and default top_k={default_top_k}")

    async def retrieve(self, query: Query, top_k: Optional[int] = None, **kwargs) -> RetrievalResult:
        """
        Performs vector similarity search based on the query's vector embedding.

        Args:
            query: The Query object containing the text and optionally its vector embedding.
            top_k: The number of top results to retrieve, overrides default_top_k if provided.
            **kwargs: Additional keyword arguments for the vector database search (e.g., filters).

        Returns:
            A RetrievalResult object containing the retrieved documents.

        Raises:
            ValueError: If the query does not contain a vector embedding.
            Exception: For issues during the vector database interaction.
        """
        if query.vector_embedding is None:
            error_msg = "VectorRetriever requires 'query.vector_embedding' to be present for retrieval."
            logger.error(error_msg)
            return RetrievalResult(query=query, documents=[], source_module=self.__class__.__name__, error=error_msg)

        k_value = top_k if top_k is not None else query.top_k if query.top_k is not None else self.default_top_k
        index_name = kwargs.pop('index_name', self.default_index_name)

        retrieved_docs: List[Document] = []
        error_message: Optional[str] = None

        logger.debug(f"Performing vector search on index '{index_name}' for query '{query.text[:50]}...' with top_k={k_value}")

        try:
            # The vector_db_provider's search method should return a list of
            # dictionaries or Pydantic models that can be converted to Document.
            # It's assumed to handle its own specific query format.
            db_results = await self.vector_db_provider.search(
                embedding=query.vector_embedding,
                index_name=index_name,
                top_k=k_value,
                **kwargs
            )

            if not db_results:
                logger.info(f"No vector search results found for query '{query.text[:50]}...' in index '{index_name}'.")
            else:
                for result in db_results:
                    # Assuming db_results are dict-like objects or Pydantic models
                    # that can be directly passed to Document.
                    # Adjust this mapping based on the actual output format of your vector_db_provider.
                    try:
                        doc = Document(
                            id=str(result.get('id', '')),
                            text=result.get('text', ''),
                            metadata=result.get('metadata', {}),
                            score=result.get('score', 0.0)
                        )
                        retrieved_docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to convert vector DB result to Document: {e}. Result: {result}")
                        continue

        except ConnectionError as ce:
            error_message = f"Vector DB connection error during retrieval: {ce}"
            logger.error(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred during vector retrieval: {e}"
            logger.error(error_message, exc_info=True)

        logger.debug(f"VectorRetriever completed with {len(retrieved_docs)} documents.")
        return RetrievalResult(
            query=query,
            documents=retrieved_docs,
            source_module=self.__class__.__name__,
            error=error_message
        )