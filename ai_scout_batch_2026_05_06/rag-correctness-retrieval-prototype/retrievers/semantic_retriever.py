from typing import List, Optional

# Assuming a Document class structure for consistency across the RAG system.
# In a full project, this might be imported from a common 'models.py' or 'data_loader.py'.
class Document:
    """Represents a chunk of text with associated metadata."""
    def __init__(self, page_content: str, metadata: dict = None):
        if not isinstance(page_content, str):
            raise TypeError("page_content must be a string.")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        # Provide a concise representation for debugging
        content_preview = self.page_content[:100] + "..." if len(self.page_content) > 100 else self.page_content
        return f"Document(page_content='{content_preview}', metadata={self.metadata})"

    def to_dict(self):
        """Converts the Document object to a dictionary."""
        return {"page_content": self.page_content, "metadata": self.metadata}

# Import the abstract base class for retrievers.
# This assumes 'base_retriever.py' defines a class named 'BaseRetriever'
# in the same 'retrievers' directory.
from retrievers.base_retriever import BaseRetriever

# Import the VectorStoreManager to interact with the vector database.
# This assumes 'vector_store_manager.py' is in the project's root or a common utility folder
# and is importable like 'from vector_store_manager import VectorStoreManager'.
# Adjust import path if vector_store_manager is in a different location relative to retrievers/.
try:
    from vector_store_manager import VectorStoreManager
except ImportError:
    # A simple mock for isolated testing or if the file structure isn't fully set up yet.
    # In a production environment, this ImportError should indicate a missing dependency.
    print("Warning: Could not import VectorStoreManager. Using a mock class. "
          "Ensure 'vector_store_manager.py' is correctly placed and accessible.")
    class VectorStoreManager: # Mock class
        def __init__(self, *args, **kwargs):
            print("Mock VectorStoreManager initialized.")
        def query_vector_store(self, query: str, k: int = 4) -> List[Document]:
            print(f"MOCK VectorStoreManager: Simulating query for '{query}' with k={k}")
            return [Document(f"Mock document content related to '{query}' (part {i+1})", {"source": "mock_data"})
                    for i in range(k)]

# Import the LLMClient. While not directly used by this specific retriever,
# it might be part of the common interface for BaseRetriever or for future expansion.
try:
    from llm_client import LLMClient
except ImportError:
    # A simple mock for isolated testing or if the file structure isn't fully set up yet.
    print("Warning: Could not import LLMClient. Using a mock class. "
          "Ensure 'llm_client.py' is correctly placed and accessible.")
    class LLMClient: # Mock class
        def __init__(self, *args, **kwargs):
            print("Mock LLMClient initialized.")
        def generate(self, prompt: str, **kwargs) -> str:
            print(f"MOCK LLMClient: Generating response for prompt snippet: '{prompt[:50]}...'")
            return "This is a mock LLM generated response."


class SemanticRetriever(BaseRetriever):
    """
    Implements a baseline retriever using standard semantic similarity search.
    It queries a vector store to find documents most semantically similar to the input query.
    This retriever prioritizes 'relevance' based on embedding similarity, serving as a foundational
    component for more sophisticated retrieval strategies.
    """

    def __init__(self, vector_store_manager: VectorStoreManager, llm_client: Optional[LLMClient] = None):
        """
        Initializes the SemanticRetriever.

        Args:
            vector_store_manager (VectorStoreManager): An instance of the vector store manager,
                                                       responsible for interacting with the vector database.
            llm_client (Optional[LLMClient]): An optional LLM client instance. While not directly used
                                               by this basic semantic retriever for its core function,
                                               it's included for API consistency if the BaseRetriever
                                               interface or other components expect it across all retriever types.
        Raises:
            TypeError: If vector_store_manager is not an instance of VectorStoreManager.
        """
        if not isinstance(vector_store_manager, VectorStoreManager):
            raise TypeError("vector_store_manager must be an instance of VectorStoreManager.")
        self.vector_store_manager = vector_store_manager
        self.llm_client = llm_client  # Stored, but not actively used in the retrieve method itself.

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieves documents from the vector store based on semantic similarity to the query.

        This method performs a direct vector search to find the top 'k' documents
        whose embeddings are most similar to the query embedding.

        Args:
            query (str): The user's query string for which to find relevant documents.
            k (int): The number of top relevant documents to retrieve. Must be a positive integer.

        Returns:
            List[Document]: A list of Document objects semantically similar to the query.
                            Returns an empty list if no documents are found, or if an error occurs,
                            or if the input query is invalid.
        """
        if not query or not isinstance(query, str):
            print("SemanticRetriever Warning: Query must be a non-empty string. Returning empty list.")
            return []
        if not isinstance(k, int) or k <= 0:
            print(f"SemanticRetriever Warning: 'k' must be a positive integer. Received '{k}'. Defaulting to 4.")
            k = 4 # Reset to default valid value

        try:
            # Perform the semantic similarity search using the vector store manager
            print(f"SemanticRetriever: Performing semantic search for query: '{query[:75]}...' (k={k})")
            retrieved_docs = self.vector_store_manager.query_vector_store(query, k=k)

            if not retrieved_docs:
                print(f"SemanticRetriever: No documents found for query: '{query[:75]}...'.")

            return retrieved_docs

        except Exception as e:
            # Catch broad exceptions during retrieval to prevent application crashes
            print(f"SemanticRetriever Error during retrieval for query '{query[:75]}...': {e}")
            return []