```python
import json
from typing import List, Dict, Any, Union

# Assuming these imports exist in the project context as per ARCHITECTURE NOTES
from retrievers.base_retriever import BaseRetriever
from llm_client import LLMClient
from vector_store_manager import VectorStoreManager
from prompts.retrieval_prompts import QUERY_REWRITING_PROMPT
from config import settings

class QueryRewritingRetriever(BaseRetriever):
    """
    Implements an agentic search retriever that uses an LLM to rewrite,
    expand, or generate multiple permutations of the initial user query
    to cast a wider net and capture more comprehensive context.

    This retriever aims to improve factuality and completeness by exploring
    a broader range of search terms beyond the initial user query.
    """

    def __init__(self, llm_client: LLMClient, vector_store_manager: VectorStoreManager):
        """
        Initializes the QueryRewritingRetriever with an LLM client and a vector store manager.

        Args:
            llm_client (LLMClient): An instance of the LLMClient for interacting with LLMs.
            vector_store_manager (VectorStoreManager): An instance of the VectorStoreManager
                                                       for performing document retrieval.
        Raises:
            TypeError: If llm_client or vector_store_manager are not of the expected types.
            AttributeError: If required settings are not defined in config.
        """
        if not isinstance(llm_client, LLMClient):
            raise TypeError("llm_client must be an instance of LLMClient.")
        if not isinstance(vector_store_manager, VectorStoreManager):
            raise TypeError("vector_store_manager must be an instance of VectorStoreManager.")

        self.llm_client = llm_client
        self.vector_store_manager = vector_store_manager
        self.rewriting_prompt = QUERY_REWRITING_PROMPT

        # Ensure settings has required attributes
        if not hasattr(settings, 'LLM_MODEL_NAME'):
            raise AttributeError("settings.LLM_MODEL_NAME is not defined in config.")
        if not hasattr(settings, 'LLM_TEMPERATURE_CREATIVE'):
            raise AttributeError("settings.LLM_TEMPERATURE_CREATIVE is not defined in config.")

    def _rewrite_query(self, query: str) -> List[str]:
        """
        Uses an LLM to generate multiple rewritten or expanded versions of the original query.
        The LLM is expected to return a JSON array of strings, which is then parsed.
        Includes robust error handling and fallback parsing.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list of unique, rewritten query strings. Returns an empty list
                       if rewriting fails or yields no valid queries.
        """
        messages = [
            {"role": "system", "content": self.rewriting_prompt},
            {"role": "user", "content": f"Original query: {query}"}
        ]
        
        try:
            # Get completion from LLM with a creative temperature for diverse rewrites
            response_text = self.llm_client.get_completion(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE_CREATIVE,
                max_tokens=256  # Limit token generation for rewritten queries
            )
            
            # Parse the LLM's response into a list of queries
            rewritten_queries = self._parse_llm_rewrites(response_text)
            
            # Filter out empty or duplicate queries, maintaining order
            rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]
            rewritten_queries = list(dict.fromkeys(rewritten_queries)) # Deduplicate while preserving order
            
            return rewritten_queries
        except Exception as e:
            print(f"Error during query rewriting for query '{query}': {e}")
            # In a production system, consider more sophisticated error handling,
            # e.g., logging to a monitoring system or a more detailed fallback strategy.
            return [] # Return empty list on error, allowing the system to proceed with just the original query

    def _parse_llm_rewrites(self, llm_response: str) -> List[str]:
        """
        Parses the LLM's string response into a list of query strings.
        It first attempts to parse as a JSON array of strings. If that fails,
        it employs fallback strategies like splitting by newlines or commas.

        Args:
            llm_response (str): The raw string response from the LLM.

        Returns:
            List[str]: A list of parsed query strings. Returns an empty list
                       if parsing fails to yield any valid queries.
        """
        try:
            parsed = json.loads(llm_response)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
            else:
                # If JSON is valid but not a list of strings (e.g., {"query": "..."} or a single string)
                print(f"LLM returned valid JSON but not a list of strings as expected. Raw response: {llm_response}")
                # Attempt to extract a string if it's a dict or a single string
                if isinstance(parsed, dict) and "query" in parsed:
                    return [str(parsed["query"])] if isinstance(parsed["query"], str) else []
                if isinstance(parsed, str):
                    return [parsed]
                return []
        except json.JSONDecodeError:
            # Fallback for malformed JSON or non-JSON responses
            print(f"LLM response is not valid JSON. Attempting fallback parsing. Raw response: {llm_response}")
            queries = []
            # Try splitting by newlines first
            if '\n' in llm_response:
                queries = [q.strip() for q in llm_response.split('\n') if q.strip()]
            
            # If no queries found by newline, try splitting by commas
            if not queries and ',' in llm_response:
                queries = [q.strip() for q in llm_response.split(',') if q.strip()]
            
            # As a last resort, if the response isn't empty, treat the entire response as a single query
            if not queries and llm_response.strip():
                queries = [llm_response.strip()]
            
            return queries
        except Exception as e:
            print(f"Unexpected error in _parse_llm_rewrites: {e}. Raw response: {llm_response}")
            return []

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs retrieval using the original query and its LLM-rewritten permutations.
        Combines and deduplicates results from all queries to ensure comprehensive context.

        Args:
            query (str): The original user query.
            top_k (int): The number of top documents to retrieve for *each* individual query
                         (original and rewritten). The final output may contain more than
                         `top_k` unique documents if multiple queries yield distinct results,
                         thus ensuring a wider net for correctness.

        Returns:
            List[Dict[str, Any]]: A list of unique retrieved document dictionaries.
                                  Each dictionary typically contains 'text' and 'metadata'.
        """
        if not isinstance(query, str) or not query.strip():
            print("Warning: Received an empty or invalid query. Returning no documents.")
            return []

        if not isinstance(top_k, int) or top_k <= 0:
            print(f"Warning: Invalid top_k value ({top_k}). Using default top_k=5 for retrieval per sub-query.")
            top_k = 5

        # Generate rewritten queries
        rewritten_queries = self._rewrite_query(query)
        
        # Always include the original query in the search set
        all_queries = [query] + rewritten_queries
        
        # Deduplicate and filter empty queries from the combined list, maintaining order
        all_queries = [q.strip() for q in all_queries if q.strip()]
        all_queries = list(dict.fromkeys(all_queries)) # Ensure uniqueness and preserve order

        if not all_queries:
            print("No valid queries were generated or provided to perform retrieval.")
            return []

        # Use a dictionary to store unique documents, keyed by a stable identifier.
        # This prevents duplicate documents from being returned even if found by different queries.
        unique_documents: Dict[Union[str, int], Dict[str, Any]] = {} 

        for q_to_search in all_queries:
            try:
                # Retrieve documents for the current query from the vector store
                # Assuming vector_store_manager.retrieve returns a list of dicts,
                # where each dict represents a document chunk (e.g., {'text': '...', 'metadata': {'id': '...'}}).
                results = self.vector_store_manager.retrieve(q_to_search, top_k=top_k)
                
                for doc in results:
                    # Generate a stable document ID for deduplication.
                    # Prioritize 'id' from metadata, then fallback to hashing the entire document content.
                    doc_id = doc.get('metadata', {}).get('id')
                    if doc_id is None:
                        # If no explicit ID, use a hash of the document's content for uniqueness.
                        # Using frozenset(doc.items()) makes the dict hashable and robust against key order.
                        doc_id = hash(frozenset(doc.items())) 

                    if doc_id not in unique_documents:
                        unique_documents[doc_id] = doc

            except Exception as e:
                print(f"Error retrieving for query '{q_to_search}': {e}")
                # Log the error but continue with other queries to maximize context gathering.
                
        # Return the list of unique retrieved documents.
        # The number of documents may exceed the initial `top_k` due to multiple queries.
        return list(unique_documents.values())

```