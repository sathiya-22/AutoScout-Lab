import logging
from typing import List, Optional

# Assume these components exist in their respective directories
from config import Config
from llm_client import LLMClient
from retrievers.query_rewriting_retriever import QueryRewritingRetriever
from retrievers.multi_stage_retriever import MultiStageRetriever
from retrievers.correctness_verifier import CorrectnessVerifier
from prompts import generation_prompts

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    The central orchestrator for the RAG system. It coordinates the entire workflow
    from receiving a user query to generating a verified response.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        query_rewriter: QueryRewritingRetriever,
        multi_stage_retriever: MultiStageRetriever,
        correctness_verifier: CorrectnessVerifier
    ):
        """
        Initializes the RAGOrchestrator with instances of its core components.

        Args:
            llm_client: An instance of LLMClient for final answer generation.
            query_rewriter: An instance of QueryRewritingRetriever for initial query expansion.
            multi_stage_retriever: An instance of MultiStageRetriever for comprehensive context gathering.
            correctness_verifier: An instance of CorrectnessVerifier for factual validation.
        """
        self.llm_client = llm_client
        self.query_rewriter = query_rewriter
        self.multi_stage_retriever = multi_stage_retriever
        self.correctness_verifier = correctness_verifier

        logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("RAGOrchestrator initialized.")

    def orchestrate_rag_flow(self, user_query: str) -> str:
        """
        Coordinates the full RAG workflow for a given user query.

        Args:
            user_query: The initial query from the user.

        Returns:
            The final generated and verified response, or an error message if something fails.
        """
        if not user_query:
            logger.warning("Received an empty user query.")
            return "Please provide a non-empty query."

        logger.info(f"Starting RAG orchestration for query: '{user_query}'")
        retrieved_context: List[str] = []
        verified_context: List[str] = []
        final_answer: Optional[str] = None

        try:
            # 1. Query Rewriting/Expansion
            logger.debug("Step 1: Rewriting/Expanding query...")
            rewritten_queries = self.query_rewriter.rewrite_query(user_query)
            if not rewritten_queries:
                logger.warning("Query rewriting yielded no expanded queries. Using original query for retrieval.")
                queries_for_retrieval = [user_query]
            else:
                logger.info(f"Rewritten queries: {rewritten_queries}")
                queries_for_retrieval = rewritten_queries

            # 2. Multi-Stage Retrieval
            logger.debug("Step 2: Performing multi-stage context retrieval...")
            retrieved_context = self.multi_stage_retriever.retrieve_context(queries_for_retrieval)

            if not retrieved_context:
                logger.warning("No context retrieved for the given queries.")
                return "I couldn't find relevant information to answer your query. Please try rephrasing."

            logger.info(f"Retrieved {len(retrieved_context)} context chunks.")
            # Limit context for logging to avoid excessive output
            logger.debug(f"First few retrieved chunks: {retrieved_context[:Config.MAX_LOG_CONTEXT_CHUNKS]}...")

            # 3. Correctness Verification
            logger.debug("Step 3: Verifying retrieved context for correctness and completeness...")
            verified_context = self.correctness_verifier.verify_context(
                context_chunks=retrieved_context,
                original_query=user_query
            )

            if not verified_context:
                logger.warning("Correctness verification found no reliable context after filtering/refinement.")
                return ("I found some information, but it couldn't be fully verified for correctness or completeness. "
                        "Therefore, I cannot provide a confident answer at this time.")

            logger.info(f"After verification, {len(verified_context)} context chunks remain for generation.")
            logger.debug(f"First few verified chunks: {verified_context[:Config.MAX_LOG_CONTEXT_CHUNKS]}...")

            # 4. Final Answer Generation
            logger.debug("Step 4: Generating final answer with verified context...")
            generation_prompt = generation_prompts.generate_answer_prompt(
                context=verified_context,
                query=user_query
            )
            final_answer = self.llm_client.generate_response(generation_prompt)

            if not final_answer:
                logger.error("LLM client returned an empty response during final generation.")
                return "I encountered an issue generating a response based on the verified context."

            logger.info("RAG orchestration completed successfully.")
            return final_answer

        except Exception as e:
            logger.exception(f"An unexpected error occurred during RAG orchestration: {e}")
            return (f"An internal error occurred while processing your request. Please try again later. "
                    f"Error details: {str(e)}")

# Example of how to integrate/instantiate (for development testing, not part of the class)
if __name__ == "__main__":
    logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running RAGOrchestrator demonstration.")

    # These would typically be initialized in main.py
    _llm_client = LLMClient(model_name=Config.LLM_MODEL)
    _query_rewriter = QueryRewritingRetriever(llm_client=_llm_client, vector_store_manager=None) # vector_store_manager might be needed for actual rewriting
    _multi_stage_retriever = MultiStageRetriever(llm_client=_llm_client, vector_store_manager=None) # vector_store_manager is crucial here
    _correctness_verifier = CorrectnessVerifier(llm_client=_llm_client)

    # Basic mock setup for demonstration if actual components aren't fully functional yet
    class MockLLMClient:
        def generate_response(self, prompt: str) -> str:
            if "rewrite" in prompt.lower():
                return "How does photosynthesis work in plants?;What is the process of photosynthesis?"
            if "verify" in prompt.lower():
                return "Verified context: ['Photosynthesis converts light energy into chemical energy. It occurs in chloroplasts.']"
            return "Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy."
    
    class MockQueryRewritingRetriever:
        def __init__(self, llm_client): pass
        def rewrite_query(self, query: str) -> List[str]:
            if "photosynthesis" in query.lower():
                return ["detailed mechanism of photosynthesis", "energy conversion in plants"]
            return [query]

    class MockMultiStageRetriever:
        def __init__(self, llm_client, vector_store_manager): pass
        def retrieve_context(self, queries: List[str]) -> List[str]:
            if "photosynthesis" in "".join(queries).lower():
                return [
                    "Photosynthesis is a process used by plants, algae, and cyanobacteria to convert light energy into chemical energy.",
                    "It involves carbon dioxide, water, and sunlight to produce glucose and oxygen.",
                    "Chlorophyll, a green pigment, plays a crucial role in absorbing light energy."
                ]
            return []
            
    class MockCorrectnessVerifier:
        def __init__(self, llm_client): pass
        def verify_context(self, context_chunks: List[str], original_query: str) -> List[str]:
            # For demonstration, assume all context is verified
            if context_chunks:
                logger.info("Mock verification: all context passed.")
                return context_chunks
            return []

    # Use mocks for the demo
    mock_llm_client = MockLLMClient()
    mock_query_rewriter = MockQueryRewritingRetriever(mock_llm_client)
    mock_multi_stage_retriever = MockMultiStageRetriever(mock_llm_client, None)
    mock_correctness_verifier = MockCorrectnessVerifier(mock_llm_client)

    orchestrator = RAGOrchestrator(
        llm_client=mock_llm_client,
        query_rewriter=mock_query_rewriter,
        multi_stage_retriever=mock_multi_stage_retriever,
        correctness_verifier=mock_correctness_verifier
    )

    query = "How does photosynthesis work?"
    response = orchestrator.orchestrate_rag_flow(query)
    print(f"\nUser Query: {query}")
    print(f"Orchestrator Response: {response}")

    query_no_context = "Tell me about theoretical physics in the 17th century."
    response_no_context = orchestrator.orchestrate_rag_flow(query_no_context)
    print(f"\nUser Query: {query_no_context}")
    print(f"Orchestrator Response: {response_no_context}")
    
    query_empty = ""
    response_empty = orchestrator.orchestrate_rag_flow(query_empty)
    print(f"\nUser Query: '{query_empty}'")
    print(f"Orchestrator Response: {response_empty}")