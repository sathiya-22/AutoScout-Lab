import os
import sys
from typing import Dict, Any

# Add the 'src' directory to the Python path to allow importing modules directly from it.
# This assumes main.py is in the root directory and 'src' is a sibling directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import config
from utils.logger import setup_logging

# Import core components
from models.llm_interface import LLMInterface
from models.embedding_manager import EmbeddingManager

# Import retrieval components
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.graph_retriever import GraphRetriever
from retrieval.logical_structure_processor import LogicalStructureProcessor
from retrieval.hybrid_retriever import HybridRetriever

# Import query processing components
from query_processing.query_rewriter import QueryRewriter
from query_processing.agent_orchestrator import AgentOrchestrator

# Setup logging for the main entry point
logger = setup_logging(__name__)

def initialize_components() -> Dict[str, Any]:
    """
    Initializes all necessary components of the Advanced RAG system.
    This includes LLM/Embedding interfaces, various retrievers,
    query rewriter, and the central agent orchestrator.
    """
    logger.info("Starting component initialization...")
    try:
        # Initialize core model interfaces
        llm_interface = LLMInterface(
            model_name=config.LLM_MODEL_NAME,
            api_key=config.OPENAI_API_KEY # Assuming OpenAI, adjust as per config
        )
        logger.info(f"LLMInterface initialized with model: {config.LLM_MODEL_NAME}")

        embedding_manager = EmbeddingManager(
            model_name=config.EMBEDDING_MODEL_NAME,
            api_key=config.OPENAI_API_KEY # Assuming OpenAI, adjust as per config
        )
        logger.info(f"EmbeddingManager initialized with model: {config.EMBEDDING_MODEL_NAME}")

        # Initialize individual retrievers
        # These will connect to their respective stores (Faiss/Chroma/Weaviate, BM25 index, Neo4j/NetworkX)
        vector_retriever = VectorRetriever(
            embedding_manager=embedding_manager,
            vector_db_config=config.VECTOR_DB_CONFIG
        )
        logger.info("VectorRetriever initialized.")

        keyword_retriever = KeywordRetriever(
            index_path=config.KEYWORD_INDEX_PATH # Path to pre-built BM25/TF-IDF index
        )
        logger.info("KeywordRetriever initialized.")

        graph_retriever = GraphRetriever(
            graph_db_config=config.GRAPH_DB_CONFIG # Configuration for Neo4j/NetworkX
        )
        logger.info("GraphRetriever initialized.")

        logical_structure_processor = LogicalStructureProcessor(
            llm_interface=llm_interface
        )
        logger.info("LogicalStructureProcessor initialized.")

        # Initialize the Hybrid Retriever which orchestrates all individual retrievers
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            graph_retriever=graph_retriever,
            logical_structure_processor=logical_structure_processor,
            llm_interface=llm_interface # For re-ranking or fusion logic
        )
        logger.info("HybridRetriever initialized.")

        # Initialize the Query Rewriter
        query_rewriter = QueryRewriter(llm_interface=llm_interface)
        logger.info("QueryRewriter initialized.")

        # Initialize the Agent Orchestrator, the brain of the RAG system
        agent_orchestrator = AgentOrchestrator(
            query_rewriter=query_rewriter,
            hybrid_retriever=hybrid_retriever,
            llm_interface=llm_interface
        )
        logger.info("AgentOrchestrator initialized.")

        logger.info("All RAG system components initialized successfully.")
        return {
            "llm_interface": llm_interface,
            "agent_orchestrator": agent_orchestrator
        }
    except Exception as e:
        logger.critical(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise

def run_rag_query(query: str, agent_orchestrator: AgentOrchestrator) -> str:
    """
    Executes a user query through the Advanced RAG pipeline,
    orchestrated by the Agent Orchestrator.
    """
    if not query:
        logger.warning("Received an empty query.")
        return "Please provide a valid query."

    logger.info(f"Processing user query: '{query}'")
    try:
        # The agent orchestrator encapsulates the entire logic flow:
        # query breakdown/reformulation -> iterative/hybrid retrieval -> answer synthesis
        final_answer = agent_orchestrator.process_query(query)

        if not final_answer or final_answer.strip() == "":
            logger.warning(f"Agent Orchestrator returned an empty or invalid answer for query: '{query}'.")
            # Fallback if no specific answer could be generated
            return (f"I couldn't find a precise answer for your query: '{query}'. "
                    "It might involve information beyond my current knowledge base, "
                    "or require further context. Please try rephrasing.")
        
        logger.info(f"Successfully generated final answer for query: '{query}'")
        return final_answer
    except Exception as e:
        logger.error(f"An error occurred during query processing for '{query}': {e}", exc_info=True)
        return "I apologize, an internal error occurred while processing your query. Please try again."

def main():
    """
    Main function to run the Advanced RAG Prototype.
    Initializes components and enters an interactive query loop.
    """
    logger.info("Starting the Advanced RAG Prototype application.")

    components = {}
    try:
        components = initialize_components()
        agent_orchestrator = components["agent_orchestrator"]
        # llm_interface is also available but agent_orchestrator will use it directly.

        print("\n--- Welcome to the Advanced RAG Prototype ---")
        print("This system is designed for complex, logically dependent queries.")
        print("Type 'exit' to quit at any time.")

        while True:
            user_query = input("\nEnter your complex query: ")
            if user_query.lower() == 'exit':
                logger.info("User requested exit. Shutting down.")
                break

            response = run_rag_query(user_query, agent_orchestrator)
            print("\n--- RAG Response ---")
            print(response)
            print("--------------------\n")

    except Exception as e:
        logger.critical(f"A critical error occurred in the main application loop: {e}", exc_info=True)
        print("\nFATAL ERROR: The Advanced RAG system encountered a critical issue and cannot continue.")
        print("Please check the logs for more details.")
    finally:
        logger.info("Advanced RAG Prototype application stopped.")

if __name__ == "__main__":
    main()