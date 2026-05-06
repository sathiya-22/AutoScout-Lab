import os
import sys

# Ensure the project root is in the path for modular imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from config import Config
from llm_client import LLMClient
from data_management.data_loader import DataLoader
from data_management.vector_store_manager import VectorStoreManager
from prompts import retrieval_prompts, generation_prompts, verification_prompts
from retrievers.semantic_retriever import SemanticRetriever
from retrievers.query_rewriting_retriever import QueryRewritingRetriever
from retrievers.multi_stage_retriever import MultiStageRetriever
from retrievers.correctness_verifier import CorrectnessVerifier
from rag_orchestrator import RAGOrchestrator

def main():
    """
    Main entry point for the RAG application.
    Initializes configurations, sets up the RAG pipeline, and handles user queries.
    """
    print("Initializing RAG system...")

    try:
        # 1. Load Configurations
        config = Config()
        print(f"Loaded configuration for LLM: {config.LLM_MODEL_NAME}")

        # 2. Initialize LLM Client
        llm_client = LLMClient(api_key=config.OPENAI_API_KEY, model_name=config.LLM_MODEL_NAME)
        print("LLM Client initialized.")

        # 3. Initialize Data Management
        data_loader = DataLoader(
            raw_data_path=config.RAW_DATA_PATH,
            processed_data_path=config.PROCESSED_CHUNKS_PATH,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        vector_store_manager = VectorStoreManager(
            vector_store_path=config.VECTOR_STORE_PATH,
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            llm_client=llm_client # For embedding model via LLM client or direct embedding model client
        )
        print("Data Management components initialized.")

        # 4. Data Ingestion & Vector Store Preparation
        if not vector_store_manager.vector_store_exists() or config.REBUILD_VECTOR_STORE:
            print("Vector store not found or rebuild requested. Starting data ingestion and vectorization...")
            try:
                processed_chunks = data_loader.load_and_process_documents()
                vector_store_manager.create_or_update_vector_store(processed_chunks)
                print("Vector store created/updated successfully.")
            except Exception as e:
                print(f"Error during data ingestion or vector store creation: {e}")
                sys.exit(1)
        else:
            print("Vector store already exists. Loading existing store.")
            vector_store_manager.load_vector_store()

        # 5. Initialize Retrieval Components
        # Get the base retriever from the vector store manager
        base_vector_retriever = vector_store_manager.get_retriever()

        semantic_retriever = SemanticRetriever(base_vector_retriever)
        query_rewriting_retriever = QueryRewritingRetriever(llm_client, retrieval_prompts.QUERY_REWRITE_PROMPT)
        correctness_verifier = CorrectnessVerifier(llm_client, verification_prompts.VERIFICATION_PROMPT)
        multi_stage_retriever = MultiStageRetriever(
            base_retriever=semantic_retriever, # Use semantic as the initial stage
            query_rewriting_retriever=query_rewriting_retriever,
            llm_client=llm_client,
            rerank_prompt=retrieval_prompts.RERANK_PROMPT,
            num_initial_chunks=config.NUM_INITIAL_CHUNKS,
            num_final_chunks=config.NUM_FINAL_CHUNKS
        )
        print("Retrieval components initialized.")

        # 6. Initialize RAG Orchestrator
        rag_orchestrator = RAGOrchestrator(
            llm_client=llm_client,
            retriever=multi_stage_retriever, # Use the sophisticated multi-stage retriever
            verifier=correctness_verifier,
            generation_prompt_template=generation_prompts.GENERATION_PROMPT
        )
        print("RAG Orchestrator initialized.")
        print("\nSystem ready. Enter your queries below.")

        # 7. Main Query Loop
        while True:
            user_query = input("\nEnter your query (type 'quit' to exit): ")
            if user_query.lower() == 'quit':
                print("Exiting RAG system. Goodbye!")
                break

            if not user_query.strip():
                print("Query cannot be empty. Please enter a valid query.")
                continue

            print("\nProcessing query...")
            try:
                response = rag_orchestrator.answer_query(user_query)
                print("\n--- RAG Response ---")
                print(response)
                print("--------------------\n")
            except Exception as e:
                print(f"An error occurred during query processing: {e}")
                # Optional: print full traceback for debugging
                # import traceback
                # traceback.print_exc()

    except Exception as e:
        print(f"A critical error occurred during system initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()