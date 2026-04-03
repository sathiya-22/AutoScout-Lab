# Intelligent RAG & Knowledge Graph Fusion System

## 🚀 Project Overview

Current Retrieval Augmented Generation (RAG) systems struggle with complex information retrieval tasks. Their reliance on surface-level semantic similarity and basic chunking often leads to inaccuracies and irrelevance when dealing with:
*   **Logical Dependency Understanding:** Failing to connect causally or sequentially related pieces of information.
*   **Precise Numerical Computation:** Inability to accurately extract and process numerical data or perform calculations.
*   **Multi-Entity Relationship Mapping:** Difficulty in understanding intricate connections between multiple entities.
*   **Source Authority & Freshness:** Neglecting to prioritize authoritative and up-to-date information, leading to outdated or untrustworthy results.

This prototype addresses these critical limitations by developing an "Intelligent RAG & Knowledge Graph Fusion System."

## ✨ Solution & Key Features

Our system is designed to overcome the challenges of conventional RAG by integrating advanced data ingestion, multi-modal retrieval strategies, agentic query refinement, and robust metadata management.

**Key Features:**

1.  **Multi-Modal/Multi-Strategy Retrieval:** Combines the strengths of semantic search, keyword search, knowledge graph queries, and structured data queries to provide comprehensive and precise context.
2.  **Agentic Query Refinement:** Leverages an LLM-powered agent to iteratively analyze, decompose, and refine user queries, ensuring more targeted and effective retrieval.
3.  **Source Authority & Freshness Integration:** Incorporates critical metadata during retrieval and ranking to prioritize authoritative and current information.
4.  **Granular Data Ingestion:** Advanced strategies to preserve logical units, relationships, and structural integrity during data loading and chunking.
5.  **Knowledge Graph Fusion:** Extracts and utilizes a robust knowledge graph to understand multi-entity relationships and logical dependencies.

## 🏗️ Architecture Overview

The Intelligent RAG & Knowledge Graph Fusion System prototype is designed with **modularity** and **extensibility** as core principles. It integrates sophisticated data handling, multi-strategy retrieval, agentic refinement, and source metadata considerations to directly address the limitations of current RAG systems.

The system is organized into distinct modules, each responsible for a specific aspect of the RAG pipeline, from data ingestion to final response generation.

## 📁 Project Structure

```
.
├── main.py
├── config/
│   └── settings.py
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── chunking_strategy.py
│   │   ├── metadata_extractor.py
│   │   ├── knowledge_graph_builder.py
│   │   └── data_processor.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── kg_store.py
│   │   ├── structured_data_store.py
│   │   ├── hybrid_retriever.py
│   │   └── metadata_filter_ranker.py
│   ├── agents/
│   │   ├── query_refinement_agent.py
│   │   └── agent_tools.py
│   ├── generation/
│   │   ├── llm_interface.py
│   │   └── response_synthesizer.py
│   └── utils/
│       ├── schemas.py
│       ├── logger.py
│       ├── embedding_model.py
│       └── constants.py
│   └── evals/
│       ├── evaluation_metrics.py
│       └── rag_evaluator.py
└── README.md
```

### 1. Granular Data Ingestion (`src/ingestion/`)

This module focuses on intelligently preparing data for retrieval, preserving context and relationships.

*   `document_loader.py`: Handles loading diverse document types (PDFs, web pages, databases) while preserving inherent structure. Includes basic error handling for file access and format parsing.
*   `chunking_strategy.py`: Implements advanced chunking beyond basic token limits, focusing on logical units, relationships, and semantic coherence to mitigate issues with logical dependency understanding. Handles cases where documents are too large or too small for optimal chunking.
*   `metadata_extractor.py`: Extracts critical metadata such as source authority (e.g., publication reputation, author credentials), freshness (last updated date), and data types (numerical, textual) directly during ingestion. This metadata is crucial for later retrieval filtering and ranking. Includes fallback mechanisms for missing metadata.
*   `knowledge_graph_builder.py`: Extracts entities, relationships, attributes, and events from ingested text and structured data, constructing a robust knowledge graph. This component is key for understanding multi-entity relationships and logical dependencies. Manages graph integrity and conflict resolution.
*   `data_processor.py`: Orchestrates the entire ingestion pipeline, taking raw data through loading, chunking, metadata extraction, embedding (via `src/utils/embedding_model.py`), and populating various stores (vector, KG, structured). Includes error handling for pipeline failures and retry mechanisms.

### 2. Multi-Modal/Multi-Strategy Retrieval (`src/retrieval/`)

This module is responsible for fetching relevant information from various data stores using diverse query methods.

*   `vector_store.py`: Manages interactions with vector databases for semantic similarity search on text embeddings. Implements basic error handling for database connection issues and query failures.
*   `kg_store.py`: Provides an interface to query the knowledge graph (e.g., SPARQL, Cypher-like queries, KG embedding search) for precise relationship mapping, logical paths, and factual lookup. Includes error handling for malformed queries or graph traversal issues.
*   `structured_data_store.py`: Handles queries against structured databases (SQL, NoSQL tables) for precise numerical computations and specific data points. Incorporates connection management and query exception handling.
*   `hybrid_retriever.py`: Acts as the central orchestrator, intelligently combining results from semantic search, keyword search (potentially integrated within `vector_store.py` or as a separate component), knowledge graph queries, and structured data queries. It can employ techniques like re-ranking or fusion to blend results effectively. Handles cases where one retrieval method fails gracefully.
*   `metadata_filter_ranker.py`: Post-processes results from the `hybrid_retriever`, applying filters and ranking based on extracted source authority, freshness, and relevance scores, ensuring authoritative and up-to-date information is prioritized. Manages edge cases with missing or conflicting metadata.

### 3. Agentic Query Refinement (`src/agents/`)

This module introduces an LLM-powered agent to enhance query understanding and execution.

*   `query_refinement_agent.py`: An LLM-powered agent responsible for iteratively analyzing the initial user query, identifying ambiguity or missing context, and generating refined sub-queries or follow-up questions. It can decompose complex questions into simpler, actionable parts. Includes mechanisms to handle LLM response failures or irrelevant suggestions.
*   `agent_tools.py`: Defines the 'tools' available to the `query_refinement_agent`. These tools expose functionalities from the `hybrid_retriever` (e.g., 'search_semantic', 'query_knowledge_graph', 'lookup_structured_data') allowing the agent to dynamically interact with the retrieval system and explore the knowledge base. Includes robust tool invocation and error feedback to the agent.

### 4. Generation (`src/generation/`)

This module is responsible for synthesizing a coherent and accurate answer from the retrieved context.

*   `llm_interface.py`: Provides a standardized interface for interacting with various LLM providers, abstracting away API calls and model specifics. Includes retry logic and error handling for API communication issues.
*   `response_synthesizer.py`: Takes the refined query, the retrieved context (including source metadata), and generates a coherent, accurate, and contextually rich answer using the LLM. It's responsible for citing sources and ensuring factual grounding. Incorporates mechanisms to detect and mitigate hallucination, and handle empty or insufficient context.

### 5. Core & Utilities (`main.py`, `config/`, `src/utils/`, `src/evals/`)

These provide the foundational components and supporting functionalities for the entire system.

*   `main.py`: The application's entry point, orchestrating the user query flow through the agent, retrieval, and generation modules. Manages the overall lifecycle and user interaction.
*   `config/`: Contains all system configurations, API keys, database connection strings, and LLM model parameters for easy management.
    *   `settings.py`: Centralized configuration management.
*   `src/utils/`: Includes common helper functions, schema definitions, logging, embedding model management, and global constants.
    *   `schemas.py`: Defines data structures and Pydantic models for consistent data handling.
    *   `logger.py`: Standardized logging utility for the entire application.
    *   `embedding_model.py`: Manages the loading and usage of embedding models, including fallback options.
    *   `constants.py`: Stores global constants and magic strings.
*   `src/evals/`: Dedicated for evaluating the system's performance.
    *   `evaluation_metrics.py`: Defines custom metrics for retrieval relevance, factual accuracy, and response quality.
    *   `rag_evaluator.py`: Orchestrates the running of evaluations against predefined datasets and metrics.

## 🛠️ Setup and Installation (Placeholder)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/intelligent-rag-kg-fusion.git
    cd intelligent-rag-kg-fusion
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure environment variables:**
    Create a `.env` file or populate `config/settings.py` with necessary API keys (for LLMs, vector DBs, etc.) and database connection strings.

## 🚀 Usage (Placeholder)

To run the main application:
```bash
python main.py
```
Further instructions on how to interact with the system (e.g., via a command-line interface or a simple web interface) will be provided as development progresses.

## 🔬 Technologies Used (Implied)

*   **Python 3.9+**
*   **Large Language Models (LLMs):** OpenAI, Anthropic, or similar API-based models.
*   **Vector Databases:** Pinecone, Weaviate, Chroma, Qdrant, etc.
*   **Knowledge Graph Databases:** Neo4j, ArangoDB, Amazon Neptune, RDF stores (e.g., Virtuoso).
*   **Structured Databases:** PostgreSQL, MongoDB, etc.
*   **Data Processing:** LangChain, LlamaIndex, spaCy, NLTK.
*   **PDF/Web Processing:** unstructured, BeautifulSoup, Playwright.