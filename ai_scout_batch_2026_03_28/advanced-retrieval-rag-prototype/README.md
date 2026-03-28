# Advanced RAG for Complex Queries and Logical Dependencies

## 1. Core Objective

This project aims to build a robust RAG (Retrieval Augmented Generation) system capable of understanding and answering complex queries that involve logical reasoning, multi-hop relationships, and structured information by moving beyond simplistic semantic similarity.

## 2. Key Challenges Addressed

*   **Logical Reasoning:** Overcoming the inherent difficulty of vector search in inferring logical connections (e.g., causality, temporal sequences, conditional statements).
*   **Complex Queries:** Handling queries requiring multiple steps of information retrieval and synthesis (e.g., 'What are the implications if X occurs, given Y and Z?').
*   **Context-Awareness:** Ensuring chunking and retrieval preserve crucial document structure and logical flow.

## 3. System Architecture & Flow

This prototype addresses the semantic limitations of traditional embedding-based RAG for complex queries and logical dependencies. It integrates advanced retrieval methods, knowledge representation, and agent-driven query processing.

### Data Ingestion & Preprocessing (`src/ingestion/`)

This module is responsible for preparing raw documents for indexing and retrieval.

*   `document_loader.py`: Handles loading diverse document types (web pages, PDFs, text files).
*   `preprocessing.py`: Cleans and normalizes raw text.
*   `chunking_strategies.py`: Implements advanced, context-aware chunking. This goes beyond fixed-size chunks, considering document structure (headings, sections), semantic boundaries, and explicit logical markers to create more meaningful retrieval units.
*   `graph_extractor.py`: Extracts entities, relationships, events, and potentially logical predicates from documents. This forms the basis for the knowledge graph. Initial prototypes may use rule-based extraction or LLM-driven entity/relation extraction.

### Knowledge Representation (`data/`)

Where processed information is stored and managed.

*   **Vector Store:** For storing embedding vectors of text chunks, enabling semantic similarity search (managed by `src/models/embedding_manager.py`).
*   **Knowledge Graph (KG):** Stores extracted entities and their relationships. For a prototype, this could be a simple graph database (e.g., Neo4j, or a graph structure managed in-memory/SQLite) used by `data/graph_db/`.
*   `data/documents/`: Stores original source documents.
*   `data/processed/`: Stores processed chunks and intermediate data.

### Model Interfaces (`src/models/`)

Provides unified interfaces for interacting with various AI models.

*   `embedding_manager.py`: Standardized interface for various embedding models (e.g., OpenAI, Hugging Face sentence-transformers).
*   `llm_interface.py`: Unified API wrapper for different LLMs, used for query reformulation, summarization, and final answer generation.

### Advanced Retrieval Layer (`src/retrieval/`)

The core of our advanced retrieval capabilities, combining multiple strategies.

*   `vector_retriever.py`: Standard embedding-based search against the vector store.
*   `keyword_retriever.py`: Implements keyword-based search (e.g., BM25, TF-IDF) for precise matches.
*   `graph_retriever.py`: Queries the knowledge graph based on entities and relationships identified in the user query, facilitating multi-hop and relational retrieval.
*   `logical_structure_processor.py`: A specialized component designed to infer or explicitly process logical structures within documents. This might involve rule-based parsing for 'if-then' statements, causal links, or utilizing a fine-tuned model to identify logical dependencies across chunks. It retrieves relevant information based on these inferred structures.
*   `hybrid_retriever.py`: Orchestrates and combines results from the vector, keyword, graph, and logical structure retrievers. It employs strategies (e.g., re-ranking, weighted fusion, iterative querying) to produce a comprehensive set of relevant documents/facts.

### Agent-Driven Query Processing (`src/query_processing/`)

The intelligent layer that orchestrates the overall query answering process.

*   `query_rewriter.py`: An LLM-powered module that analyzes complex user queries, breaks them down into sub-queries, identifies key entities and logical operators, and reformulates queries for optimal retrieval across different modalities.
*   `agent_orchestrator.py`: The central intelligence unit. It manages the iterative interaction between query reformulation and retrieval. It decides which retrieval method(s) to use for sub-queries, evaluates intermediate results, and potentially asks clarifying questions or initiates further retrieval steps based on the user's initial query and retrieved context.

### Utilities & Configuration (`src/utils/`, `config.py`)

Supporting modules for common tasks and project configuration.

*   `src/utils/helpers.py`: General utility functions.
*   `src/utils/logger.py`: Centralized logging setup.
*   `config.py`: Stores API keys, model names, database connection strings, and other configurable parameters.

### Entry Point (`main.py`)

*   `main.py`: Orchestrates the overall RAG pipeline, taking user queries, invoking the agent, managing retrieval, and synthesizing the final response.

### Testing (`tests/`)

*   `tests/`: Includes unit and integration tests for each component to ensure correctness and validate the complex interactions within the system.

## 4. Technology Stack (Prototype)

*   **Language:** Python
*   **Vector DB:** Faiss, ChromaDB, or Weaviate for efficient vector search.
*   **Graph DB:** Neo4j (for production) or NetworkX/SQLite for a simpler prototype graph representation.
*   **LLM Frameworks:** LangChain, LlamaIndex, or direct API integrations (OpenAI, Anthropic, Hugging Face models).
*   **Keyword Search:** `rank_bm25` or custom TF-IDF for simplicity, or Elasticsearch for more robust indexing.

## 5. Expected Outcome

A functional prototype demonstrating superior retrieval accuracy and reasoning capabilities for complex, logically dependent queries compared to standard embedding-only RAG systems.

---

## Setup & Installation

**(Placeholder - detailed instructions to be added during implementation)**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/advanced-rag-prototype.git
    cd advanced-rag-prototype
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure environment variables:**
    Rename `config.example.py` to `config.py` and update with your API keys (e.g., OpenAI, Hugging Face) and database connection strings.

## Usage

**(Placeholder - example usage to be added)**

1.  **Ingest documents:**
    ```bash
    python main.py --action ingest --document_path path/to/your/document.pdf
    ```
2.  **Query the system:**
    ```bash
    python main.py --action query --text "What are the implications if X occurs, given Y and Z?"
    ```

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` for more details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.