# Hybrid Semantic and Logical Retrieval System for Advanced RAG

## Project Overview

Traditional Retrieval-Augmented Generation (RAG) systems often fall short when dealing with complex information. They struggle to capture deep semantic relationships and logical dependencies (e.g., cause-effect, pre-requisites, code call graphs) beyond surface-level embedding similarity, which limits their ability to perform sophisticated reasoning. Furthermore, current chunking and embedding approaches frequently fail to effectively integrate and query specific structured or semi-structured data (like individual database rows or tabular data within documents) without confusing the Language Model (LLM) or losing critical contextual details.

## Solution Sketch

This project implements a 'Hybrid Semantic and Logical Retrieval' system designed to overcome these limitations. It achieves this by combining various advanced retrieval techniques:
*   **Vector Search:** For general semantic similarity.
*   **Knowledge Graph Embeddings:** For navigating and querying logical relationships and dependencies.
*   **Abstract Syntax Tree (AST) Analysis:** Specifically for code, to understand structural and logical dependencies (e.g., function calls, variable scope).
*   **Specialized Parsers for Tabular Data:** To identify, extract, and index individual rows/records/fields within larger document chunks. This enables precise, attribute-based retrieval (e.g., using 'SQL-on-text' or schema-guided extraction) before dynamic and contextual injection into the LLM's prompt.

This hybrid approach ensures that the LLM receives a rich, logically coherent, and precisely targeted context, enabling more accurate and complex reasoning.

## Architecture

The 'Hybrid Semantic and Logical Retrieval' system is designed with a modular, layered architecture to integrate diverse information types and retrieval mechanisms, focusing on precision, semantic depth, and logical reasoning.

### 1. Layered Ingestion & Parsing

This layer is responsible for loading documents and extracting various types of information from them through specialized parsers.

*   `src/ingestion/document_loader.py`: Handles loading documents from various formats such as PDF, Markdown, Code files, plain text, etc.
*   `src/ingestion/parsers/`: Contains specialized parsers for different content types:
    *   `text_parser.py`: Standard text chunking, tokenization, and pre-processing for general textual content.
    *   `code_ast_parser.py`: Generates Abstract Syntax Trees (ASTs) from code, extracting crucial elements like functions, classes, method calls, and their logical dependencies to understand code structure.
    *   `tabular_data_parser.py`: Identifies and extracts tabular data (e.g., from HTML tables, Markdown tables, CSV content), tracking row/column metadata and cell values for precise attribute-based retrieval.
    *   `kg_extractor.py`: Extracts entities, relationships, and their attributes from ingested text and code to populate a knowledge graph.

### 2. Multi-Modal Indexing & Storage

This layer manages the storage and indexing of the parsed information in formats optimized for different retrieval strategies.

*   `src/indexing/vector_store_manager.py`: Manages dense vector embeddings for efficient semantic similarity search across text and code snippets. Supports various vector database backends.
*   `src/indexing/graph_store_manager.py`: Manages a graph database (e.g., Neo4j, ArangoDB) for storing and querying logical relationships, dependencies, and paths extracted by KG and AST parsers.
*   `src/indexing/structured_data_indexer.py`: A specialized index designed for tabular data, enabling attribute-based queries (e.g., 'SQL-on-text' capabilities) on individual rows or specific fields within documents.
*   `src/indexing/metadata_store.py`: A central repository for all document metadata, chunk relationships, schema information, and cross-references between different data stores.

### 3. Specialized Embedding Generation

This layer generates different types of embeddings tailored to the nature of the data, enhancing retrieval accuracy.

*   `src/embeddings/embedder_factory.py`: Provides an interface for creating and managing different embedder types (e.g., for text, code, knowledge graph).
*   `src/embeddings/text_embedder.py`: Generates general-purpose semantic embeddings for textual content, optimized for natural language understanding.
*   `src/embeddings/code_embedder.py`: Generates embeddings specifically optimized for code semantics, understanding syntax, structure, and potential functionality.
*   `src/embeddings/kg_embedder.py`: Generates embeddings for nodes and relationships within the knowledge graph, enabling vector-based graph queries and relationship similarity.

### 4. Hybrid Retrieval Orchestration

This is the core intelligence layer, responsible for analyzing queries, selecting appropriate retrieval mechanisms, and combining results.

*   `src/retrieval/vector_retriever.py`: Performs semantic similarity searches against the `vector_store_manager`.
*   `src/retrieval/kg_retriever.py`: Queries the `graph_store_manager` for logical dependencies, causal paths, and specific relationships.
*   `src/retrieval/structured_data_retriever.py`: Executes precise attribute-based lookups and queries on the `structured_data_indexer` for tabular data.
*   `src/retrieval/hybrid_orchestrator.py`: The central component. It analyzes user queries to determine the most relevant retrieval strategy (e.g., semantic, logical, attribute-based). It orchestrates parallel or sequential calls to specialized retrievers, then intelligently fuses, de-duplicates, and re-ranks the diverse results based on relevance and type.

### 5. Dynamic Context Generation & LLM Integration

This layer processes the retrieved information, synthesizes a coherent context, and prepares the final prompt for the LLM.

*   `src/context_generation/context_synthesizer.py`: Synthesizes a coherent, concise, and non-redundant context from the mixed retrieval results (e.g., relevant text chunks, graph snippets, structured data rows/fields) for the LLM, ensuring logical flow.
*   `src/context_generation/prompt_builder.py`: Constructs the final LLM prompt. It carefully integrates the synthesized context, the original user query, and any necessary formatting for structured data (e.g., markdown tables, key-value pairs) to prevent misinterpretation and guide the LLM's reasoning.
*   `src/llm_interface/llm_adapter.py`: Provides an abstract interface for seamless interaction with various LLM providers (e.g., OpenAI, Anthropic, local open-source models), abstracting away API specifics.

## Workflow

1.  **Ingestion:** Documents (text, code, tabular data) are loaded by `document_loader.py`.
2.  **Parsing:** Specialized parsers (`text_parser.py`, `code_ast_parser.py`, `tabular_data_parser.py`, `kg_extractor.py`) process the documents, extracting chunks, ASTs, tabular data, and knowledge graph entities/relationships.
3.  **Embedding:** `embedder_factory.py` orchestrates `text_embedder.py`, `code_embedder.py`, and `kg_embedder.py` to generate appropriate embeddings for different data types.
4.  **Indexing:** The extracted data and their embeddings are stored in their respective, optimized indexes: `vector_store_manager.py` (for semantic text/code), `graph_store_manager.py` (for logical relationships), and `structured_data_indexer.py` (for attribute-based tabular data). Metadata is managed by `metadata_store.py`.
5.  **Query & Orchestration:** Upon a user query, `hybrid_orchestrator.py` intelligently analyzes the query intent and selects the most relevant retrieval strategy. It executes calls to `vector_retriever.py`, `kg_retriever.py`, and/or `structured_data_retriever.py` in parallel or sequentially.
6.  **Fusion & Synthesis:** The diverse retrieved information is fused, re-ranked, and then processed by `context_synthesizer.py` to create a coherent and comprehensive context.
7.  **Prompt Construction:** `prompt_builder.py` constructs the final LLM prompt, carefully integrating the synthesized context, user query, and structured formatting.
8.  **LLM Interaction:** `llm_adapter.py` sends the prompt to the LLM and receives the generated response.

This comprehensive workflow enables the system to handle complex queries requiring deep semantic understanding, logical reasoning, and precise data retrieval across diverse information types.

## Getting Started (Placeholder)

Detailed setup and installation instructions will be provided here.

### Prerequisites (Placeholder)

*   Python 3.x
*   (List specific libraries, e.g., FastAPI, Neo4j driver, embedding models)

### Installation (Placeholder)

```bash
# Clone the repository
git clone https://github.com/your-org/hybrid-rag.git
cd hybrid-rag

# Install dependencies
pip install -r requirements.txt

# Setup graph database (e.g., Neo4j)
# (Instructions for local setup or connecting to a cloud instance)

# Setup vector database (e.g., Chroma, Weaviate, Milvus)
# (Instructions for local setup or connecting to a cloud instance)
```

## Usage (Placeholder)

Examples of how to ingest documents and query the system will be provided here.

```python
# Example Python usage
from src.ingestion.document_loader import DocumentLoader
from src.retrieval.hybrid_orchestrator import HybridOrchestrator
from src.llm_interface.llm_adapter import LLMAdapter

# (Assume setup and indexing are complete)

# Load documents (example)
# loader = DocumentLoader()
# loader.load_and_process_document("path/to/my_code.py", doc_type="code")
# loader.load_and_process_document("path/to/report.pdf", doc_type="pdf")

# Initialize orchestrator and LLM adapter
orchestrator = HybridOrchestrator()
llm_adapter = LLMAdapter()

user_query = "Explain the `calculate_total` function's dependencies and how it relates to the sales data from the Q3 report table."

# Get relevant context
retrieved_context = orchestrator.retrieve_context(user_query)

# Build prompt and query LLM
final_prompt = f"User Query: {user_query}\n\nContext: {retrieved_context}\n\nAnswer:"
llm_response = llm_adapter.generate_response(final_prompt)

print(llm_response)
```

## Contributing (Placeholder)

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to get involved.

## License (Placeholder)

This project is licensed under the MIT License - see the `LICENSE` file for details.