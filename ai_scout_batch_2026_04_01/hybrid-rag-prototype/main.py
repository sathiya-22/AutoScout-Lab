import os
import uuid
import json
from typing import List, Dict, Any, Optional

# --- ARCHITECTURE MOCK IMPLEMENTATIONS ---
# These are simplified mocks to demonstrate the integration points.
# In a real system, these would be fully implemented in their respective files.

# Layered Ingestion & Parsing
class DocumentLoader:
    """Mocks loading various document formats."""
    def load_document(self, file_path: str) -> Dict[str, Any]:
        print(f"Loading document: {file_path}")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            doc_type = "unknown"
            if file_path.endswith(".txt"):
                doc_type = "text"
            elif file_path.endswith(".md"):
                doc_type = "markdown"
            elif file_path.endswith(".py"):
                doc_type = "code"

            return {"id": str(uuid.uuid4()), "content": content, "type": doc_type, "source": file_path}
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            raise

class TextParser:
    """Mocks standard text chunking and pre-processing."""
    def parse(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Parsing text content for doc_id: {document['id']}")
        content = document.get('content', '')
        if not content:
            return []

        # Simple chunking for demo: split into chunks of max 500 characters
        chunks = []
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": document['id'],
                "content": chunk_content,
                "type": "text_chunk",
                "metadata": {"source": document['source'], "type": document['type'], "start_char": i, "end_char": i + len(chunk_content)}
            })
        return chunks

class CodeASTParser:
    """Mocks generating Abstract Syntax Trees (ASTs) from code."""
    def parse(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Parsing code AST for doc_id: {document['id']}")
        if document.get('type') != 'code':
            return []

        content = document.get('content', '')
        if not content:
            return []

        code_chunks = []
        lines = content.split('\n')
        current_function_body: List[str] = []
        current_function_name: Optional[str] = None
        current_function_start_line: Optional[int] = None

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("def ") or stripped_line.startswith("async def "):
                # New function definition
                if current_function_name: # Store previous function
                    code_chunks.append({
                        "id": str(uuid.uuid4()),
                        "doc_id": document['id'],
                        "content": "\n".join(current_function_body),
                        "type": "code_function",
                        "metadata": {"function_name": current_function_name, "line_start": current_function_start_line, "line_end": i-1, "source": document['source']}
                    })
                
                # Start new function
                parts = stripped_line.split('(')[0].split()
                func_name = parts[-1]
                current_function_name = func_name
                current_function_start_line = i
                current_function_body = [line]
            elif current_function_name:
                current_function_body.append(line)
            else: # Not inside a function, treat as global code block
                code_chunks.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": document['id'],
                    "content": line,
                    "type": "code_global_line",
                    "metadata": {"line_num": i, "source": document['source']}
                })

        # Add the last function if exists
        if current_function_name:
            code_chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": document['id'],
                "content": "\n".join(current_function_body),
                "type": "code_function",
                "metadata": {"function_name": current_function_name, "line_start": current_function_start_line, "line_end": len(lines)-1, "source": document['source']}
            })

        # Mock AST-like information: call graph
        # This would typically be derived from static analysis of the full AST
        if "function_a" in content and "function_b" in content:
            code_chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": document['id'],
                "content": "Call graph: function_a calls function_b",
                "type": "code_call_graph",
                "metadata": {"relationships": [{"source": "function_a", "target": "function_b", "type": "CALLS"}], "source": document['source']}
            })

        return code_chunks

class TabularDataParser:
    """Mocks identifying and extracting tabular data."""
    def parse(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Parsing tabular data for doc_id: {document['id']}")
        content = document.get('content', '')
        if not content:
            return []

        table_data_chunks = []
        lines = content.split('\n')
        in_table = False
        headers: List[str] = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Simple markdown table detection
            if stripped_line.startswith('|') and stripped_line.count('|') > 2:
                if not in_table: # First line of a table (headers)
                    headers = [h.strip() for h in stripped_line.strip('|').split('|')]
                    in_table = True
                elif '---' in stripped_line: # Separator line, ignore
                    continue
                else: # Data row
                    row_values = [v.strip() for v in stripped_line.strip('|').split('|')]
                    if len(row_values) == len(headers):
                        row_dict = {headers[j]: row_values[j] for j in range(len(headers))}
                        table_data_chunks.append({
                            "id": str(uuid.uuid4()),
                            "doc_id": document['id'],
                            "content": json.dumps(row_dict), # Raw content for embedding/context
                            "data": row_dict, # Structured data for precise querying
                            "type": "tabular_row",
                            "metadata": {"line_num": i, "headers": headers, "source": document['source']}
                        })
            elif in_table: # Line after table, but not part of it
                in_table = False
        return table_data_chunks

class KGExtractor:
    """Mocks extracting entities and relationships for a knowledge graph."""
    def extract(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Extracting KG entities/relationships for doc_id: {document['id']}")
        content = document.get('content', '')
        if not content:
            return []

        kg_elements = []
        doc_id = document['id']
        source_doc = document['source']

        # Mock entity and relationship extraction based on keywords
        entities_found = set()
        
        if "RAG system" in content.lower():
            kg_elements.append({"id": "RAG_System", "type": "CONCEPT", "name": "RAG System"})
            entities_found.add("RAG_System")
        if "semantic similarity" in content.lower():
            kg_elements.append({"id": "Semantic_Similarity", "type": "CONCEPT", "name": "Semantic Similarity"})
            entities_found.add("Semantic_Similarity")
            if "RAG_System" in entities_found:
                kg_elements.append({"source": "RAG_System", "target": "Semantic_Similarity", "type": "USES"})
        if "complex reasoning" in content.lower():
            kg_elements.append({"id": "Complex_Reasoning", "type": "CONCEPT", "name": "Complex Reasoning"})
            entities_found.add("Complex_Reasoning")
            if "RAG_System" in entities_found:
                 kg_elements.append({"source": "RAG_System", "target": "Complex_Reasoning", "type": "STRUGGLES_WITH"})

        if document.get('type') == 'code':
            if "function_a" in content:
                kg_elements.append({"id": "function_a", "type": "CODE_FUNCTION", "name": "function_a"})
            if "function_b" in content:
                kg_elements.append({"id": "function_b", "type": "CODE_FUNCTION", "name": "function_b"})
            if "function_a" in content and "function_b" in content and "return function_b" in content:
                kg_elements.append({"source": "function_a", "target": "function_b", "type": "CALLS"})

        formatted_kg_results = []
        for elem in kg_elements:
            if "source" in elem and "target" in elem and "type" in elem: # It's a relationship
                formatted_kg_results.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "type": "kg_relationship",
                    "content": f"{elem['source']} {elem['type']} {elem['target']}",
                    "data": elem,
                    "metadata": {"source_doc": source_doc}
                })
            else: # It's an entity
                formatted_kg_results.append({
                    "id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "type": "kg_entity",
                    "content": elem['name'],
                    "data": elem,
                    "metadata": {"source_doc": source_doc}
                })
        return formatted_kg_results


# Multi-Modal Indexing & Storage
class VectorStoreManager:
    """Mocks a vector store for dense embeddings."""
    def __init__(self):
        self._store: List[Dict[str, Any]] = []

    def add_vectors(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadatas lists must have the same length.")
        print(f"Adding {len(embeddings)} vectors to vector store.")
        for i in range(len(embeddings)):
            self._store.append({"embedding": embeddings[i], "metadata": metadatas[i]})

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        print(f"Searching vector store for top {k} similar items.")
        if not self._store:
            return []
        # Mock search: In reality, use cosine similarity or other metric
        # For demo, just return first k results or random if less than k
        return self._store[:k] if len(self._store) >= k else self._store

class GraphStoreManager:
    """Mocks a graph database for logical relationships."""
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._relationships: List[Dict[str, Any]] = []

    def add_node(self, node_id: str, properties: Dict[str, Any]):
        if node_id not in self._nodes:
            self._nodes[node_id] = {"id": node_id, "properties": properties}
            print(f"Added graph node: {node_id} ({properties.get('type', '')})")

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Optional[Dict[str, Any]] = None):
        # Ensure nodes exist (mock)
        if source_id not in self._nodes:
            self.add_node(source_id, {"type": "UNKNOWN"})
        if target_id not in self._nodes:
            self.add_node(target_id, {"type": "UNKNOWN"})

        self._relationships.append({
            "source": source_id, "target": target_id, "type": rel_type, "properties": properties or {}
        })
        print(f"Added graph relationship: {source_id} -[{rel_type}]-> {target_id}")

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        print(f"Querying graph store with: '{query_str}'")
        results = []
        lower_query = query_str.lower()

        if "concepts used by rag system" in lower_query:
            for rel in self._relationships:
                if rel['source'] == "RAG_System" and rel['type'] == "USES":
                    target_node = self._nodes.get(rel['target'])
                    if target_node:
                        results.append({"node": target_node, "relationship": rel})
        elif "struggles with" in lower_query:
            for rel in self._relationships:
                if rel['source'] == "RAG_System" and rel['type'] == "STRUGGLES_WITH":
                    target_node = self._nodes.get(rel['target'])
                    if target_node:
                        results.append({"node": target_node, "relationship": rel})
        elif "function call relationships" in lower_query:
            for rel in self._relationships:
                if rel['type'] == "CALLS":
                    source_node = self._nodes.get(rel['source'])
                    target_node = self._nodes.get(rel['target'])
                    results.append({"node_source": source_node, "node_target": target_node, "relationship": rel})
        
        return results

class StructuredDataIndexer:
    """A specialized index for tabular data, allowing attribute-based queries."""
    def __init__(self):
        self._index: List[Dict[str, Any]] = [] # Stores rows with their metadata

    def add_row(self, row_data: Dict[str, Any], doc_id: str, chunk_id: str):
        row_data_with_meta = row_data.copy()
        row_data_with_meta["doc_id"] = doc_id
        row_data_with_meta["chunk_id"] = chunk_id
        self._index.append(row_data_with_meta)
        print(f"Added structured data row (chunk_id: {chunk_id[:8]}...) for doc_id: {doc_id[:8]}...")

    def query(self, attribute_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Querying structured data index with filters: {attribute_filters}")
        results = []
        if not attribute_filters:
            return self._index # Return all if no filters (for demo)

        for row in self._index:
            match = True
            for attr, value in attribute_filters.items():
                if attr not in row:
                    match = False
                    break
                
                # Mock comparison: case-insensitive for strings, exact for others
                if isinstance(value, str) and value.lower() != str(row[attr]).lower():
                    match = False
                    break
                elif not isinstance(value, str) and value != row[attr]:
                    match = False
                    break
            if match:
                results.append(row)
        return results

class MetadataStore:
    """Central repository for all document metadata."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def add_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        self._store[doc_id] = metadata
        print(f"Added metadata for doc_id: {doc_id[:8]}...")

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(doc_id)

    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        # In a real system, chunks would be stored with direct links or in a dedicated chunk store.
        # This mock searches all document metadata for a chunk with the given ID.
        for doc_meta in self._store.values():
            if 'chunks' in doc_meta: # If we stored chunks directly here
                for chunk in doc_meta['chunks']:
                    if chunk['id'] == chunk_id:
                        return chunk['metadata']
            # Fallback for structured data chunks (assuming their metadata contains chunk_id)
            if 'structured_chunks' in doc_meta:
                 for chunk in doc_meta['structured_chunks']:
                    if chunk['id'] == chunk_id:
                        return chunk['metadata']
        return None

# Specialized Embedding Generation
class EmbedderFactory:
    """Provides an interface for creating different embedder types."""
    def get_embedder(self, embedder_type: str):
        if embedder_type == "text":
            return TextEmbedder()
        elif embedder_type == "code":
            return CodeEmbedder()
        elif embedder_type == "kg":
            return KGEmbedder()
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

class TextEmbedder:
    """Generates general-purpose text embeddings."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"Generating text embeddings for {len(texts)} texts.")
        # Mock embedding: return dummy vectors
        # In a real system, this would call a model like OpenAI's text-embedding-ada-002
        return [[float(hash(t) % 1000) / 1000.0 for _ in range(768)] for t in texts]

class CodeEmbedder:
    """Generates embeddings specifically optimized for code semantics."""
    def embed(self, code_snippets: List[str]) -> List[List[float]]:
        print(f"Generating code embeddings for {len(code_snippets)} snippets.")
        # Mock embedding: return dummy vectors
        # In a real system, this would use a code-specific model (e.g., CodeBERT)
        return [[float(hash(c) % 1000) / 1000.0 for _ in range(512)] for c in code_snippets]

class KGEmbedder:
    """Generates embeddings for nodes and relationships within the knowledge graph."""
    def embed_nodes(self, node_ids: List[str]) -> List[List[float]]:
        print(f"Generating KG node embeddings for {len(node_ids)} nodes.")
        return [[float(hash(n) % 1000) / 1000.0 for _ in range(128)] for n in node_ids]

    def embed_relationships(self, relationship_triples: List[Dict[str, str]]) -> List[List[float]]:
        print(f"Generating KG relationship embeddings for {len(relationship_triples)} relationships.")
        return [[float(hash(f"{t['source']}{t['type']}{t['target']}") % 1000) / 1000.0 for _ in range(128)] for t in relationship_triples]


# Hybrid Retrieval Orchestration
class VectorRetriever:
    """Performs semantic similarity searches."""
    def __init__(self, vector_store: VectorStoreManager, embedder: TextEmbedder):
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print(f"Vector Retriever: Searching for '{query}' (top {k} results)")
        try:
            query_embedding = self._embedder.embed([query])[0]
            results = self._vector_store.search(query_embedding, k)
            return [{"source": "vector_store", "content": r['metadata']['content'], "metadata": r['metadata']} for r in results if r.get('metadata')]
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []

class KGRetriever:
    """Queries the graph database for logical dependencies, paths, and relationships."""
    def __init__(self, graph_store: GraphStoreManager, kg_embedder: KGEmbedder):
        self._graph_store = graph_store
        self._kg_embedder = kg_embedder # Included for architectural completeness, not used in mock query directly

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        print(f"KG Retriever: Querying graph for '{query}'")
        graph_query_template = ""
        lower_query = query.lower()

        if "rag system uses" in lower_query or "components used in rag" in lower_query:
            graph_query_template = "Find concepts used by RAG System"
        elif "struggle with rag" in lower_query or "limitations of rag" in lower_query:
            graph_query_template = "Find concepts RAG System struggles with"
        elif "function calls" in lower_query or "call graph" in lower_query:
            graph_query_template = "Find function call relationships"
        else:
            return [] # No specific graph query pattern matched

        try:
            results = self._graph_store.query(graph_query_template)
            formatted_results = []
            for r in results:
                if "relationship" in r:
                    source_id = r['relationship']['source']
                    target_id = r['relationship']['target']
                    rel_type = r['relationship']['type']
                    
                    source_name = r.get('node_source', {}).get('properties', {}).get('name', source_id)
                    target_name = r.get('node_target', {}).get('properties', {}).get('name', target_id)
                    
                    formatted_results.append({
                        "source": "graph_store",
                        "content": f"{source_name} {rel_type.replace('_', ' ').lower()} {target_name}",
                        "metadata": r['relationship']
                    })
                elif "node" in r and r.get('node'):
                    node_props = r['node'].get('properties', {})
                    formatted_results.append({
                        "source": "graph_store",
                        "content": f"Node: {node_props.get('name', r['node']['id'])} (Type: {node_props.get('type', 'N/A')})",
                        "metadata": node_props
                    })
            return formatted_results
        except Exception as e:
            print(f"KG retrieval failed: {e}")
            return []

class StructuredDataRetriever:
    """Executes precise attribute-based lookups on the `structured_data_indexer`."""
    def __init__(self, structured_data_indexer: StructuredDataIndexer):
        self._indexer = structured_data_indexer

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        print(f"Structured Data Retriever: Analyzing query for structured attributes: '{query}'")
        attribute_filters: Dict[str, Any] = {}
        lower_query = query.lower()

        # Simple keyword-based parsing for attribute filters
        if "price of" in lower_query and "item named" in lower_query:
            # e.g., "What is the price of an item named 'Laptop X'?"
            # This is a very simplistic regex-like parse for demonstration
            import re
            match = re.search(r"item named '([^']+)'", lower_query)
            if match:
                attribute_filters["Item_Name"] = match.group(1).title() # Convert to title case for matching
                attribute_filters["Price"] = "*" # Indicate interest in price

        elif "customer" in lower_query and "order_id" in lower_query:
            # e.g., "Who is the customer for order_id 'ORD-001'?"
            import re
            match = re.search(r"order_id '([^']+)'", lower_query)
            if match:
                attribute_filters["Order_ID"] = match.group(1)
                attribute_filters["Customer_Name"] = "*" # Indicate interest in customer name
        
        if not attribute_filters:
            return []

        try:
            results = self._indexer.query(attribute_filters)
            formatted_results = []
            for r in results:
                content_parts = []
                for key, value in r.items():
                    # Only include attributes that were filtered or explicitly requested with "*"
                    if key in attribute_filters or attribute_filters.get(key, None) == "*":
                        content_parts.append(f"{key}: {value}")
                
                if content_parts: # Ensure there's content to add
                    formatted_results.append({
                        "source": "structured_data_store",
                        "content": ", ".join(content_parts),
                        "metadata": r
                    })
            return formatted_results
        except Exception as e:
            print(f"Structured data retrieval failed: {e}")
            return []

class HybridOrchestrator:
    """Core intelligence, analyzing user queries to determine the most relevant retrieval strategy."""
    def __init__(self, vector_retriever: VectorRetriever, kg_retriever: KGRetriever,
                 structured_data_retriever: StructuredDataRetriever):
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._structured_data_retriever = structured_data_retriever

    def _determine_strategy(self, query: str) -> List[str]:
        strategies = []
        lower_query = query.lower()
        
        # Rule-based strategy determination
        if "price of" in lower_query or "customer for" in lower_query or "order details" in lower_query or "table" in lower_query:
            strategies.append("structured")
        if "relationship" in lower_query or "calls" in lower_query or "depends on" in lower_query or "graph" in lower_query or "logical dependencies" in lower_query:
            strategies.append("kg")
        # If no specific structured/KG keywords, or for general explanations, use vector search
        if not strategies or "explain" in lower_query or "describe" in lower_query or "what is" in lower_query:
            strategies.append("vector")
        
        # Ensure unique strategies
        return list(set(strategies))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        print(f"\nHybrid Orchestrator: Orchestrating retrieval for query: '{query}'")
        retrieval_strategies = self._determine_strategy(query)
        print(f"Identified retrieval strategies: {', '.join(retrieval_strategies)}")
        all_results = []

        if "vector" in retrieval_strategies:
            all_results.extend(self._vector_retriever.retrieve(query))

        if "kg" in retrieval_strategies:
            all_results.extend(self._kg_retriever.retrieve(query))

        if "structured" in retrieval_strategies:
            all_results.extend(self._structured_data_retriever.retrieve(query))

        # Fusion and re-ranking (mock for now)
        print(f"Hybrid Orchestrator: Fusing and re-ranking {len(all_results)} raw results.")
        
        # Simple deduplication and re-ranking by type priority (structured > kg > vector)
        unique_results_map: Dict[str, Dict[str, Any]] = {} # content -> best_result
        for res in all_results:
            content_key = res['content'] # Use content for deduplication
            existing_res = unique_results_map.get(content_key)

            if existing_res:
                # Prioritize structured > kg > vector
                priority_map = {"structured_data_store": 3, "graph_store": 2, "vector_store": 1}
                current_priority = priority_map.get(res.get('source', ''), 0)
                existing_priority = priority_map.get(existing_res.get('source', ''), 0)
                if current_priority > existing_priority:
                    unique_results_map[content_key] = res # Replace with higher priority result
            else:
                unique_results_map[content_key] = res
        
        final_results = list(unique_results_map.values())
        # Sort for consistent output, e.g., by source type or simply as retrieved
        final_results.sort(key=lambda x: x.get('source', ''))
        
        print(f"Hybrid Orchestrator: Fused and re-ranked into {len(final_results)} unique results.")
        return final_results[:5] # Limit to top 5 results for brevity in context

# Dynamic Context Generation & LLM Integration
class ContextSynthesizer:
    """Synthesizes a coherent and concise context from mixed retrieval results."""
    def synthesize(self, retrieved_results: List[Dict[str, Any]]) -> str:
        print(f"Context Synthesizer: Synthesizing context from {len(retrieved_results)} results.")
        context_parts = []
        for i, result in enumerate(retrieved_results):
            source = result.get('source', 'unknown')
            content = result.get('content', 'No content.')
            metadata = result.get('metadata', {})

            if source == "structured_data_store" and "data" in metadata:
                # Format tabular data nicely for LLM
                data_dict = metadata['data']
                header_row = "| " + " | ".join(data_dict.keys()) + " |"
                separator_row = "|---" * len(data_dict.keys()) + "|"
                data_row = "| " + " | ".join(str(v) for v in data_dict.values()) + " |"
                context_parts.append(
                    f"### Structured Data from Document ID {metadata.get('doc_id', 'N/A')[:8]}...\n"
                    f"{header_row}\n{separator_row}\n{data_row}"
                )
            elif source == "graph_store":
                context_parts.append(
                    f"### Logical Relationship (extracted from {metadata.get('source_doc', 'N/A')}:\n"
                    f"- **Relationship:** {content}\n"
                    f"- **Type:** {metadata.get('type', 'N/A')}"
                )
            elif source == "vector_store":
                context_parts.append(
                    f"### Semantic Text Snippet (from {metadata.get('source', 'N/A')} - Type: {metadata.get('type', 'N/A')}):\n"
                    f"{content}"
                )
            else:
                context_parts.append(f"### Retrieved Information (Source: {source}):\n{content}")
        
        if not context_parts:
            return "No relevant information found in the knowledge base to synthesize context."
            
        return "\n\n---\n\n".join(context_parts)

class PromptBuilder:
    """Constructs the final LLM prompt."""
    def build_prompt(self, user_query: str, context: str) -> str:
        print("Prompt Builder: Building final LLM prompt.")
        prompt = f"""You are an advanced AI assistant capable of complex reasoning based on diverse information.
Your task is to answer the user's query comprehensively and accurately, strictly using the provided context. If the context does not contain enough information to answer, you must state that you don't have enough information.

User Query: {user_query}

Context:
---
{context}
---

Please provide a comprehensive answer based *only* on the context provided, citing which part of the context your answer is derived from if applicable.
"""
        return prompt

class LLMAdapter:
    """Provides an abstract interface for interacting with various LLM providers."""
    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        print(f"\nLLM Adapter: Sending prompt to {self.model_name}...")
        
        # Simulate LLM processing and response based on prompt content
        if "No relevant information found" in prompt:
            return "I apologize, but based on the provided context, I don't have enough information to answer your query."
        
        if "RAG system uses Semantic Similarity" in prompt and "struggles with Complex Reasoning" in prompt:
            return "Based on the context, RAG systems use Semantic Similarity but fundamentally struggle with Complex Reasoning and logical dependencies like cause-effect. This limitation prevents them from performing advanced reasoning beyond surface-level embedding similarity."
        elif "Laptop X" in prompt and "Price" in prompt:
            return "According to the structured data provided, the 'Laptop X' has a Price of '1200'."
        elif "function_a calls function_b" in prompt:
            return "The logical relationship extracted from the code indicates that 'function_a' calls 'function_b'."
        elif "problem statement highlights that RAG systems struggle" in prompt:
            return "The context indicates that RAG systems struggle with capturing deep semantic relationships and logical dependencies (e.g., cause-effect, pre-requisites, code call graphs) beyond surface-level embedding similarity, which impairs their ability to perform complex reasoning."
        else:
            return "This is a mock LLM response. The system successfully retrieved and synthesized diverse information. I can see the various data points (text, structured data, logical relationships) in the provided context, which would enable me to answer specific questions related to them."

# --- MAIN APPLICATION LOGIC ---

def setup_environment():
    """Create dummy files for demonstration."""
    os.makedirs("data", exist_ok=True)
    with open("data/document1.txt", "w", encoding='utf-8') as f:
        f.write("""
The RAG system described here fundamentally struggles with capturing deep semantic relationships and logical dependencies (e.g., cause-effect, pre-requisites, code call graphs) beyond surface-level embedding similarity. This leads to an inability to perform complex reasoning. The system, however, uses semantic similarity for initial retrieval.
Another challenge is effectively integrating specific structured or semi-structured data like database rows or tabular data within documents without confusing the LLM or losing critical context.
        """)
    with open("data/document2.md", "w", encoding='utf-8') as f:
        f.write("""
# Product Information

Here's a table of our popular products:

| Item_Name | Category    | Price | In_Stock |
|-----------|-------------|-------|----------|
| Laptop X  | Electronics | 1200  | Yes      |
| Keyboard Y| Peripherals | 75    | Yes      |
| Mouse Z   | Peripherals | 25    | No       |

This table provides important details for our sales team.
        """)
    with open("data/sample_code.py", "w", encoding='utf-8') as f:
        f.write("""
# A sample Python module
import logging

logging.basicConfig(level=logging.INFO)

def initialize_data(source_path: str) -> dict:
    \"\"\"Loads and processes initial data.\"\"\"
    logging.info(f"Initializing data from {source_path}")
    data = {"path": source_path, "status": "initialized"}
    return data

def process_data(data: dict) -> dict:
    \"\"\"Performs some processing on the data.\"\"\"
    logging.info("Processing data...")
    data["status"] = "processed"
    return data

def analyze_results(processed_data: dict):
    \"\"\"Analyzes the processed data and prints a summary.\"\"\"
    logging.info("Analyzing results...")
    print(f"Analysis complete for: {processed_data.get('path', 'unknown')}")

def main_workflow():
    \"\"\"Orchestrates the main data processing workflow.\"\"\"
    initial_data = initialize_data("my_source.csv")
    processed_data = process_data(initial_data)
    analyze_results(processed_data)
    logging.info("Workflow complete.")

if __name__ == "__main__":
    main_workflow()
        """)

def cleanup_environment():
    """Remove dummy files."""
    print("\nCleaning up dummy files...")
    for f_name in ["document1.txt", "document2.md", "sample_code.py"]:
        f_path = os.path.join("data", f_name)
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"Error removing {f_path}: {e}")
    if os.path.exists("data"):
        try:
            os.rmdir("data")
        except OSError as e:
            print(f"Error removing directory data: {e}")
    print("Cleanup complete.")


def main():
    print("Initializing Hybrid Semantic and Logical Retrieval System...\n")

    # --- 1. Initialize Components ---
    # Ingestion & Parsing
    doc_loader = DocumentLoader()
    text_parser = TextParser()
    code_ast_parser = CodeASTParser()
    tabular_data_parser = TabularDataParser()
    kg_extractor = KGExtractor()

    # Specialized Embedding Generation
    embedder_factory = EmbedderFactory()
    text_embedder = embedder_factory.get_embedder("text")
    code_embedder = embedder_factory.get_embedder("code")
    kg_embedder = embedder_factory.get_embedder("kg")

    # Multi-Modal Indexing & Storage
    vector_store_manager = VectorStoreManager()
    graph_store_manager = GraphStoreManager()
    structured_data_indexer = StructuredDataIndexer()
    metadata_store = MetadataStore()

    # Hybrid Retrieval Orchestration
    vector_retriever = VectorRetriever(vector_store_manager, text_embedder)
    kg_retriever = KGRetriever(graph_store_manager, kg_embedder)
    structured_data_retriever = StructuredDataRetriever(structured_data_indexer)
    hybrid_orchestrator = HybridOrchestrator(
        vector_retriever, kg_retriever, structured_data_retriever
    )

    # Dynamic Context Generation & LLM Integration
    context_synthesizer = ContextSynthesizer()
    prompt_builder = PromptBuilder()
    llm_adapter = LLMAdapter()

    print("Components initialized successfully.\n")

    # --- 2. Ingest and Index Example Documents ---
    print("--- INGESTION & INDEXING PHASE ---")
    documents_to_process = [
        "data/document1.txt",
        "data/document2.md",
        "data/sample_code.py"
    ]

    for doc_path in documents_to_process:
        try:
            document = doc_loader.load_document(doc_path)
            doc_id = document['id']
            print(f"\nProcessing document: {doc_path} (ID: {doc_id[:8]}...)")

            # Store initial document metadata
            metadata_store.add_metadata(doc_id, {"source": doc_path, "type": document['type'], "parsed_chunks": []})

            all_chunks_for_doc: List[Dict[str, Any]] = []

            # Parse text/markdown documents
            if document['type'] == 'text' or document['type'] == 'markdown':
                text_chunks = text_parser.parse(document)
                all_chunks_for_doc.extend(text_chunks)
                
                tabular_chunks = tabular_data_parser.parse(document)
                for tab_chunk in tabular_chunks:
                    structured_data_indexer.add_row(tab_chunk['data'], doc_id, tab_chunk['id'])
                all_chunks_for_doc.extend(tabular_chunks)

            # Parse code documents
            if document['type'] == 'code':
                code_chunks = code_ast_parser.parse(document)
                all_chunks_for_doc.extend(code_chunks)
            
            # Extract KG elements for all document types
            kg_elements = kg_extractor.extract(document)
            for kg_elem in kg_elements:
                if kg_elem['type'] == 'kg_entity':
                    graph_store_manager.add_node(kg_elem['data']['id'], kg_elem['data'])
                elif kg_elem['type'] == 'kg_relationship':
                    graph_store_manager.add_relationship(
                        kg_elem['data']['source'], kg_elem['data']['target'], kg_elem['data']['type']
                    )
                all_chunks_for_doc.append(kg_elem) # Also store KG elements as chunks for metadata/vector indexing

            # Update document metadata with all parsed chunk info
            doc_meta = metadata_store.get_metadata(doc_id)
            if doc_meta:
                doc_meta['parsed_chunks'] = all_chunks_for_doc
                metadata_store.add_metadata(doc_id, doc_meta) # Re-add to update

            # Generate embeddings and add to vector store for relevant chunk types
            text_chunks_for_embed = [c for c in all_chunks_for_doc if 'text' in c['type'] or 'markdown' in c['type'] or 'tabular_row' in c['type'] or 'kg_entity' in c['type'] or 'kg_relationship' in c['type']]
            if text_chunks_for_embed:
                text_contents = [c['content'] for c in text_chunks_for_embed]
                if text_contents:
                    text_embeddings = text_embedder.embed(text_contents)
                    vector_store_manager.add_vectors(text_embeddings, text_chunks_for_embed)

            code_chunks_for_embed = [c for c in all_chunks_for_doc if 'code' in c['type']]
            if code_chunks_for_embed:
                code_contents = [c['content'] for c in code_chunks_for_embed]
                if code_contents:
                    code_embeddings = code_embedder.embed(code_contents)
                    # For demo, add code embeddings to the same vector store as text
                    vector_store_manager.add_vectors(code_embeddings, code_chunks_for_embed)
            
            print(f"Indexed {len(all_chunks_for_doc)} total chunks (text, tabular, code, KG) for {doc_path}.")
            
        except FileNotFoundError as e:
            print(f"Skipping document {doc_path}: {e}")
        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")

    print("\n--- INGESTION & INDEXING PHASE COMPLETE ---\n")

    # --- 3. Query and Retrieve Example ---
    print("--- QUERY & RETRIEVAL PHASE ---")

    queries = [
        "Explain how RAG systems struggle with complex reasoning and what they use for initial retrieval.",
        "What is the price of an item named 'Laptop X'?",
        "Show me the function call relationships in the sample code.",
        "Describe the limitations of RAG systems regarding logical dependencies.",
        "What is 'Mouse Z'?"
    ]

    for i, user_query in enumerate(queries):
        print(f"\n--- Processing Query {i+1}: {user_query} ---")
        try:
            retrieved_results = hybrid_orchestrator.retrieve(user_query)

            # --- 4. Context Generation & LLM Integration ---
            synthesized_context = context_synthesizer.synthesize(retrieved_results)
            final_prompt = prompt_builder.build_prompt(user_query, synthesized_context)
            llm_response = llm_adapter.generate_response(final_prompt)

            print(f"\nAI Assistant Response:\n{llm_response}")
            print("\n-------------------------------------------------\n")

        except Exception as e:
            print(f"An unexpected error occurred during query processing for '{user_query}': {e}")


    print("\nHybrid Semantic and Logical Retrieval System Demo Finished.")

if __name__ == "__main__":
    setup_environment()
    try:
        main()
    finally:
        cleanup_environment()