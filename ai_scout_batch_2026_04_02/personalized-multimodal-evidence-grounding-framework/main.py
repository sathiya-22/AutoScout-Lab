import os
import logging
from typing import List, Dict, Any, Optional

# --- Mock Implementations for PMEG Framework Components ---
# These mocks simulate the behavior of the modules described in the architecture.
# In a full implementation, these would be proper classes imported from their
# respective files (e.g., `from config import Config`, `from utils.data_models import StandardizedOutput`).

# Mock Configuration (config.py)
class Config:
    VECTOR_DB_URL: str = "mock_vector_db_path"
    GRAPH_DB_URL: str = "mock_graph_db_path"
    RELATIONAL_DB_URL: str = "mock_relational_db_path"
    LOG_LEVEL: str = "INFO" # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    PROCESSING_DIR: str = "temp_pmeg_processing"

# Mock Data Models (utils/data_models.py)
class StandardizedOutput:
    """Represents a standardized output from a perception agent."""
    def __init__(self, file_path: str, modality: str, content: str,
                 features: Optional[List[float]] = None,
                 entities: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        self.modality = modality
        self.content = content
        self.features = features if features is not None else []
        self.entities = entities if entities is not None else []
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"StandardizedOutput(file='{os.path.basename(self.file_path)}', modality='{self.modality}', content='{self.content[:50]}...')"

class GroundedEvidence:
    """Represents a piece of evidence grounded to a user query."""
    def __init__(self, snippet: str, source_file: str, modality: str,
                 confidence: float, relevance: float,
                 context: Optional[Dict[str, Any]] = None):
        self.snippet = snippet
        self.source_file = source_file
        self.modality = modality
        self.confidence = confidence
        self.relevance = relevance
        self.context = context if context is not None else {}

    def __repr__(self):
        return f"GroundedEvidence(snippet='{self.snippet[:50]}...', file='{os.path.basename(self.source_file)}', relevance={self.relevance:.2f})"

class SynthesisResult:
    """Represents the output of the iterative synthesis engine."""
    def __init__(self, answer: str, supporting_evidence: List[GroundedEvidence],
                 reasoning_steps: Optional[List[str]] = None):
        self.answer = answer
        self.supporting_evidence = supporting_evidence
        self.reasoning_steps = reasoning_steps if reasoning_steps is not None else []

    def __repr__(self):
        return f"SynthesisResult(answer='{self.answer[:100]}...', num_evidence={len(self.supporting_evidence)})"

# Mock Ingestion Module (ingestion/file_processor.py, ingestion/document_loader.py)
class MockFileProcessor:
    """Simulates scanning a directory for new or modified files."""
    def scan_for_new_files(self, directory: str) -> List[str]:
        logging.debug(f"Scanning directory for new files: {directory}")
        if not os.path.exists(directory):
            logging.warning(f"Ingestion directory not found: {directory}")
            return []
        
        # In a real system, this would compare against a last_scanned_timestamp
        # or file hashes. For a prototype, just list files.
        found_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                found_files.append(os.path.join(root, file))
        return found_files

class MockDocumentLoader:
    """Simulates loading raw file content."""
    def load_content(self, file_path: str) -> str:
        logging.debug(f"Loading raw content for: {file_path}")
        try:
            # For simplicity, we'll just read text files or return a placeholder
            if file_path.endswith(('.txt', '.md', '.log')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            return f"Placeholder content for {os.path.basename(file_path)}"
        except Exception as e:
            logging.error(f"Failed to load content for {file_path}: {e}")
            return ""

# Mock Perception Agents (perception/*.py)
class MockBaseAgent:
    """Base interface for perception agents."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        raise NotImplementedError("Subclasses must implement process method.")

class MockOCRAgent(MockBaseAgent):
    """Mocks OCR processing for PDF-like files."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        if file_path.lower().endswith(('.pdf', '.tiff')):
            logging.debug(f"Performing OCR on: {file_path}")
            content = f"OCR text from {os.path.basename(file_path)}. It mentions 'John Doe' and 'Project Alpha'."
            entities = [{"type": "PERSON", "name": "John Doe"}, {"type": "PROJECT", "name": "Project Alpha"}]
            return StandardizedOutput(file_path, "text", content, entities=entities)
        return None

class MockImageCaptioningAgent(MockBaseAgent):
    """Mocks image captioning."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        if file_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            logging.debug(f"Captioning image: {file_path}")
            content = f"A photo depicting a person smiling at a 'PMEG' presentation slide from {os.path.basename(file_path)}."
            entities = [{"type": "PERSON", "name": "person"}, {"type": "EVENT", "name": "presentation"}, {"type": "PROJECT", "name": "PMEG"}]
            return StandardizedOutput(file_path, "image", content, entities=entities)
        return None

class MockAudioTranscriptionAgent(MockBaseAgent):
    """Mocks audio transcription."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        if file_path.lower().endswith(('.mp3', '.wav')):
            logging.debug(f"Transcribing audio: {file_path}")
            content = f"Audio transcript from {os.path.basename(file_path)}: '...discussing the core components of the PMEG framework. Jane Smith raised a good point...'"
            entities = [{"type": "PROJECT", "name": "PMEG framework"}, {"type": "PERSON", "name": "Jane Smith"}]
            return StandardizedOutput(file_path, "audio", content, entities=entities)
        return None

class MockVideoAnalysisAgent(MockBaseAgent):
    """Mocks video content analysis."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        if file_path.lower().endswith(('.mp4', '.mov')):
            logging.debug(f"Analyzing video: {file_path}")
            content = f"Video analysis summary from {os.path.basename(file_path)}: detected 'meeting' context, 'John Doe' visible, discussing 'quarterly report'."
            entities = [{"type": "EVENT", "name": "meeting"}, {"type": "PERSON", "name": "John Doe"}, {"type": "DOCUMENT", "name": "quarterly report"}]
            return StandardizedOutput(file_path, "video", content, entities=entities)
        return None

class MockDocumentParserAgent(MockBaseAgent):
    """Mocks parsing of common document types."""
    def process(self, file_path: str) -> Optional[StandardizedOutput]:
        if file_path.lower().endswith(('.doc', '.docx', '.txt', '.csv', '.json')):
            logging.debug(f"Parsing document: {file_path}")
            loader = MockDocumentLoader()
            raw_content = loader.load_content(file_path)
            # Simple entity extraction from content for mock
            entities = []
            if "John Doe" in raw_content: entities.append({"type": "PERSON", "name": "John Doe"})
            if "PMEG" in raw_content: entities.append({"type": "PROJECT", "name": "PMEG"})
            if "meeting" in raw_content: entities.append({"type": "EVENT", "name": "meeting"})

            return StandardizedOutput(file_path, "document", raw_content, entities=entities)
        return None

# Mock Embedding & Alignment (embeddings/*.py)
class MockMultimodalModel:
    """Simulates generating cross-modal embeddings."""
    def generate_embedding(self, data: StandardizedOutput) -> List[float]:
        logging.debug(f"Generating embedding for {data.file_path} ({data.modality})")
        # Simple hash-based embedding for prototype
        content_hash = hash(data.content) % 1000
        # Return a fixed-size vector based on the hash
        return [float(content_hash / 1000.0)] * 128 # 128-dimensional mock embedding

class MockVectorStore:
    """In-memory mock for a vector database."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {} # {file_path: {"embedding": [], "metadata": {}}}
        logging.info("MockVectorStore initialized.")

    def upsert(self, file_path: str, embedding: List[float], metadata: Dict[str, Any]):
        self._store[file_path] = {"embedding": embedding, "metadata": metadata}
        logging.debug(f"Upserted embedding for {file_path}")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        logging.debug(f"Querying vector store with embedding. Top {top_k} results.")
        results = []
        # In a real scenario, this would involve cosine similarity or other metric
        # For mock, we'll return items with some "similarity" based on overlap in content hints
        for fp, data in self._store.items():
            # Simulate a score; for prototype, higher score if query_embedding has higher average value
            # and if the file path implies a match
            score = 0.5 + (sum(query_embedding) / 1280.0) # Base score + influence from query_embedding
            if "PMEG" in fp.lower() or "john doe" in fp.lower(): # Simple keyword match
                score += 0.2
            results.append({"file_path": fp, "score": min(score, 1.0), **data["metadata"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Mock Semantic Entity Graph (knowledge_graph/*.py)
class MockEntityExtractor:
    """Extracts entities from standardized output."""
    def extract_entities(self, data: StandardizedOutput) -> List[Dict[str, Any]]:
        logging.debug(f"Extracting entities from {data.file_path}")
        # For the prototype, just return the entities identified by perception agents.
        # A real extractor might use NER models.
        return data.entities + [{"type": "FILE_SOURCE", "name": os.path.basename(data.file_path), "path": data.file_path, "modality": data.modality}]

class MockDisambiguator:
    """Resolves entity ambiguities."""
    def disambiguate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.debug(f"Disambiguating {len(entities)} entities.")
        # For prototype, assign a simple unique ID
        disambiguated_entities = []
        seen_entities = {} # { (type, name): unified_id }
        for entity in entities:
            key = (entity['type'], entity['name'])
            if key not in seen_entities:
                unified_id = f"{entity['type'].upper()}_{entity['name'].replace(' ', '_').replace('.', '_').upper()}"
                seen_entities[key] = unified_id
            else:
                unified_id = seen_entities[key]
            
            disambiguated_entity = entity.copy()
            disambiguated_entity['unified_id'] = unified_id
            disambiguated_entities.append(disambiguated_entity)
        return disambiguated_entities

class MockGraphBuilder:
    """Builds and updates an in-memory knowledge graph."""
    def __init__(self):
        self._graph: Dict[str, Dict[str, Any]] = {} # {unified_id: {properties: {}, relations: []}}
        logging.info("MockGraphBuilder initialized.")

    def update_graph(self, entities: List[Dict[str, Any]], provenance: Dict[str, Any]):
        logging.debug(f"Updating graph with {len(entities)} entities from {provenance.get('file_path')}")
        file_node_id = f"FILE_{os.path.basename(provenance['file_path']).replace('.', '_').upper()}"
        
        # Ensure file node exists
        if file_node_id not in self._graph:
            self._graph[file_node_id] = {
                "properties": {"type": "FILE", "name": os.path.basename(provenance['file_path']), "path": provenance['file_path'], "modality": provenance['modality']},
                "relations": []
            }

        for entity in entities:
            uid = entity.get('unified_id')
            if not uid:
                logging.warning(f"Entity without unified_id encountered: {entity}. Skipping graph update for this entity.")
                continue

            if uid not in self._graph:
                self._graph[uid] = {"properties": {"type": entity['type'], "name": entity['name']}, "relations": []}
            
            # Add relationship between entity and file
            relation_to_file = {"type": "MENTIONED_IN", "target_id": file_node_id, "file_path": provenance['file_path'], "modality": provenance['modality']}
            if relation_to_file not in self._graph[uid]["relations"]: # Avoid duplicates
                self._graph[uid]["relations"].append(relation_to_file)
            
            # Add relationship from file to entity
            relation_from_file = {"type": "MENTIONS", "target_id": uid, "entity_type": entity['type'], "entity_name": entity['name']}
            if relation_from_file not in self._graph[file_node_id]["relations"]:
                self._graph[file_node_id]["relations"].append(relation_from_file)

        logging.debug(f"Graph now has {len(self._graph)} nodes.")

class MockGraphDBInterface:
    """Abstraction for interacting with a graph database."""
    def __init__(self, graph_builder: MockGraphBuilder):
        self._graph_builder = graph_builder # Access the in-memory graph directly for prototype
        logging.info("MockGraphDBInterface initialized.")

    def query_entities(self, entity_type: Optional[str] = None, entity_name: Optional[str] = None) -> List[Dict[str, Any]]:
        logging.debug(f"Querying graph for entities: type={entity_type}, name='{entity_name}'")
        results = []
        for uid, data in self._graph_builder._graph.items():
            props = data['properties']
            match = True
            if entity_type and props.get('type') != entity_type.upper():
                match = False
            if entity_name and entity_name.lower() not in props.get('name', '').lower():
                match = False
            
            if match:
                results.append({"id": uid, **props, "relations": data['relations']})
        return results

# Mock Database for Metadata (database/crud.py, database/schema.py)
class MockCRUDBase:
    """Generic in-memory CRUD operations."""
    def __init__(self):
        self._db: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        new_id = self._next_id
        self._next_id += 1
        data['id'] = new_id
        self._db[new_id] = data.copy()
        logging.debug(f"Created record: {data}")
        return self._db[new_id]

    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        return self._db.get(item_id)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._db.values())

    def update(self, item_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if item_id in self._db:
            self._db[item_id].update(data)
            logging.debug(f"Updated record {item_id}: {self._db[item_id]}")
            return self._db[item_id]
        return None

    def delete(self, item_id: int) -> bool:
        if item_id in self._db:
            del self._db[item_id]
            logging.debug(f"Deleted record {item_id}")
            return True
        return False

class MockFileMetadataCRUD(MockCRUDBase):
    """CRUD operations specifically for file metadata."""
    def get_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        for record in self._db.values():
            if record.get('file_path') == file_path:
                return record
        return None

# Mock Contextual Grounding Module (grounding/*.py)
class MockRetriever:
    """Retrieves relevant evidence from vector store and knowledge graph."""
    def __init__(self, vector_store: MockVectorStore, graph_db: MockGraphDBInterface, file_metadata_crud: MockFileMetadataCRUD):
        self._vector_store = vector_store
        self._graph_db = graph_db
        self._file_metadata_crud = file_metadata_crud
        logging.info("MockRetriever initialized.")

    def retrieve_evidence(self, query: str, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        logging.debug(f"Retrieving evidence for query: '{query}'")
        
        # 1. Retrieve from Vector Store (semantic similarity)
        query_embedding = [hash(query) % 1000 / 1000.0] * 128 # Mock query embedding
        vector_results = self._vector_store.query(query_embedding, top_k=5)
        
        retrieved_items = []
        for res in vector_results:
            file_meta = self._file_metadata_crud.get_by_path(res['file_path'])
            if file_meta:
                retrieved_items.append({
                    "content_snippet": res.get("content_snippet", f"Content from {os.path.basename(res['file_path'])}"),
                    "file_path": res['file_path'],
                    "modality": file_meta.get('modality', 'unknown'),
                    "score": res['score'],
                    "source_type": "vector_store"
                })

        # 2. Retrieve from Knowledge Graph (structured relationships)
        # Extract potential entities from query for graph lookup
        query_entities = []
        if "john doe" in query.lower(): query_entities.append({"type": "PERSON", "name": "John Doe"})
        if "pmeg project" in query.lower(): query_entities.append({"type": "PROJECT", "name": "PMEG"})
        if "jane smith" in query.lower(): query_entities.append({"type": "PERSON", "name": "Jane Smith"})

        for q_entity in query_entities:
            graph_entities = self._graph_db.query_entities(entity_type=q_entity['type'], entity_name=q_entity['name'])
            for entity in graph_entities:
                for relation in entity.get('relations', []):
                    if relation['type'] == "MENTIONED_IN" and relation.get('file_path'):
                        file_meta = self._file_metadata_crud.get_by_path(relation['file_path'])
                        if file_meta:
                            retrieved_items.append({
                                "content_snippet": f"Graph shows '{entity['name']}' ({entity['type']}) is mentioned in '{os.path.basename(relation['file_path'])}'.",
                                "file_path": relation['file_path'],
                                "modality": file_meta.get('modality', relation.get('modality', 'unknown')),
                                "score": 0.8, # Assume graph hits are highly relevant
                                "source_type": "knowledge_graph"
                            })
        return retrieved_items

class MockRanker:
    """Ranks retrieved evidence based on relevance and confidence."""
    def rank_evidence(self, query: str, evidence: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[GroundedEvidence]:
        logging.debug(f"Ranking {len(evidence)} pieces of evidence for query: '{query}'")
        ranked_results = []
        for item in evidence:
            # Simple ranking logic for prototype: use score from retriever, static confidence
            relevance = item.get('score', 0.5)
            confidence = 0.9 # Assume high confidence for retrieved items in prototype
            
            # Boost relevance if query terms are directly in snippet (for text modalities)
            if "text" in item['modality'] or "document" in item['modality']:
                if all(term.lower() in item['content_snippet'].lower() for term in query.split()):
                    relevance += 0.1

            ranked_results.append(
                GroundedEvidence(
                    snippet=item['content_snippet'],
                    source_file=item['file_path'],
                    modality=item['modality'],
                    confidence=confidence,
                    relevance=min(relevance, 1.0),
                    context={"source_type": item["source_type"]}
                )
            )
        ranked_results.sort(key=lambda x: x.relevance, reverse=True)
        return ranked_results

class MockGroundingAPI:
    """Exposes the contextual grounding functionality."""
    def __init__(self, retriever: MockRetriever, ranker: MockRanker):
        self._retriever = retriever
        self._ranker = ranker
        logging.info("MockGroundingAPI initialized.")

    def ground_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> List[GroundedEvidence]:
        if user_context is None:
            user_context = {}
        logging.info(f"Attempting to ground query: '{query}'")
        try:
            raw_evidence = self._retriever.retrieve_evidence(query, user_context)
            grounded_evidence = self._ranker.rank_evidence(query, raw_evidence, user_context)
            return grounded_evidence
        except Exception as e:
            logging.error(f"Error during grounding query '{query}': {e}", exc_info=True)
            return []

# Mock Iterative Synthesis Engine (synthesis/*.py)
class MockSynthesisEngine:
    """Orchestrates complex reasoning tasks by iteratively using the grounding module."""
    def __init__(self, grounding_api: MockGroundingAPI):
        self._grounding_api = grounding_api
        logging.info("MockSynthesisEngine initialized.")

    def synthesize_response(self, task: str, user_context: Optional[Dict[str, Any]] = None, strategy: str = "summarize") -> Optional[SynthesisResult]:
        if user_context is None:
            user_context = {}
        logging.info(f"Synthesizing response for task: '{task}' using strategy: '{strategy}'")

        try:
            # Step 1: Initial grounding based on the task
            initial_evidence = self._grounding_api.ground_query(task, user_context)
            reasoning_steps = [f"Initial evidence grounded for task '{task}'. Found {len(initial_evidence)} pieces."]
            
            if not initial_evidence:
                return SynthesisResult(f"Could not find sufficient evidence for task: {task}.", [], reasoning_steps)

            # Step 2: Simulate iterative refinement/combination
            # For a prototype, we'll combine the top N pieces of evidence.
            # A real engine would loop, re-query, or use an LLM for reasoning.
            top_evidence_for_synthesis = sorted(initial_evidence, key=lambda x: x.relevance, reverse=True)[:5] # Take top 5
            
            combined_text_snippets = []
            for ev in top_evidence_for_synthesis:
                combined_text_snippets.append(f"From {os.path.basename(ev.source_file)} ({ev.modality}): \"{ev.snippet}\"")
            
            combined_summary_content = "\n".join(combined_text_snippets)

            final_answer = ""
            if strategy == "summarize":
                final_answer = f"Summary for '{task}': {combined_summary_content[:500]}..."
                reasoning_steps.append("Summarization strategy applied to combine top evidence.")
            elif strategy == "detailed_report":
                final_answer = f"Detailed Report for '{task}':\n\n{combined_summary_content}"
                reasoning_steps.append("Detailed report strategy applied to combine top evidence.")
            else: # Default or other strategies
                final_answer = f"Synthesized answer for '{task}': {combined_summary_content[:500]}..."
                reasoning_steps.append(f"Generic synthesis strategy '{strategy}' applied.")

            return SynthesisResult(final_answer, top_evidence_for_synthesis, reasoning_steps)

        except Exception as e:
            logging.error(f"Error during synthesis for task '{task}': {e}", exc_info=True)
            return None

# --- Main PMEG Framework Orchestrator ---
class PMEG_Framework:
    """
    The Personalized Multimodal Evidence Grounding (PMEG) Framework orchestrator.
    Initializes and coordinates all modular components.
    """
    def __init__(self, config: Config):
        self.config = config
        logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("PMEG_Framework")
        self.logger.info("Initializing PMEG Framework...")

        # Initialize Data Storage & Management
        self.file_metadata_crud = MockFileMetadataCRUD()
        self.vector_store = MockVectorStore()
        self.graph_builder = MockGraphBuilder()
        self.graph_db_interface = MockGraphDBInterface(self.graph_builder)

        # Initialize Ingestion Module
        self.file_processor = MockFileProcessor()
        self.document_loader = MockDocumentLoader()

        # Initialize Perception Agents
        self.perception_agents: List[MockBaseAgent] = [
            MockOCRAgent(),
            MockImageCaptioningAgent(),
            MockAudioTranscriptionAgent(),
            MockVideoAnalysisAgent(),
            MockDocumentParserAgent()
        ]

        # Initialize Embedding Module
        self.multimodal_model = MockMultimodalModel()
        # MockAlignmentTrainer(self.multimodal_model) # Trainer would be used in a separate training phase

        # Initialize Knowledge Graph Module
        self.entity_extractor = MockEntityExtractor()
        self.disambiguator = MockDisambiguator()

        # Initialize Contextual Grounding Module
        self.retriever = MockRetriever(self.vector_store, self.graph_db_interface, self.file_metadata_crud)
        self.ranker = MockRanker()
        self.grounding_api = MockGroundingAPI(self.retriever, self.ranker)

        # Initialize Iterative Synthesis Engine
        self.synthesis_engine = MockSynthesisEngine(self.grounding_api)

        self.logger.info("PMEG Framework initialized successfully.")

    def _process_single_file(self, file_path: str):
        """Processes a single file through the PMEG pipeline."""
        self.logger.info(f"Starting processing for file: {file_path}")
        try:
            # 1. Load content
            raw_content = self.document_loader.load_content(file_path)
            if not raw_content:
                self.logger.warning(f"Could not load content for {file_path}. Skipping perception.")
                return

            # 2. Perception: Find a suitable agent and process
            perceived_output: Optional[StandardizedOutput] = None
            for agent in self.perception_agents:
                output = agent.process(file_path)
                if output:
                    perceived_output = output
                    break

            if not perceived_output:
                self.logger.warning(f"No suitable perception agent found for {file_path}. Cannot proceed with embedding/graph for this file.")
                # Still store basic metadata if we loaded content
                file_meta = self.file_metadata_crud.get_by_path(file_path)
                if not file_meta:
                    self.file_metadata_crud.create({"file_path": file_path, "modality": "unknown", "status": "processed_no_perception"})
                else:
                    self.file_metadata_crud.update(file_meta['id'], {"modality": "unknown", "status": "processed_no_perception"})
                return

            # Store or update file metadata after successful perception
            file_meta_data = {
                "file_path": file_path,
                "modality": perceived_output.modality,
                "status": "perceived_and_processed"
            }
            existing_meta = self.file_metadata_crud.get_by_path(file_path)
            if not existing_meta:
                self.file_metadata_crud.create(file_meta_data)
            else:
                self.file_metadata_crud.update(existing_meta['id'], file_meta_data)


            # 3. Cross-Modal Embedding & Alignment
            embedding = self.multimodal_model.generate_embedding(perceived_output)
            self.vector_store.upsert(
                file_path,
                embedding,
                metadata={"modality": perceived_output.modality, "content_snippet": perceived_output.content[:200]} # Store a snippet for retrieval context
            )

            # 4. Semantic Entity Graph
            extracted_entities = self.entity_extractor.extract_entities(perceived_output)
            if extracted_entities:
                disambiguated_entities = self.disambiguator.disambiguate_entities(extracted_entities)
                self.graph_builder.update_graph(
                    disambiguated_entities,
                    provenance={"file_path": file_path, "modality": perceived_output.modality}
                )
            else:
                self.logger.debug(f"No entities extracted from {file_path}. Skipping graph update for entities.")

            self.logger.info(f"Successfully processed file: {file_path}")
        except Exception as e:
            self.logger.error(f"Critical error processing file {file_path}: {e}", exc_info=True)

    def ingest_and_process_files(self, directory: str):
        """Initiates file system scanning and processes found files."""
        self.logger.info(f"Initiating ingestion and processing for directory: '{directory}'")
        try:
            files_to_process = self.file_processor.scan_for_new_files(directory)
            if not files_to_process:
                self.logger.info(f"No new files found in '{directory}' to process.")
                return

            self.logger.info(f"Found {len(files_to_process)} files to process.")
            for file_path in files_to_process:
                self._process_single_file(file_path)
            self.logger.info("Finished ingestion and processing run.")
        except Exception as e:
            self.logger.critical(f"Error during ingestion and processing for directory '{directory}': {e}", exc_info=True)

    def query_grounding(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> List[GroundedEvidence]:
        """
        Public API for the Contextual Grounding Module.
        Takes a user query and returns grounded evidence.
        """
        self.logger.info(f"Received grounding query: '{query}'")
        return self.grounding_api.ground_query(query, user_context)

    def perform_synthesis(self, task: str, user_context: Optional[Dict[str, Any]] = None, strategy: str = "summarize") -> Optional[SynthesisResult]:
        """
        Public API for the Iterative Synthesis Engine.
        Takes a complex task and returns a synthesized response.
        """
        self.logger.info(f"Received synthesis task: '{task}' with strategy: '{strategy}'")
        return self.synthesis_engine.synthesize_response(task, user_context, strategy)

# --- Example Usage ---
if __name__ == "__main__":
    app_config = Config()
    app_config.LOG_LEVEL = "INFO" # Set to DEBUG for more verbose output

    # Ensure the processing directory exists for mocks
    test_directory = app_config.PROCESSING_DIR
    os.makedirs(test_directory, exist_ok=True)

    # Create some dummy files to simulate a personal file system
    dummy_files_content = {
        "report_q1_2023.pdf": "This PDF discusses Q1 2023 performance. John Doe led the marketing efforts for Project Alpha.",
        "team_photo.jpg": "A picture of the team during a PMEG project meeting.",
        "meeting_notes_audio.mp3": "Audio recording transcript: '...Jane Smith mentioned key requirements for the PMEG framework...'",
        "video_presentation.mp4": "Video of a presentation where John Doe explains project progress and metrics.",
        "readme.txt": "Important notes for PMEG Framework development. Contact Jane Smith for issues.",
        "project_plan.docx": "Detailed plan for Project Alpha, outlining tasks and milestones. (Mock content, actual parsing would extract more)",
        "marketing_brief.txt": "Brief for marketing team. John Doe to review. Focus on PMEG features.",
        "unknown_file.bin": "Binary content that no agent can process easily."
    }

    print(f"Creating dummy files in {test_directory}...")
    for filename, content in dummy_files_content.items():
        filepath = os.path.join(test_directory, filename)
        with open(filepath, "w", encoding='utf-8') as f: # Use 'w' for text, content doesn't matter for binary mock
            f.write(content)
    print("Dummy files created.")

    # Initialize the PMEG Framework
    pmeg_framework = PMEG_Framework(app_config)

    # --- Step 1: Simulate Ingestion and Processing of Files ---
    print("\n--- Starting Ingestion and Processing ---")
    pmeg_framework.ingest_and_process_files(test_directory)
    print("--- Ingestion and Processing Complete ---")

    # --- Step 2: Simulate User Query for Contextual Grounding ---
    user_query_1 = "What documents or information are related to John Doe and the PMEG project?"
    user_context_1 = {"user_id": "user123", "current_project": "PMEG"}
    print(f"\n--- Grounding Query: '{user_query_1}' ---")
    grounded_evidence_1 = pmeg_framework.query_grounding(user_query_1, user_context_1)

    if grounded_evidence_1:
        print(f"Found {len(grounded_evidence_1)} pieces of grounded evidence:")
        for i, evidence in enumerate(grounded_evidence_1):
            print(f"  {i+1}. Source: {os.path.basename(evidence.source_file)} ({evidence.modality})")
            print(f"     Snippet: {evidence.snippet[:120]}...")
            print(f"     Relevance: {evidence.relevance:.2f}, Confidence: {evidence.confidence:.2f}")
    else:
        print("No evidence found for the query.")

    user_query_2 = "Tell me about Jane Smith's involvement."
    print(f"\n--- Grounding Query: '{user_query_2}' ---")
    grounded_evidence_2 = pmeg_framework.query_grounding(user_query_2)
    if grounded_evidence_2:
        print(f"Found {len(grounded_evidence_2)} pieces of grounded evidence:")
        for i, evidence in enumerate(grounded_evidence_2):
            print(f"  {i+1}. Source: {os.path.basename(evidence.source_file)} ({evidence.modality})")
            print(f"     Snippet: {evidence.snippet[:120]}...")
            print(f"     Relevance: {evidence.relevance:.2f}, Confidence: {evidence.confidence:.2f}")
    else:
        print("No evidence found for the query.")

    # --- Step 3: Simulate User Task for Iterative Synthesis ---
    user_task_1 = "Provide a comprehensive summary of the PMEG framework, including its key participants and where it's mentioned across my files."
    print(f"\n--- Synthesis Task: '{user_task_1}' ---")
    synthesis_result_1 = pmeg_framework.perform_synthesis(user_task_1, user_context_1, strategy="detailed_report")

    if synthesis_result_1:
        print("\n--- Synthesis Result ---")
        print(f"Answer:\n{synthesis_result_1.answer}")
        print("\nReasoning Steps:")
        for step in synthesis_result_1.reasoning_steps:
            print(f"- {step}")
        print("\nTop 3 Supporting Evidence Snippets:")
        for i, evidence in enumerate(synthesis_result_1.supporting_evidence[:3]):
            print(f"  - From {os.path.basename(evidence.source_file)} ({evidence.modality}): {evidence.snippet[:100]}...")
    else:
        print("Synthesis failed or returned no result.")

    # --- Cleanup ---
    print(f"\nCleaning up dummy files in {test_directory}...")
    import shutil
    try:
        shutil.rmtree(test_directory)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Error during cleanup: {e}")