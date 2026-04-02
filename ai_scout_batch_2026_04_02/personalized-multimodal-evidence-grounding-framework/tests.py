```python
import pytest
import os
import shutil
import logging
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Optional

# Temporarily suppress INFO logs from main.py for cleaner test output
logging.getLogger("PMEG_Framework").setLevel(logging.WARNING)
logging.getLogger("MockVectorStore").setLevel(logging.WARNING)
logging.getLogger("MockGraphBuilder").setLevel(logging.WARNING)
logging.getLogger("MockGroundingAPI").setLevel(logging.WARNING)
logging.getLogger("MockSynthesisEngine").setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.DEBUG) # For test file's own logs

# Import the code to be tested
from main import (
    Config, StandardizedOutput, GroundedEvidence, SynthesisResult,
    MockFileProcessor, MockDocumentLoader,
    MockOCRAgent, MockImageCaptioningAgent, MockAudioTranscriptionAgent,
    MockVideoAnalysisAgent, MockDocumentParserAgent,
    MockMultimodalModel, MockVectorStore,
    MockEntityExtractor, MockDisambiguator, MockGraphBuilder, MockGraphDBInterface,
    MockCRUDBase, MockFileMetadataCRUD,
    MockRetriever, MockRanker, MockGroundingAPI,
    MockSynthesisEngine, PMEG_Framework
)

# --- Fixtures for common test setups ---

@pytest.fixture(scope="function")
def temp_processing_dir():
    """Provides a temporary directory for file operations, cleans up after."""
    test_dir = "temp_pmeg_test_dir"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture(scope="function")
def mock_config():
    """Provides a mock Config object with test-specific paths."""
    config = Config()
    config.PROCESSING_DIR = "temp_pmeg_test_dir" # Will be managed by temp_processing_dir fixture
    config.LOG_LEVEL = "DEBUG"
    return config

@pytest.fixture(scope="function")
def pmeg_framework_instance(mock_config, temp_processing_dir):
    """Provides an initialized PMEG_Framework instance for testing."""
    # Ensure config points to the temporary dir
    mock_config.PROCESSING_DIR = temp_processing_dir
    framework = PMEG_Framework(mock_config)
    yield framework
    # Clean up any potential state in mock components if necessary
    framework.vector_store.clear()
    framework.file_metadata_crud._db.clear()
    framework.graph_builder._graph.clear()

@pytest.fixture(scope="function")
def mock_document_loader():
    """Provides a fresh MockDocumentLoader instance."""
    return MockDocumentLoader()

@pytest.fixture(scope="function")
def mock_file_processor():
    """Provides a fresh MockFileProcessor instance."""
    return MockFileProcessor()

# --- Unit Tests: Data Models ---

def test_standardized_output_init():
    output = StandardizedOutput("path/to/file.txt", "text", "Some content.")
    assert output.file_path == "path/to/file.txt"
    assert output.modality == "text"
    assert output.content == "Some content."
    assert output.features == []
    assert output.entities == []
    assert output.metadata == {}

    output_full = StandardizedOutput(
        "path/to/img.jpg", "image", "Image caption.",
        features=[0.1, 0.2],
        entities=[{"name": "person", "type": "PERSON"}],
        metadata={"size": "1MB"}
    )
    assert output_full.features == [0.1, 0.2]
    assert output_full.entities[0]["name"] == "person"
    assert output_full.metadata["size"] == "1MB"

def test_standardized_output_repr():
    output = StandardizedOutput("/path/to/document.pdf", "document", "This is a long piece of text that should be truncated for the representation string if it's too long.")
    assert "document.pdf" in repr(output)
    assert "document" in repr(output)
    assert "This is a long piece of text that should be trun..." in repr(output)

def test_grounded_evidence_init():
    evidence = GroundedEvidence("Snippet text", "source.pdf", "text", 0.9, 0.8)
    assert evidence.snippet == "Snippet text"
    assert evidence.source_file == "source.pdf"
    assert evidence.modality == "text"
    assert evidence.confidence == 0.9
    assert evidence.relevance == 0.8
    assert evidence.context == {}

def test_grounded_evidence_repr():
    evidence = GroundedEvidence("This is a very long snippet that needs to be truncated for printing.", "/data/doc.txt", "text", 0.7, 0.6)
    assert "doc.txt" in repr(evidence)
    assert "relevance=0.60" in repr(evidence)
    assert "This is a very long snippet that needs to be trun..." in repr(evidence)

def test_synthesis_result_init():
    evidence1 = GroundedEvidence("snip1", "file1", "text", 0.9, 0.8)
    evidence2 = GroundedEvidence("snip2", "file2", "image", 0.7, 0.6)
    result = SynthesisResult("Final Answer", [evidence1, evidence2], ["step1", "step2"])
    assert result.answer == "Final Answer"
    assert len(result.supporting_evidence) == 2
    assert result.reasoning_steps == ["step1", "step2"]

def test_synthesis_result_repr():
    evidence1 = GroundedEvidence("snip1", "file1", "text", 0.9, 0.8)
    result = SynthesisResult("This is a synthesized answer that might be quite long and should be truncated.", [evidence1])
    assert "num_evidence=1" in repr(result)
    assert "This is a synthesized answer that might be quite long and should be truncat..." in repr(result)


# --- Unit Tests: Ingestion Module ---

def test_mock_file_processor_scan_empty_directory(temp_processing_dir, mock_file_processor):
    """Test scanning an empty directory."""
    files = mock_file_processor.scan_for_new_files(temp_processing_dir)
    assert files == []

def test_mock_file_processor_scan_with_files(temp_processing_dir, mock_file_processor):
    """Test scanning a directory with files."""
    file1 = os.path.join(temp_processing_dir, "test1.txt")
    file2 = os.path.join(temp_processing_dir, "subdir", "test2.pdf")
    os.makedirs(os.path.dirname(file2), exist_ok=True)
    with open(file1, "w") as f: f.write("content")
    with open(file2, "w") as f: f.write("content")

    files = mock_file_processor.scan_for_new_files(temp_processing_dir)
    assert len(files) == 2
    assert file1 in files
    assert file2 in files

def test_mock_file_processor_scan_non_existent_directory(mock_file_processor, caplog):
    """Test scanning a non-existent directory."""
    with caplog.at_level(logging.WARNING):
        files = mock_file_processor.scan_for_new_files("/non/existent/path")
        assert files == []
        assert "Ingestion directory not found" in caplog.text

def test_mock_document_loader_load_text_file(temp_processing_dir, mock_document_loader):
    """Test loading content from a text file."""
    file_path = os.path.join(temp_processing_dir, "sample.txt")
    expected_content = "Hello, this is a test text file."
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(expected_content)

    content = mock_document_loader.load_content(file_path)
    assert content == expected_content

def test_mock_document_loader_load_binary_file(temp_processing_dir, mock_document_loader):
    """Test loading content from a binary file (mock returns placeholder)."""
    file_path = os.path.join(temp_processing_dir, "image.jpg")
    with open(file_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n") # Dummy binary content

    content = mock_document_loader.load_content(file_path)
    assert f"Placeholder content for image.jpg" == content

def test_mock_document_loader_load_non_existent_file(mock_document_loader, caplog):
    """Test loading content from a non-existent file."""
    with caplog.at_level(logging.ERROR):
        content = mock_document_loader.load_content("/non/existent/file.txt")
        assert content == ""
        assert "Failed to load content" in caplog.text


# --- Unit Tests: Perception Agents ---

@pytest.fixture
def mock_file(temp_processing_dir, filename, content):
    """Creates a dummy file for agent processing."""
    file_path = os.path.join(temp_processing_dir, filename)
    mode = 'w' if isinstance(content, str) else 'wb'
    with open(file_path, mode, encoding='utf-8' if isinstance(content, str) else None) as f:
        f.write(content)
    return file_path

@pytest.mark.parametrize("filename, content, agent_class, expected_modality", [
    ("report.pdf", "dummy pdf content", MockOCRAgent, "text"),
    ("invoice.tiff", "dummy tiff content", MockOCRAgent, "text"),
    ("photo.jpg", "dummy jpg content", MockImageCaptioningAgent, "image"),
    ("logo.png", "dummy png content", MockImageCaptioningAgent, "image"),
    ("meeting.mp3", "dummy mp3 content", MockAudioTranscriptionAgent, "audio"),
    ("podcast.wav", "dummy wav content", MockAudioTranscriptionAgent, "audio"),
    ("demo.mp4", "dummy mp4 content", MockVideoAnalysisAgent, "video"),
    ("clip.mov", "dummy mov content", MockVideoAnalysisAgent, "video"),
    ("notes.txt", "PMEG project notes for John Doe.", MockDocumentParserAgent, "document"),
    ("plan.docx", "Detailed plan for Project Alpha. Meeting at 10 AM.", MockDocumentParserAgent, "document"),
])
def test_perception_agent_processes_correct_type(mock_file, filename, content, agent_class, expected_modality):
    agent = agent_class()
    output = agent.process(mock_file(filename, content))
    assert output is not None
    assert isinstance(output, StandardizedOutput)
    assert output.file_path == mock_file(filename, content)
    assert output.modality == expected_modality
    assert len(output.content) > 0
    assert len(output.entities) > 0 # Most mocks generate some entities

@pytest.mark.parametrize("filename, content, agent_class", [
    ("photo.jpg", "dummy jpg content", MockOCRAgent),          # OCR should ignore JPG unless specified as image
    ("report.pdf", "dummy pdf content", MockImageCaptioningAgent), # Captioning should ignore PDF
    ("video.mp4", "dummy mp4 content", MockAudioTranscriptionAgent), # Audio trans. should ignore video
    ("audio.mp3", "dummy mp3 content", MockVideoAnalysisAgent),  # Video analysis should ignore audio
    ("photo.jpg", "dummy jpg content", MockDocumentParserAgent), # Doc parser should ignore image
])
def test_perception_agent_ignores_incorrect_type(mock_file, filename, content, agent_class):
    agent = agent_class()
    output = agent.process(mock_file(filename, content))
    assert output is None

@pytest.mark.parametrize("agent_class", [
    MockOCRAgent, MockImageCaptioningAgent, MockAudioTranscriptionAgent,
    MockVideoAnalysisAgent, MockDocumentParserAgent
])
def test_perception_agent_non_existent_file(agent_class):
    agent = agent_class()
    output = agent.process("/non/existent/file.xyz")
    # Mocks return None directly if file doesn't match extension pattern.
    # If a mock tries to open it, it might raise, but our mocks check extension first.
    # So, checking for None is sufficient for this mock behavior.
    assert output is None


# --- Unit Tests: Embedding & Alignment ---

def test_mock_multimodal_model_generate_embedding():
    model = MockMultimodalModel()
    output = StandardizedOutput("path/file.txt", "text", "test content")
    embedding = model.generate_embedding(output)
    assert isinstance(embedding, list)
    assert len(embedding) == 128
    assert all(isinstance(x, float) for x in embedding)

def test_mock_multimodal_model_embedding_consistency():
    model = MockMultimodalModel()
    output1 = StandardizedOutput("path/file1.txt", "text", "unique content A")
    output2 = StandardizedOutput("path/file2.txt", "text", "unique content B")
    
    emb1 = model.generate_embedding(output1)
    emb2 = model.generate_embedding(output2)
    
    # Due to hash-based mock, different content should yield different embeddings
    assert emb1 != emb2

def test_mock_vector_store_upsert_and_query():
    store = MockVectorStore()
    embedding1 = [0.1] * 128
    embedding2 = [0.9] * 128 # Higher values to simulate "more similar" for mock query
    
    store.upsert("file1.txt", embedding1, {"modality": "text", "size": 100})
    store.upsert("file2.pdf", embedding2, {"modality": "document", "keywords": ["PMEG"]}) # Keyword for higher score
    
    # Query with something that looks like embedding2, and has PMEG keyword affinity
    query_embedding = [0.8] * 128 
    results = store.query(query_embedding, top_k=2)
    
    assert len(results) == 2
    assert results[0]['file_path'] == "file2.pdf" # Should be ranked higher due to keyword and higher mock score
    assert results[0]['modality'] == "document"
    assert results[0]['score'] > results[1]['score'] # Expect file2.pdf to have higher score

def test_mock_vector_store_query_empty():
    store = MockVectorStore()
    results = store.query([0.5]*128)
    assert results == []

def test_mock_vector_store_upsert_update():
    store = MockVectorStore()
    initial_embedding = [0.1] * 128
    updated_embedding = [0.2] * 128
    
    store.upsert("test_file.txt", initial_embedding, {"version": 1})
    assert store._store["test_file.txt"]["embedding"] == initial_embedding
    
    store.upsert("test_file.txt", updated_embedding, {"version": 2})
    assert store._store["test_file.txt"]["embedding"] == updated_embedding
    assert store._store["test_file.txt"]["metadata"]["version"] == 2


# --- Unit Tests: Semantic Entity Graph ---

def test_mock_entity_extractor():
    extractor = MockEntityExtractor()
    output = StandardizedOutput(
        "/path/to/report.pdf", "document", "Report content mentioning John Doe and Project Alpha.",
        entities=[{"type": "PERSON", "name": "John Doe"}, {"type": "PROJECT", "name": "Project Alpha"}]
    )
    entities = extractor.extract_entities(output)
    assert len(entities) == 3 # John Doe, Project Alpha, and FILE_SOURCE
    assert any(e['name'] == "John Doe" for e in entities)
    assert any(e['type'] == "FILE_SOURCE" and e['name'] == "report.pdf" for e in entities)

def test_mock_disambiguator():
    disambiguator = MockDisambiguator()
    entities_input = [
        {"type": "PERSON", "name": "John Doe"},
        {"type": "PERSON", "name": "John Doe"}, # Duplicate
        {"type": "PROJECT", "name": "Project Alpha"},
        {"type": "PERSON", "name": "Jane Smith"}
    ]
    disambiguated = disambiguator.disambiguate_entities(entities_input)
    assert len(disambiguated) == 4 # All inputs get an ID, but duplicates share the same unified_id
    
    john_doe_ids = [e['unified_id'] for e in disambiguated if e['name'] == "John Doe"]
    assert len(set(john_doe_ids)) == 1 # Both John Doe entries should get the same ID
    
    project_alpha_ids = [e['unified_id'] for e in disambiguated if e['name'] == "Project Alpha"]
    assert len(set(project_alpha_ids)) == 1
    assert john_doe_ids[0] != project_alpha_ids[0] # Different entities get different IDs

def test_mock_graph_builder_update_graph():
    graph_builder = MockGraphBuilder()
    
    entities = [
        {"type": "PERSON", "name": "John Doe", "unified_id": "PERSON_JOHN_DOE"},
        {"type": "PROJECT", "name": "PMEG", "unified_id": "PROJECT_PMEG"}
    ]
    provenance = {"file_path": "/path/to/doc.pdf", "modality": "document"}
    
    graph_builder.update_graph(entities, provenance)
    
    assert "PERSON_JOHN_DOE" in graph_builder._graph
    assert "PROJECT_PMEG" in graph_builder._graph
    assert "FILE_DOC_PDF" in graph_builder._graph # File node should be created

    john_doe_node = graph_builder._graph["PERSON_JOHN_DOE"]
    assert any(rel['type'] == "MENTIONED_IN" and rel['target_id'] == "FILE_DOC_PDF" for rel in john_doe_node["relations"])

    file_node = graph_builder._graph["FILE_DOC_PDF"]
    assert any(rel['type'] == "MENTIONS" and rel['target_id'] == "PERSON_JOHN_DOE" for rel in file_node["relations"])
    assert any(rel['type'] == "MENTIONS" and rel['target_id'] == "PROJECT_PMEG" for rel in file_node["relations"])

def test_mock_graph_builder_update_graph_no_unified_id(caplog):
    graph_builder = MockGraphBuilder()
    
    entities = [
        {"type": "PERSON", "name": "John Doe"}, # Missing unified_id
    ]
    provenance = {"file_path": "/path/to/doc.pdf", "modality": "document"}
    
    with caplog.at_level(logging.WARNING):
        graph_builder.update_graph(entities, provenance)
        assert "Entity without unified_id encountered" in caplog.text
    
    assert "PERSON_JOHN_DOE" not in graph_builder._graph # Should not be added

def test_mock_graph_db_interface_query_entities():
    graph_builder = MockGraphBuilder()
    graph_db = MockGraphDBInterface(graph_builder)

    entities_data = [
        {"type": "PERSON", "name": "John Doe", "unified_id": "PERSON_JOHN_DOE"},
        {"type": "PROJECT", "name": "Project Alpha", "unified_id": "PROJECT_ALPHA"},
        {"type": "PERSON", "name": "Jane Smith", "unified_id": "PERSON_JANE_SMITH"}
    ]
    provenance = {"file_path": "/path/to/file.txt", "modality": "text"}
    graph_builder.update_graph(entities_data, provenance)

    results_person = graph_db.query_entities(entity_type="PERSON")
    assert len(results_person) == 2
    assert all(r['type'] == "PERSON" for r in results_person)

    results_john = graph_db.query_entities(entity_name="John Doe")
    assert len(results_john) == 1
    assert results_john[0]['name'] == "John Doe"

    results_project = graph_db.query_entities(entity_type="PROJECT", entity_name="Project Alpha")
    assert len(results_project) == 1
    assert results_project[0]['name'] == "Project Alpha"

    results_non_existent = graph_db.query_entities(entity_name="NonExistent")
    assert len(results_non_existent) == 0


# --- Unit Tests: Database for Metadata ---

def test_mock_crud_base_operations():
    crud = MockCRUDBase()
    
    # Create
    item1 = crud.create({"name": "Item A", "value": 10})
    item2 = crud.create({"name": "Item B", "value": 20})
    assert item1['id'] == 1
    assert item2['id'] == 2
    assert len(crud.get_all()) == 2

    # Get
    retrieved_item1 = crud.get(1)
    assert retrieved_item1['name'] == "Item A"
    assert crud.get(999) is None

    # Update
    updated_item1 = crud.update(1, {"value": 15})
    assert updated_item1['value'] == 15
    assert crud.get(1)['value'] == 15
    assert crud.update(999, {"value": 100}) is None

    # Delete
    assert crud.delete(2) is True
    assert crud.get(2) is None
    assert len(crud.get_all()) == 1
    assert crud.delete(999) is False

def test_mock_file_metadata_crud_get_by_path():
    file_crud = MockFileMetadataCRUD()
    file_crud.create({"file_path": "/docs/report.pdf", "modality": "document"})
    file_crud.create({"file_path": "/images/photo.jpg", "modality": "image"})

    report_meta = file_crud.get_by_path("/docs/report.pdf")
    assert report_meta is not None
    assert report_meta['modality'] == "document"

    non_existent_meta = file_crud.get_by_path("/docs/non_existent.txt")
    assert non_existent_meta is None


# --- Unit Tests: Contextual Grounding Module ---

@pytest.fixture
def mock_retriever_deps():
    """Mocks dependencies for MockRetriever."""
    vector_store = MockVectorStore()
    graph_builder = MockGraphBuilder()
    graph_db_interface = MockGraphDBInterface(graph_builder)
    file_metadata_crud = MockFileMetadataCRUD()

    # Populate mocks for retriever to find something
    vector_store.upsert("file_vec_1.txt", [0.1]*128, {"modality": "text", "content_snippet": "Vector data about PMEG."})
    vector_store.upsert("file_vec_2.jpg", [0.9]*128, {"modality": "image", "content_snippet": "Vector data about John Doe."})
    
    file_metadata_crud.create({"file_path": "file_vec_1.txt", "modality": "text", "status": "processed"})
    file_metadata_crud.create({"file_path": "file_vec_2.jpg", "modality": "image", "status": "processed"})

    entities_data = [
        {"type": "PERSON", "name": "John Doe", "unified_id": "PERSON_JOHN_DOE"},
        {"type": "PROJECT", "name": "PMEG", "unified_id": "PROJECT_PMEG"},
        {"type": "PERSON", "name": "Jane Smith", "unified_id": "PERSON_JANE_SMITH"}
    ]
    graph_builder.update_graph(entities_data, {"file_path": "file_graph_1.pdf", "modality": "document"})
    graph_builder.update_graph(entities_data, {"file_path": "file_graph_2.mp3", "modality": "audio"})
    
    file_metadata_crud.create({"file_path": "file_graph_1.pdf", "modality": "document", "status": "processed"})
    file_metadata_crud.create({"file_path": "file_graph_2.mp3", "modality": "audio", "status": "processed"})

    return vector_store, graph_db_interface, file_metadata_crud

def test_mock_retriever_retrieve_evidence(mock_retriever_deps):
    vector_store, graph_db_interface, file_metadata_crud = mock_retriever_deps
    retriever = MockRetriever(vector_store, graph_db_interface, file_metadata_crud)
    
    query = "What about PMEG project or John Doe?"
    user_context = {}
    
    evidence = retriever.retrieve_evidence(query, user_context)
    assert len(evidence) >= 2 # At least one from vector store, one from graph
    assert any("PMEG" in e['content_snippet'] for e in evidence)
    assert any("John Doe" in e['content_snippet'] for e in evidence)
    assert any(e['source_type'] == "vector_store" for e in evidence)
    assert any(e['source_type'] == "knowledge_graph" for e in evidence)

def test_mock_ranker_rank_evidence():
    ranker = MockRanker()
    query = "relevant info"
    raw_evidence = [
        {"content_snippet": "This is very relevant info.", "file_path": "f1.txt", "modality": "text", "score": 0.7},
        {"content_snippet": "Less relevant content.", "file_path": "f2.txt", "modality": "text", "score": 0.4},
        {"content_snippet": "Another relevant piece of info here.", "file_path": "f3.jpg", "modality": "image", "score": 0.8},
    ]
    
    ranked_evidence = ranker.rank_evidence(query, raw_evidence, {})
    assert len(ranked_evidence) == 3
    assert ranked_evidence[0].snippet == "Another relevant piece of info here." # Score 0.8 is highest
    assert ranked_evidence[1].snippet == "This is very relevant info." # Score 0.7
    assert ranked_evidence[0].relevance >= ranked_evidence[1].relevance
    assert all(isinstance(e, GroundedEvidence) for e in ranked_evidence)

def test_mock_grounding_api_ground_query(mock_retriever_deps):
    vector_store, graph_db_interface, file_metadata_crud = mock_retriever_deps
    retriever = MockRetriever(vector_store, graph_db_interface, file_metadata_crud)
    ranker = MockRanker()
    grounding_api = MockGroundingAPI(retriever, ranker)
    
    query = "PMEG"
    grounded = grounding_api.ground_query(query)
    assert len(grounded) > 0
    assert any("PMEG" in e.snippet for e in grounded)
    assert all(isinstance(e, GroundedEvidence) for e in grounded)


# --- Unit Tests: Iterative Synthesis Engine ---

@pytest.fixture
def mock_grounding_api_for_synthesis():
    """A mock GroundingAPI that returns predictable evidence for synthesis tests."""
    mock = MagicMock(spec=MockGroundingAPI)
    
    # Simulate different results for different queries/iterations
    def _mock_ground_query(query: str, user_context: Optional[Dict[str, Any]] = None):
        if "PMEG framework" in query:
            return [
                GroundedEvidence(f"Snippet about PMEG core component from doc_{i}.pdf", f"doc_{i}.pdf", "document", 0.9, 0.8 + 0.01*i) 
                for i in range(2)
            ]
        elif "John Doe" in query:
             return [
                GroundedEvidence(f"John Doe's report from video_{i}.mp4", f"video_{i}.mp4", "video", 0.8, 0.7 + 0.01*i) 
                for i in range(1)
            ]
        return []
    
    mock.ground_query.side_effect = _mock_ground_query
    return mock

def test_mock_synthesis_engine_synthesize_response(mock_grounding_api_for_synthesis):
    engine = MockSynthesisEngine(mock_grounding_api_for_synthesis)
    task = "Summarize the PMEG framework and John Doe's involvement."
    
    result = engine.synthesize_response(task, strategy="summarize")
    
    assert result is not None
    assert isinstance(result, SynthesisResult)
    assert "Summary for 'Summarize the PMEG framework and John Doe's involvement.'" in result.answer
    assert len(result.supporting_evidence) > 0
    assert any("PMEG" in e.snippet for e in result.supporting_evidence)
    assert any("John Doe" in e.snippet for e in result.supporting_evidence)
    assert len(result.reasoning_steps) > 0
    assert "Summarization strategy applied" in result.reasoning_steps[1]

def test_mock_synthesis_engine_no_evidence():
    mock_grounding_api = MagicMock(spec=MockGroundingAPI)
    mock_grounding_api.ground_query.return_value = []
    
    engine = MockSynthesisEngine(mock_grounding_api)
    task = "Summarize something non-existent."
    
    result = engine.synthesize_response(task)
    
    assert result is not None
    assert "Could not find sufficient evidence" in result.answer
    assert len(result.supporting_evidence) == 0
    assert len(result.reasoning_steps) == 1


# --- Integration Test: PMEG_Framework Orchestrator ---

def test_pmeg_framework_initialization(pmeg_framework_instance):
    """Test that all components are initialized correctly."""
    framework = pmeg_framework_instance
    assert isinstance(framework.config, Config)
    assert isinstance(framework.file_metadata_crud, MockFileMetadataCRUD)
    assert isinstance(framework.vector_store, MockVectorStore)
    assert isinstance(framework.graph_db_interface, MockGraphDBInterface)
    assert isinstance(framework.file_processor, MockFileProcessor)
    assert isinstance(framework.document_loader, MockDocumentLoader)
    assert len(framework.perception_agents) == 5 # OCR, Image, Audio, Video, Document
    assert isinstance(framework.multimodal_model, MockMultimodalModel)
    assert isinstance(framework.entity_extractor, MockEntityExtractor)
    assert isinstance(framework.disambiguator, MockDisambiguator)
    assert isinstance(framework.grounding_api, MockGroundingAPI)
    assert isinstance(framework.synthesis_engine, MockSynthesisEngine)

def test_pmeg_framework_single_file_processing_full_pipeline(pmeg_framework_instance, temp_processing_dir):
    """
    Test processing a single file through the entire mocked PMEG pipeline.
    This simulates Ingestion -> Perception -> Embedding -> KG Update.
    """
    framework = pmeg_framework_instance
    
    # Create a dummy file that an agent can process
    file_path = os.path.join(temp_processing_dir, "test_report.pdf")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("This is a PDF report about PMEG. It mentions John Doe.")
    
    # Patch perception agent to return entities expected by subsequent steps
    original_ocr_process = framework.perception_agents[0].process
    def mock_ocr_process(fp: str):
        if fp == file_path:
            return StandardizedOutput(
                file_path, "text", f"OCR text from {os.path.basename(fp)}. It mentions 'John Doe' and 'Project PMEG'.",
                entities=[{"type": "PERSON", "name": "John Doe"}, {"type": "PROJECT", "name": "Project PMEG"}]
            )
        return original_ocr_process(fp)

    with patch.object(framework.perception_agents[0], 'process', side_effect=mock_ocr_process):
        framework._process_single_file(file_path)

    # Assertions for each stage:
    # 1. File Metadata
    file_meta = framework.file_metadata_crud.get_by_path(file_path)
    assert file_meta is not None
    assert file_meta['status'] == "perceived_and_processed"
    assert file_meta['modality'] == "text"

    # 2. Vector Store
    vector_results = framework.vector_store.query([0.5]*128, top_k=10) # Query with generic embedding
    assert any(res['file_path'] == file_path for res in vector_results)
    assert any("PMEG" in res.get("content_snippet", "") for res in vector_results)

    # 3. Knowledge Graph
    john_doe_entities = framework.graph_db_interface.query_entities(entity_name="John Doe")
    pmeg_entities = framework.graph_db_interface.query_entities(entity_name="PMEG")
    file_node_entities = framework.graph_db_interface.query_entities(entity_name="test_report.pdf")
    
    assert len(john_doe_entities) == 1
    assert len(pmeg_entities) == 1
    assert len(file_node_entities) == 1
    
    # Check relationships (simplified check for presence)
    john_doe_node_relations = john_doe_entities[0]['relations']
    pmeg_node_relations = pmeg_entities[0]['relations']
    
    assert any(rel['type'] == "MENTIONED_IN" and f"FILE_{os.path.basename(file_path).replace('.', '_').upper()}" in rel['target_id'] for rel in john_doe_node_relations)
    assert any(rel['type'] == "MENTIONED_IN" and f"FILE_{os.path.basename(file_path).replace('.', '_').upper()}" in rel['target_id'] for rel in pmeg_node_relations)


def test_pmeg_framework_ingest_and_process_files(pmeg_framework_instance, temp_processing_dir):
    """Test the public ingestion method for multiple files."""
    framework = pmeg_framework_instance

    # Create multiple dummy files of different types
    file1_path = os.path.join(temp_processing_dir, "doc1.txt")
    file2_path = os.path.join(temp_processing_dir, "image1.jpg")
    file3_path = os.path.join(temp_processing_dir, "video1.mp4")
    
    with open(file1_path, "w", encoding='utf-8') as f: f.write("Text about Project PMEG and Jane Smith.")
    with open(file2_path, "w") as f: f.write("dummy image data") # Content doesn't matter for mock
    with open(file3_path, "w") as f: f.write("dummy video data") # Content doesn't matter for mock

    # Patch perception agents to yield specific outputs
    def mock_doc_parser_process(fp: str):
        if fp == file1_path:
            return StandardizedOutput(
                fp, "document", f"Text about Project PMEG and Jane Smith from {os.path.basename(fp)}.",
                entities=[{"type": "PROJECT", "name": "Project PMEG"}, {"type": "PERSON", "name": "Jane Smith"}]
            )
        return None
    
    def mock_image_captioning_process(fp: str):
        if fp == file2_path:
            return StandardizedOutput(
                fp, "image", f"A photo of Project PMEG team members from {os.path.basename(fp)}.",
                entities=[{"type": "PROJECT", "name": "Project PMEG"}]
            )
        return None

    def mock_video_analysis_process(fp: str):
        if fp == file3_path:
            return StandardizedOutput(
                fp, "video", f"Video analysis summary from {os.path.basename(fp)}: PMEG presentation.",
                entities=[{"type": "PROJECT", "name": "PMEG presentation"}]
            )
        return None

    # Apply patches to specific agents
    with patch.object(framework.perception_agents[4], 'process', side_effect=mock_doc_parser_process), \
         patch.object(framework.perception_agents[1], 'process', side_effect=mock_image_captioning_process), \
         patch.object(framework.perception_agents[3], 'process', side_effect=mock_video_analysis_process):

        framework.ingest_and_process_files(temp_processing_dir)

    # Verify all files were processed and added to metadata
    assert framework.file_metadata_crud.get_by_path(file1_path) is not None
    assert framework.file_metadata_crud.get_by_path(file2_path) is not None
    assert framework.file_metadata_crud.get_by_path(file3_path) is not None

    # Verify at least one entity from each file type found its way into the graph/vector store (simplified check)
    assert len(framework.vector_store._store) >= 3 # At least 3 embeddings
    assert len(framework.graph_builder._graph) >= 3 # At least 3 file nodes and entities

    assert any(e['name'] == "Project PMEG" for e in framework.graph_db_interface.query_entities(entity_type="PROJECT"))
    assert any(e['name'] == "Jane Smith" for e in framework.graph_db_interface.query_entities(entity_type="PERSON"))

def test_pmeg_framework_query_grounding(pmeg_framework_instance, temp_processing_dir):
    """Test the public API for contextual grounding."""
    framework = pmeg_framework_instance
    
    # Pre-populate framework with some data (minimal for query to hit)
    file_path = os.path.join(temp_processing_dir, "query_target.txt")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("This file is about Project PMEG and discusses milestones.")
    
    # Manually add to mock components to simulate prior processing
    framework.file_metadata_crud.create({"file_path": file_path, "modality": "text", "status": "processed"})
    framework.vector_store.upsert(file_path, [0.8]*128, {"modality": "text", "content_snippet": "This file is about Project PMEG and discusses milestones."})
    
    entities_data = [{"type": "PROJECT", "name": "Project PMEG", "unified_id": "PROJECT_PMEG"}]
    framework.graph_builder.update_graph(entities_data, {"file_path": file_path, "modality": "text"})

    query = "What are the milestones of Project PMEG?"
    grounded_evidence = framework.query_grounding(query)
    
    assert len(grounded_evidence) > 0
    assert any("PMEG" in ev.snippet for ev in grounded_evidence)
    assert all(isinstance(ev, GroundedEvidence) for ev in grounded_evidence)

def test_pmeg_framework_perform_synthesis(pmeg_framework_instance, temp_processing_dir):
    """Test the public API for iterative synthesis."""
    framework = pmeg_framework_instance

    # Pre-populate framework with some data
    file_path_1 = os.path.join(temp_processing_dir, "synthesis_doc1.txt")
    file_path_2 = os.path.join(temp_processing_dir, "synthesis_doc2.pdf")
    
    with open(file_path_1, "w", encoding='utf-8') as f: f.write("Jane Smith confirmed the PMEG launch date.")
    with open(file_path_2, "w", encoding='utf-8') as f: f.write("Project PMEG is a key initiative. John Doe is leading it.")
    
    # Manually add to mock components to simulate prior processing
    framework.file_metadata_crud.create({"file_path": file_path_1, "modality": "text", "status": "processed"})
    framework.file_metadata_crud.create({"file_path": file_path_2, "modality": "document", "status": "processed"})

    framework.vector_store.upsert(file_path_1, [0.7]*128, {"modality": "text", "content_snippet": "Jane Smith confirmed the PMEG launch date."})
    framework.vector_store.upsert(file_path_2, [0.9]*128, {"modality": "document", "content_snippet": "Project PMEG is a key initiative. John Doe is leading it."})
    
    entities_data_1 = [{"type": "PERSON", "name": "Jane Smith", "unified_id": "PERSON_JANE_SMITH"}, {"type": "PROJECT", "name": "PMEG", "unified_id": "PROJECT_PMEG"}]
    entities_data_2 = [{"type": "PERSON", "name": "John Doe", "unified_id": "PERSON_JOHN_DOE"}, {"type": "PROJECT", "name": "PMEG", "unified_id": "PROJECT_PMEG"}]

    framework.graph_builder.update_graph(entities_data_1, {"file_path": file_path_1, "modality": "text"})
    framework.graph_builder.update_graph(entities_data_2, {"file_path": file_path_2, "modality": "document"})

    task = "Provide a detailed report on Project PMEG, including key personnel and dates."
    synthesis_result = framework.perform_synthesis(task, strategy="detailed_report")
    
    assert synthesis_result is not None
    assert isinstance(synthesis_result, SynthesisResult)
    assert "Detailed Report for 'Provide a detailed report on Project PMEG" in synthesis_result.answer
    assert any("Jane Smith" in synthesis_result.answer for ev in synthesis_result.supporting_evidence)
    assert any("John Doe" in synthesis_result.answer for ev in synthesis_result.supporting_evidence)
    assert len(synthesis_result.supporting_evidence) > 0
    assert len(synthesis_result.reasoning_steps) > 0
    assert "Detailed report strategy applied" in synthesis_result.reasoning_steps[-1]

def test_pmeg_framework_process_file_no_perception_agent(pmeg_framework_instance, temp_processing_dir, caplog):
    """Test _process_single_file when no perception agent can handle the file."""
    framework = pmeg_framework_instance
    
    # Create a file type that none of the mock agents process (e.g., .bin)
    file_path = os.path.join(temp_processing_dir, "unsupported_file.bin")
    with open(file_path, "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    
    with caplog.at_level(logging.WARNING):
        framework._process_single_file(file_path)
        assert "No suitable perception agent found" in caplog.text

    # Should still create basic metadata
    file_meta = framework.file_metadata_crud.get_by_path(file_path)
    assert file_meta is not None
    assert file_meta['status'] == "processed_no_perception"
    assert file_meta['modality'] == "unknown" # Default if no agent processes

    # Should NOT have been added to vector store or graph
    vector_results = framework.vector_store.query([0.5]*128)
    assert not any(res['file_path'] == file_path for res in vector_results)
    
    graph_entities = framework.graph_db_interface.query_entities(entity_name="unsupported_file.bin")
    assert len(graph_entities) == 0

```