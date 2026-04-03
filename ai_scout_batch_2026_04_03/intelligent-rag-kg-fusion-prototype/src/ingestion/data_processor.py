```python
import uuid
import logging
from typing import List, Dict, Any, Union

# Assuming these modules exist in their respective paths
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_strategy import ChunkingStrategy
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.knowledge_graph_builder import KnowledgeGraphBuilder

from src.utils.embedding_model import EmbeddingModel
from src.utils.schemas import Document, Chunk, StructuredData # Assuming these schemas are defined
from src.utils.logger import get_logger

from src.retrieval.vector_store import VectorStore
from src.retrieval.kg_store import KGStore
from src.retrieval.structured_data_store import StructuredDataStore

logger = get_logger(__name__)

class DataProcessor:
    """
    Orchestrates the entire data ingestion pipeline, taking raw data through
    loading, chunking, metadata extraction, embedding, and populating various
    stores (vector, knowledge graph, structured).
    """
    def __init__(self,
                 document_loader: DocumentLoader,
                 chunking_strategy: ChunkingStrategy,
                 metadata_extractor: MetadataExtractor,
                 kg_builder: KnowledgeGraphBuilder,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 kg_store: KGStore,
                 structured_data_store: StructuredDataStore = None
                ):
        """
        Initializes the DataProcessor with instances of all necessary ingestion
        and storage components.

        Args:
            document_loader (DocumentLoader): Component for loading diverse document types.
            chunking_strategy (ChunkingStrategy): Component for splitting documents into logical chunks.
            metadata_extractor (MetadataExtractor): Component for extracting metadata.
            kg_builder (KnowledgeGraphBuilder): Component for extracting KG elements.
            embedding_model (EmbeddingModel): Component for generating text embeddings.
            vector_store (VectorStore): Interface for storing vector embeddings.
            kg_store (KGStore): Interface for storing knowledge graph elements.
            structured_data_store (StructuredDataStore, optional): Interface for storing structured data.
                                                                     Defaults to None if not used.
        """
        self.document_loader = document_loader
        self.chunking_strategy = chunking_strategy
        self.metadata_extractor = metadata_extractor
        self.kg_builder = kg_builder
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.kg_store = kg_store
        self.structured_data_store = structured_data_store

    def process_data(self, data_source: str, source_type: str = "filepath", **kwargs) -> Dict[str, Any]:
        """
        Orchestrates the ingestion pipeline for a given data source.

        The process typically involves:
        1. Loading the document.
        2. Extracting document-level metadata.
        3. Handling structured data separately if identified.
        4. Chunking the document into smaller, logical units.
        5. For each chunk:
           a. Extracting chunk-specific metadata.
           b. Generating text embeddings.
           c. Storing the chunk and its embedding in the vector store.
           d. Extracting entities and relationships for the knowledge graph.
        6. Storing all extracted knowledge graph components in the KG store.

        Args:
            data_source (str): The path, URL, or raw content for the data.
            source_type (str): Type of the data source (e.g., "filepath", "url", "raw_text", "database").
            **kwargs: Additional parameters for loading, chunking, or metadata extraction,
                      e.g., `loader_args`, `chunking_args`, `metadata_args`, `table_name`.

        Returns:
            Dict[str, Any]: A summary of the ingestion process, including IDs of ingested data
                            and counts of stored items.
        """
        logger.info(f"Starting data ingestion for source: {data_source}, type: {source_type}")
        document_id = str(uuid.uuid4()) # Generate a unique ID for the document being processed
        ingestion_summary = {
            "document_id": document_id,
            "vector_store_chunks_count": 0,
            "kg_entities_count": 0,
            "kg_relationships_count": 0,
            "structured_data_rows_count": 0,
            "status": "failed",
            "error_message": None
        }

        try:
            # 1. Load Document
            logger.debug(f"Loading document from source: {data_source}")
            document: Union[Document, StructuredData] = self.document_loader.load(
                data_source,
                source_type=source_type,
                document_id=document_id,
                **kwargs.get("loader_args", {})
            )
            logger.info(f"Document '{document.doc_id}' ({document.doc_type}) loaded successfully.")

            # 2. Extract Document-level Metadata
            doc_metadata = self.metadata_extractor.extract_document_metadata(
                document,
                **kwargs.get("metadata_args", {})
            )
            document.metadata.update(doc_metadata) # Merge extracted metadata into the document object
            logger.debug(f"Document-level metadata extracted for '{document.doc_id}'.")

            # Handle structured data if applicable (e.g., CSV, database table dumps)
            if isinstance(document, StructuredData):
                if self.structured_data_store:
                    logger.info(f"Document '{document.doc_id}' identified as structured data. Storing directly.")
                    table_name = kwargs.get("table_name", document.table_name or document.doc_id)
                    rows_inserted = self.structured_data_store.insert_data(table_name, document.rows)
                    ingestion_summary["structured_data_rows_count"] = rows_inserted
                    ingestion_summary["status"] = "completed"
                    logger.info(f"Structured data from '{document.doc_id}' stored in table '{table_name}'. Rows: {rows_inserted}")
                else:
                    logger.warning(f"Structured data detected for '{document.doc_id}' but no structured data store configured. Skipping structured data storage.")
                    ingestion_summary["status"] = "completed_partially_structured_skipped"
                return ingestion_summary # Structured data usually doesn't go through chunking/embedding/KG building

            # Proceed with text-based document processing (chunking, embedding, KG)
            # 3. Chunk Document
            chunks: List[Chunk] = self.chunking_strategy.chunk(
                document,
                **kwargs.get("chunking_args", {})
            )
            logger.info(f"Document '{document.doc_id}' chunked into {len(chunks)} parts.")

            all_extracted_entities = []
            all_extracted_relationships = []

            for i, chunk in enumerate(chunks):
                # Inherit document metadata and add chunk-specific metadata
                chunk.metadata.update(document.metadata)
                chunk.metadata["chunk_index"] = i
                chunk.metadata["document_title"] = document.metadata.get("title", f"Document {document.doc_id}")

                # 4. Extract Chunk-level Metadata
                chunk_metadata = self.metadata_extractor.extract_chunk_metadata(
                    chunk,
                    **kwargs.get("metadata_args", {})
                )
                chunk.metadata.update(chunk_metadata)
                logger.debug(f"Metadata extracted for chunk {chunk.chunk_id}.")

                # 5. Generate Embeddings
                try:
                    embedding = self.embedding_model.get_embedding(chunk.text)
                    if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                        logger.warning(f"Embedding generation failed or returned empty for chunk {chunk.chunk_id}. Skipping vector store entry for this chunk.")
                        continue
                except Exception as emb_e:
                    logger.warning(f"Error generating embedding for chunk {chunk.chunk_id}: {emb_e}. Skipping vector store entry.")
                    continue

                # 6. Store in Vector Store
                try:
                    self.vector_store.add_chunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        embedding=embedding,
                        metadata=chunk.metadata
                    )
                    ingestion_summary["vector_store_chunks_count"] += 1
                    logger.debug(f"Chunk {chunk.chunk_id} added to vector store.")
                except Exception as vs_e:
                    logger.error(f"Failed to add chunk {chunk.chunk_id} to vector store: {vs_e}")

                # 7. Extract and queue KG components
                try:
                    entities, relationships = self.kg_builder.extract_and_add_to_graph(
                        chunk.text,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id
                    )
                    all_extracted_entities.extend(entities)
                    all_extracted_relationships.extend(relationships)
                    logger.debug(f"KG components extracted from chunk {chunk.chunk_id}: {len(entities)} entities, {len(relationships)} relationships.")
                except Exception as kg_e:
                    logger.error(f"Failed to extract KG components from chunk {chunk.chunk_id}: {kg_e}")

            # 8. Store KG Components (batch insert for efficiency)
            try:
                if all_extracted_entities:
                    inserted_entities = self.kg_store.add_entities(all_extracted_entities)
                    ingestion_summary["kg_entities_count"] = len(inserted_entities)
                    logger.info(f"Added {len(inserted_entities)} entities to KG store for document {document.doc_id}.")
                if all_extracted_relationships:
                    inserted_relationships = self.kg_store.add_relationships(all_extracted_relationships)
                    ingestion_summary["kg_relationships_count"] = len(inserted_relationships)
                    logger.info(f"Added {len(inserted_relationships)} relationships to KG store for document {document.doc_id}.")
            except Exception as kg_batch_e:
                logger.error(f"Failed to add batch KG components for document {document.doc_id}: {kg_batch_e}")


            ingestion_summary["status"] = "completed"
            logger.info(f"Data ingestion for '{document.doc_id}' completed successfully. Summary: {ingestion_summary}")

        except Exception as e:
            logger.exception(f"Critical error during data ingestion for source {data_source}: {e}")
            ingestion_summary["status"] = "failed"
            ingestion_summary["error_message"] = str(e)

        return ingestion_summary
```