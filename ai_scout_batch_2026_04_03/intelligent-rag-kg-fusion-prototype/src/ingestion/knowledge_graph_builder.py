```python
import spacy
from typing import List, Dict, Union, Any
import logging
import uuid

# --- Logger setup ---
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import src.utils.logger. Using basic logging configuration.")

# --- Schema definitions (fallback if src/utils/schemas.py is not available) ---
# These classes are designed to be Pydantic-like data structures.
# In a full system, they would inherit from a BaseSchema/BaseModel for validation and serialization.
try:
    from src.utils.schemas import (
        KGA_Entity,
        KGA_Relationship,
        KGA_Attribute,
        KGA_Event,
        KnowledgeGraphElements  # A wrapper for all elements
    )
    logger.info("Successfully imported KG schemas from src.utils.schemas.")
except ImportError:
    logger.warning("Could not import src.utils.schemas. Defining basic KG structures locally.")

    class KGA_Entity:
        def __init__(self, id: str, label: str, type: str, source_chunk_id: str = None, attributes: Dict[str, Any] = None):
            self.id = id
            self.label = label
            self.type = type
            self.source_chunk_id = source_chunk_id
            self.attributes = attributes if attributes is not None else {}

        def to_dict(self):
            return {"id": self.id, "label": self.label, "type": self.type, "source_chunk_id": self.source_chunk_id, "attributes": self.attributes}

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            if not isinstance(other, KGA_Entity):
                return NotImplemented
            return self.id == other.id

    class KGA_Relationship:
        def __init__(self, id: str, head_entity_id: str, tail_entity_id: str, type: str, source_chunk_id: str = None, attributes: Dict[str, Any] = None):
            self.id = id
            self.head_entity_id = head_entity_id
            self.tail_entity_id = tail_entity_id
            self.type = type
            self.source_chunk_id = source_chunk_id
            self.attributes = attributes if attributes is not None else {}

        def to_dict(self):
            return {"id": self.id, "head": self.head_entity_id, "tail": self.tail_entity_id, "type": self.type, "source_chunk_id": self.source_chunk_id, "attributes": self.attributes}

    class KGA_Attribute:
        def __init__(self, id: str, entity_id: str, key: str, value: Any, source_chunk_id: str = None):
            self.id = id
            self.entity_id = entity_id
            self.key = key
            self.value = value
            self.source_chunk_id = source_chunk_id

        def to_dict(self):
            return {"id": self.id, "entity_id": self.entity_id, "key": self.key, "value": self.value, "source_chunk_id": self.source_chunk_id}

    class KGA_Event:
        def __init__(self, id: str, description: str, type: str, participants: List[str] = None, temporal_info: Dict[str, Any] = None, source_chunk_id: str = None, attributes: Dict[str, Any] = None):
            self.id = id
            self.description = description
            self.type = type
            self.participants = participants if participants is not None else []
            self.temporal_info = temporal_info if temporal_info is not None else {}
            self.source_chunk_id = source_chunk_id
            self.attributes = attributes if attributes is not None else {}

        def to_dict(self):
            return {"id": self.id, "description": self.description, "type": self.type, "participants": self.participants, "temporal_info": self.temporal_info, "source_chunk_id": self.source_chunk_id, "attributes": self.attributes}

    class KnowledgeGraphElements:
        def __init__(self, entities: List[KGA_Entity] = None, relationships: List[KGA_Relationship] = None, attributes: List[KGA_Attribute] = None, events: List[KGA_Event] = None):
            self.entities = entities if entities is not None else []
            self.relationships = relationships if relationships is not None else []
            self.attributes = attributes if attributes is not None else []
            self.events = events if events is not None else []

        def to_dict(self):
            return {
                "entities": [e.to_dict() for e in self.entities],
                "relationships": [r.to_dict() for r in self.relationships],
                "attributes": [a.to_dict() for a in self.attributes],
                "events": [e.to_dict() for e in self.events],
            }

# --- Helper function for ID generation ---
def _generate_id(prefix: str = "kge") -> str:
    """Generates a unique ID with a given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class KnowledgeGraphBuilder:
    """
    Extracts entities, relationships, attributes, and events from text and
    structured data to construct elements for a knowledge graph.
    """
    def __init__(self, nlp_model_name: str = "en_core_web_sm"):
        """
        Initializes the KnowledgeGraphBuilder with an NLP model.

        Args:
            nlp_model_name (str): The name of the spaCy model to load.
        """
        try:
            self.nlp = spacy.load(nlp_model_name)
            logger.info(f"Loaded spaCy model: {nlp_model_name}")
        except OSError:
            logger.error(f"spaCy model '{nlp_model_name}' not found. Attempting to download...")
            try:
                spacy.cli.download(nlp_model_name)
                self.nlp = spacy.load(nlp_model_name)
                logger.info(f"Successfully downloaded and loaded spaCy model: {nlp_model_name}")
            except Exception as e:
                logger.critical(f"Failed to download and load spaCy model '{nlp_model_name}': {e}. "
                                "Knowledge Graph Builder will not function correctly for text processing.")
                self.nlp = None # Set to None to indicate failure, other methods should check this.
                raise

        # Dictionary to store mapping from a unique entity key (label-type) to KGA_Entity object
        # Used for de-duplication within a single data unit processing.
        self._known_entities: Dict[str, KGA_Entity] = {}

    def _get_or_create_entity(self, label: str, entity_type: str, source_chunk_id: str) -> KGA_Entity:
        """
        Retrieves an existing entity or creates a new one if it doesn't exist within the
        current processing scope (i.e., this chunk/record).
        Simple de-duplication based on label and type.
        """
        # Normalize label for consistent key generation
        normalized_label = label.strip().lower()
        key = f"{normalized_label}-{entity_type}"
        
        if key not in self._known_entities:
            entity_id = _generate_id("ent")
            entity = KGA_Entity(id=entity_id, label=label, type=entity_type, source_chunk_id=source_chunk_id)
            self._known_entities[key] = entity
            logger.debug(f"Created new entity: {entity.label} ({entity.type}) - ID: {entity.id}")
        else:
            logger.debug(f"Reusing existing entity: {self._known_entities[key].label} ({self._known_entities[key].type}) - ID: {self._known_entities[key].id}")
        return self._known_entities[key]

    def _extract_entities_from_text(self, text: str, chunk_id: str):
        """
        Extracts entities from a given text using spaCy's NER and populates _known_entities.
        """
        if not self.nlp:
            logger.warning("NLP model not loaded. Skipping entity extraction from text.")
            return

        doc = self.nlp(text)
        for ent in doc.ents:
            # Use _get_or_create_entity to manage uniqueness for this chunk
            self._get_or_create_entity(
                label=ent.text.strip(),
                entity_type=ent.label_,
                source_chunk_id=chunk_id
            )
            
    def _extract_relationships_from_text(self, text: str, chunk_id: str) -> List[KGA_Relationship]:
        """
        Extracts relationships between entities from text using dependency parsing.
        Focuses on simple subject-verb-object patterns.
        """
        if not self.nlp:
            logger.warning("NLP model not loaded. Skipping relationship extraction from text.")
            return []

        doc = self.nlp(text)
        relationships: List[KGA_Relationship] = []

        for sent in doc.sents:
            for token in sent:
                if "VERB" in token.pos_:
                    subject_token = None
                    object_token = None
                    verb_lemma = token.lemma_

                    # Find subject (nsubj)
                    for child in token.children:
                        if "nsubj" in child.dep_:
                            subject_token = child
                            break

                    # Find direct object (dobj) or nominal object (pobj)
                    for child in token.children:
                        if "dobj" in child.dep_ or "pobj" in child.dep_:
                            object_token = child
                            break

                    if subject_token and object_token:
                        # Ensure entities are managed by _get_or_create_entity
                        subject_obj = self._get_or_create_entity(
                            label=subject_token.text.strip(),
                            entity_type=subject_token.ent_type_ if subject_token.ent_type_ else "GENERIC_NOUN",
                            source_chunk_id=chunk_id
                        )
                        object_obj = self._get_or_create_entity(
                            label=object_token.text.strip(),
                            entity_type=object_token.ent_type_ if object_token.ent_type_ else "GENERIC_NOUN",
                            source_chunk_id=chunk_id
                        )
                        
                        rel_id = _generate_id("rel")
                        relationship = KGA_Relationship(
                            id=rel_id,
                            head_entity_id=subject_obj.id,
                            tail_entity_id=object_obj.id,
                            type=verb_lemma.upper().replace(" ", "_"),
                            source_chunk_id=chunk_id
                        )
                        relationships.append(relationship)
                        logger.debug(f"Extracted relationship: {subject_obj.label} -[{relationship.type}]-> {object_obj.label}")

        return relationships

    def _extract_attributes_from_text(self, text: str, chunk_id: str) -> List[KGA_Attribute]:
        """
        Extracts attributes for entities. For simplicity, this looks for
        adjectives or numerical values directly attached to recognized entities.
        """
        if not self.nlp:
            logger.warning("NLP model not loaded. Skipping attribute extraction from text.")
            return []

        doc = self.nlp(text)
        attributes: List[KGA_Attribute] = []
        
        for ent in doc.ents:
            entity_obj = self._get_or_create_entity(
                label=ent.text.strip(),
                entity_type=ent.label_,
                source_chunk_id=chunk_id
            )

            # Look for adjectives directly modifying the entity's root token
            for token in ent.root.children: 
                if token.pos_ == "ADJ":
                    attr_id = _generate_id("attr")
                    attribute = KGA_Attribute(
                        id=attr_id,
                        entity_id=entity_obj.id,
                        key="has_property",
                        value=token.text,
                        source_chunk_id=chunk_id
                    )
                    attributes.append(attribute)
                    logger.debug(f"Extracted attribute: {entity_obj.label} has_property {attribute.value}")
                
                # Consider numerical values near entities as potential attributes
                if token.like_num and token.head == ent.root:
                    try:
                        numerical_value = float(token.text.replace(',', ''))
                        attr_id = _generate_id("attr")
                        attribute = KGA_Attribute(
                            id=attr_id,
                            entity_id=entity_obj.id,
                            key="numerical_value",
                            value=numerical_value,
                            source_chunk_id=chunk_id
                        )
                        attributes.append(attribute)
                        logger.debug(f"Extracted numerical attribute: {entity_obj.label} has numerical_value {attribute.value}")
                    except ValueError:
                        logger.debug(f"Could not convert '{token.text}' to float for attribute.")
        return attributes

    def _extract_events_from_text(self, text: str, chunk_id: str) -> List[KGA_Event]:
        """
        Extracts events from text. This is a simplified approach for the prototype,
        focusing on action verbs and associated temporal/location expressions.
        """
        if not self.nlp:
            logger.warning("NLP model not loaded. Skipping event extraction from text.")
            return []

        doc = self.nlp(text)
        events: List[KGA_Event] = []

        for token in doc:
            if "VERB" in token.pos_ and token.dep_ != "aux": # Exclude auxiliary verbs
                event_description = token.text
                event_type = token.lemma_.upper() + "_EVENT"
                participants_ids = []
                temporal_info = {}

                # Look for associated subjects/objects (participants)
                for child in token.children:
                    if child.dep_ in ["nsubj", "dobj", "pobj"]:
                        participant_entity = self._get_or_create_entity(
                            label=child.text.strip(),
                            entity_type=child.ent_type_ if child.ent_type_ else "GENERIC_PARTICIPANT",
                            source_chunk_id=chunk_id
                        )
                        participants_ids.append(participant_entity.id)

                # Look for temporal and geographical expressions using spaCy's NER
                for ent in doc.ents:
                    if ent.label_ in ["DATE", "TIME", "EVENT", "GPE", "LOC"]: 
                        temporal_info[ent.label_] = ent.text

                # Ensure unique participants
                unique_participants = list(set(participants_ids))

                event_id = _generate_id("event")
                event = KGA_Event(
                    id=event_id,
                    description=event_description,
                    type=event_type,
                    participants=unique_participants,
                    temporal_info=temporal_info,
                    source_chunk_id=chunk_id
                )
                events.append(event)
                logger.debug(f"Extracted event: {event.description} ({event.type}) with participants {event.participants}")

        return events

    def _process_text_chunk(self, chunk_text: str, chunk_metadata: Dict) -> KnowledgeGraphElements:
        """
        Orchestrates the extraction of KG elements from a single text chunk.
        """
        chunk_id = chunk_metadata.get("chunk_id", _generate_id("chunk"))
        logger.info(f"Processing text chunk (ID: {chunk_id})...")

        # Clear known entities to ensure de-duplication is scoped to this chunk
        self._known_entities.clear()

        # Step 1: Extract initial entities and populate _known_entities
        self._extract_entities_from_text(chunk_text, chunk_id)
        
        # Step 2: Extract relationships (will use/create entities via _get_or_create_entity)
        relationships = self._extract_relationships_from_text(chunk_text, chunk_id)
        
        # Step 3: Extract attributes (will use/create entities via _get_or_create_entity)
        attributes = self._extract_attributes_from_text(chunk_text, chunk_id)
        
        # Step 4: Extract events (will use/create entities via _get_or_create_entity)
        events = self._extract_events_from_text(chunk_text, chunk_id)

        # All entities discovered during the process for this chunk are in _known_entities
        all_unique_entities_for_chunk = list(self._known_entities.values())
        
        # Clear known entities for the next data unit processing
        self._known_entities.clear() 

        logger.info(f"Finished processing chunk {chunk_id}. Found {len(all_unique_entities_for_chunk)} entities, "
                         f"{len(relationships)} relationships, {len(attributes)} attributes, {len(events)} events.")
        return KnowledgeGraphElements(
            entities=all_unique_entities_for_chunk,
            relationships=relationships,
            attributes=attributes,
            events=events
        )

    def _process_structured_data(self, data: Dict, metadata: Dict) -> KnowledgeGraphElements:
        """
        Extracts KG elements from structured data (e.g., a row from a database table).
        Assumes 'data' is a dictionary where keys are column names and values are data.
        """
        record_id = metadata.get("record_id", _generate_id("record"))
        source_name = metadata.get("source_name", "StructuredData")
        logger.info(f"Processing structured data record (ID: {record_id})...")

        # Clear known entities to ensure de-duplication is scoped to this record
        self._known_entities.clear()

        attributes: List[KGA_Attribute] = []
        relationships: List[KGA_Relationship] = [] 
        events: List[KGA_Event] = [] # Less likely from single structured record unless explicitly defined

        # Create a primary entity for the record itself
        # Prioritize 'name' or 'title' for label, fallback to ID
        primary_entity_label = data.get("name", data.get("title", str(data.get("id", f"{source_name}_Record_{record_id}"))))
        primary_entity_type = metadata.get("entity_type", source_name.replace(" ", "_").upper() + "_RECORD")
        primary_entity = self._get_or_create_entity(primary_entity_label, primary_entity_type, record_id)
        
        # Iterate through data items to create attributes and potential relationships
        for key, value in data.items():
            # Skip keys that are already used for primary entity (e.g., 'id', 'name', 'title') or are complex objects
            if key in ["id", "name", "title"] or isinstance(value, (dict, list)):
                continue

            attr_id = _generate_id("attr")
            attribute = KGA_Attribute(
                id=attr_id,
                entity_id=primary_entity.id,
                key=key,
                value=value,
                source_chunk_id=record_id
            )
            attributes.append(attribute)
            logger.debug(f"Extracted attribute for {primary_entity.label}: {key}={value}")

            # If there's a reference to another entity (e.g., "author_id"), create a relationship
            if key.endswith("_id") and isinstance(value, (str, int)): 
                # Attempt to find a more meaningful label for the related entity from other fields, or use a generic one
                related_entity_label_candidate = data.get(key.replace('_id', '_name'), data.get(key.replace('_id', '_title'), f"{key.replace('_id', '').capitalize()}_{value}"))
                related_entity_type = key.replace('_id', '').upper()
                
                related_entity = self._get_or_create_entity(str(related_entity_label_candidate), related_entity_type, record_id)
                
                rel_id = _generate_id("rel")
                relationship = KGA_Relationship(
                    id=rel_id,
                    head_entity_id=primary_entity.id,
                    tail_entity_id=related_entity.id,
                    type=f"HAS_{key.upper()}", # e.g., HAS_AUTHOR_ID
                    source_chunk_id=record_id
                )
                relationships.append(relationship)
                logger.debug(f"Extracted relationship: {primary_entity.label} -[{relationship.type}]-> {related_entity.label}")

        # All entities for this record (primary + related) are in _known_entities
        all_unique_entities_for_record = list(self._known_entities.values())

        # Clear known entities for the next data unit processing
        self._known_entities.clear()

        logger.info(f"Finished processing record {record_id}. Found {len(all_unique_entities_for_record)} entities, "
                         f"{len(relationships)} relationships, {len(attributes)} attributes.")

        return KnowledgeGraphElements(
            entities=all_unique_entities_for_record,
            relationships=relationships,
            attributes=attributes,
            events=events
        )


    def build_knowledge_graph_elements(self, data_unit: Union[str, Dict], metadata: Dict = None) -> KnowledgeGraphElements:
        """
        Main public method to extract KG elements from a data unit (text chunk or structured data).

        Args:
            data_unit (Union[str, Dict]): The data to process. Can be a text string (chunk)
                                          or a dictionary representing structured data (e.g., a database row).
            metadata (Dict, optional): Associated metadata for the data unit (e.g., chunk_id, record_id, source_name).
                                       Defaults to None.

        Returns:
            KnowledgeGraphElements: An object containing lists of extracted entities, relationships, attributes, and events.
        """
        if metadata is None:
            metadata = {}

        try:
            if isinstance(data_unit, str):
                return self._process_text_chunk(data_unit, metadata)
            elif isinstance(data_unit, dict):
                return self._process_structured_data(data_unit, metadata)
            else:
                logger.error(f"Unsupported data_unit type: {type(data_unit)}. Must be str (text chunk) or dict (structured data).")
                raise ValueError("Unsupported data_unit type. Must be str (text chunk) or dict (structured data).")
        except Exception as e:
            logger.error(f"Error building knowledge graph elements from data unit: {e}", exc_info=True)
            # Return empty elements on error to prevent pipeline failure
            return KnowledgeGraphElements()

```