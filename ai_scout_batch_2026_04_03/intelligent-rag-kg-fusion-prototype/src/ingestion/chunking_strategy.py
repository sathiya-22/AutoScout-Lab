import abc
import hashlib
import re
from typing import List, Dict, Any, Optional

# Assume these utilities are available from src/utils/
try:
    from src.utils.embedding_model import EmbeddingModel
    from src.utils.logger import get_logger
except ImportError:
    # Fallback for testing or if utils are not yet fully implemented
    class EmbeddingModel:
        def __init__(self, model_name: str = "default"):
            print(f"Warning: Using mock EmbeddingModel for {model_name}")
        def encode(self, texts: List[str]) -> Any:
            import numpy as np
            # Simulate embeddings with random vectors
            return [np.random.rand(768) for _ in texts]
    
    class MockLogger:
        def debug(self, msg, *args, **kwargs): pass
        def info(self, msg, *args, **kwargs): pass
        def warning(self, msg, *args, **kwargs): pass
        def error(self, msg, *args, **kwargs): pass
        def exception(self, msg, *args, **kwargs): pass
    
    _logger = MockLogger()
    get_logger = lambda name: _logger

_logger = get_logger(__name__)

class ChunkingStrategy(abc.ABC):
    """
    Abstract base class for all chunking strategies.
    Defines the interface for how documents are broken down into logical chunks.
    """
    @abc.abstractmethod
    def chunk(self, document_id: str, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks the input text into a list of dictionaries, where each dictionary
        represents a chunk and contains its text and metadata.

        Args:
            document_id (str): A unique identifier for the original document.
            text (str): The raw text content of the document.
            document_metadata (Optional[Dict[str, Any]]): Optional metadata associated
                                                           with the entire document.

        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries. Each dictionary
                                  must contain at least 'text' and 'metadata'.
                                  The 'metadata' should include 'document_id' and
                                  'chunk_id'.
        """
        pass

class RecursiveCharacterChunkingStrategy(ChunkingStrategy):
    """
    Implements a recursive character text splitting strategy, similar to LangChain's
    RecursiveCharacterTextSplitter. It attempts to split along various separators
    in order to keep chunks semantically coherent where possible.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initializes the recursive character chunking strategy.

        Args:
            chunk_size (int): The maximum number of characters in a chunk.
            chunk_overlap (int): The number of characters to overlap between adjacent chunks.
            separators (Optional[List[str]]): A list of separators to try in order.
                                               Defaults to common markdown/text separators.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            _logger.warning("chunk_overlap is greater than or equal to chunk_size. This may lead to redundant chunks.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""] # Try double newlines, then single, then space, then char

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Splits text by the given separators, prioritizing longer separators.
        """
        if not text:
            return []
        
        final_chunks: List[str] = []
        current_text = text

        for separator in separators:
            if separator == "": # Character-level split fallback
                if len(current_text) > self.chunk_size:
                    # If current_text is still too long and no other separator worked,
                    # just split by chunk_size without overlap for this stage.
                    # Overlap will be handled later.
                    for i in range(0, len(current_text), self.chunk_size):
                        final_chunks.append(current_text[i:i + self.chunk_size])
                    current_text = "" # Processed
                break # No more separators to try if we reached character-level

            parts = current_text.split(separator)
            temp_chunks = []
            
            for part in parts:
                if len(part) >= self.chunk_size:
                    # If a part is still too large, it needs further splitting
                    # with the *next* separator in the list, or character-level.
                    # So, we append it to be processed further if needed.
                    temp_chunks.append(part)
                else:
                    # This part is small enough, append it to final chunks
                    final_chunks.append(part)
            
            # If all parts were small enough, we are done with this level of separator
            if not temp_chunks and parts:
                break
            
            # If some parts were still too large, we re-assemble and try next separator
            # We add back separators to keep context when recursing
            current_text = separator.join(temp_chunks)
            if not current_text: # All parts were small enough
                break
        
        if current_text: # If there's still unprocessed text
            # This handles the case where the last separator was "" (character level)
            # or if the text was just one very long block without any of the separators
            # that was greater than chunk_size.
            if len(current_text) > self.chunk_size:
                for i in range(0, len(current_text), self.chunk_size):
                    final_chunks.append(current_text[i:i + self.chunk_size])
            else:
                final_chunks.append(current_text)


        # Now, consolidate and apply overlap
        merged_chunks = []
        current_chunk_start = 0
        while current_chunk_start < len(final_chunks):
            current_chunk_text = ""
            current_idx = current_chunk_start
            
            while current_idx < len(final_chunks) and len(current_chunk_text) < self.chunk_size:
                next_part = final_chunks[current_idx]
                if len(current_chunk_text) + len(next_part) <= self.chunk_size:
                    current_chunk_text += next_part + (self.separators[0] if self.separators and self.separators[0] != "" else " ") # Add a separator back if joining
                    current_idx += 1
                else:
                    # Current part would exceed chunk_size, stop here
                    break
            
            current_chunk_text = current_chunk_text.strip()
            if current_chunk_text:
                merged_chunks.append(current_chunk_text)

            # Determine the next starting point for overlap
            if current_idx > current_chunk_start: # If we added anything
                overlap_text = ""
                overlap_idx = current_idx - 1 # Start from the last piece added
                
                # Try to gather `chunk_overlap` characters from the end of the current chunk
                while overlap_idx >= current_chunk_start and len(overlap_text) < self.chunk_overlap:
                    overlap_text = final_chunks[overlap_idx] + (self.separators[0] if self.separators and self.separators[0] != "" else " ") + overlap_text
                    overlap_idx -= 1
                
                # Find the actual start index for the next chunk, accounting for overlap
                # This logic can be tricky. A simpler approach for overlap is to move back
                # a fixed amount of characters from the current chunk's end.
                
                # Simplification: move back 'overlap' characters from the *end of the last part added*
                # and restart from the part that contains that overlap point.
                # This ensures we don't start from an empty string if the last piece was short.
                next_start_char_idx = len(text)
                if current_chunk_text:
                    end_of_current_chunk_in_original = text.find(current_chunk_text, current_chunk_start) # This is approximate
                    if end_of_current_chunk_in_original != -1:
                        next_start_char_idx = end_of_current_chunk_in_original + len(current_chunk_text) - self.chunk_overlap
                        # Ensure next_start_char_idx is not negative
                        if next_start_char_idx < 0:
                            next_start_char_idx = 0
                
                # Find the first full piece that starts after or at next_start_char_idx
                found_next_start = False
                for i, p in enumerate(final_chunks):
                    # Reconstruct string up to this point to find char index. This is slow.
                    # Simpler: just move back based on the list index itself
                    if i >= current_chunk_start: # Make sure we don't go backwards too far
                        estimated_char_pos = sum(len(x) + len(self.separators[0]) if self.separators and self.separators[0] != "" else 1 for x in final_chunks[:i])
                        if estimated_char_pos >= (len(text) - self.chunk_overlap): # Rough estimation
                            current_chunk_start = i
                            found_next_start = True
                            break
                if not found_next_start: # Default: move to next piece
                    current_chunk_start = current_idx # No overlap applied if couldn't find a good overlap point
                else:
                    # Move forward, but ensure we don't start from the very same point if overlap caused it.
                    # This implementation of overlap is more involved than just splitting.
                    # For simplicity, we can do a simple sliding window on the `final_chunks` list.
                    # The current `current_chunk_start` and `current_idx` are relative to `final_chunks`.
                    # Let's adjust for overlap by moving `current_chunk_start` back.
                    overlap_pieces_count = 1 # A heuristic: overlap by at least one piece
                    if current_idx - overlap_pieces_count > current_chunk_start:
                         current_chunk_start = current_idx - overlap_pieces_count
                    else:
                        current_chunk_start = current_idx # No effective overlap if chunks are too small
                    
                    # Or, more robustly, find the character position to restart.
                    # This might require re-thinking how `final_chunks` are handled to make overlap simpler.
                    # A common way is to make `final_chunks` the actual output before overlap, then apply overlap.
                    break_point_char_pos = 0
                    current_total_len = 0
                    for k in range(current_chunk_start, current_idx):
                        current_total_len += len(final_chunks[k])
                    
                    if current_total_len > self.chunk_overlap:
                        overlap_char_pos = current_total_len - self.chunk_overlap
                        
                        # Find which piece `final_chunks[i]` contains or starts after `overlap_char_pos`
                        cumulative_len = 0
                        new_start_idx_for_next_chunk = current_chunk_start
                        for k in range(current_chunk_start, current_idx):
                            cumulative_len += len(final_chunks[k])
                            if cumulative_len >= overlap_char_pos:
                                new_start_idx_for_next_chunk = k
                                break
                        current_chunk_start = new_start_idx_for_next_chunk
                    else:
                         current_chunk_start = current_idx # Fallback: no effective overlap


            else:
                current_chunk_start = current_idx # No overlap if nothing was added

        return merged_chunks


    def chunk(self, document_id: str, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks the input text using recursive character splitting.
        """
        if not text:
            _logger.warning(f"Document {document_id} has no text to chunk.")
            return []

        chunks: List[Dict[str, Any]] = []
        _logger.debug(f"Chunking document {document_id} with RecursiveCharacterChunkingStrategy.")

        # This method is simplified for direct splitting logic without full LangChain complexity
        # and focuses on making the generated chunks from `_split_text_with_separators`
        # and then handling overlap on the resulting list.
        # A more robust implementation would handle overlap during the splitting process itself.

        # Step 1: Split into "base" chunks using separators, ensuring no base chunk exceeds chunk_size
        base_chunks = self._split_text_by_length_and_separator(text, self.separators, self.chunk_size)

        # Step 2: Assemble final chunks with overlap from base chunks
        current_idx = 0
        while current_idx < len(base_chunks):
            start_offset = 0
            if current_idx > 0:
                # Calculate overlap from previous chunk
                previous_chunk_text = base_chunks[current_idx - 1]
                overlap_start = max(0, len(previous_chunk_text) - self.chunk_overlap)
                start_offset = previous_chunk_text[overlap_start:].count(' ') # count words for simple overlap estimation
                
            current_chunk_text = ""
            char_count = 0
            temp_list = []
            
            # Find the starting point within the current base_chunk for overlap
            # Simplified: just iterate from current_idx, but include text from previous if overlap needed
            
            # A more robust overlap logic involves working with character indices:
            # maintain a window over the original text
            
            # Simple approach: build chunks by concatenating base_chunks until chunk_size is met
            # and then move the start index for the next chunk back by a certain amount.
            
            chunk_start_char_idx = 0
            if chunks: # If there are previous chunks, calculate starting point for overlap
                last_chunk = chunks[-1]
                last_chunk_end_char = last_chunk['metadata'].get('end_char_idx', 0)
                chunk_start_char_idx = max(0, last_chunk_end_char - self.chunk_overlap)
            
            current_assembled_text = ""
            current_end_char_idx = -1
            
            # Find the starting position within the `text` for this current chunk
            # This is slow if `text.find` is used repeatedly.
            # A better way is to track start/end character indices from the beginning.
            
            # Let's simplify and make base_chunks the final chunks without explicit overlap on creation.
            # If overlap is strictly needed, it would require generating sub-chunks and then recombining.
            # For this prototype, I will make the overlap apply as a "look-back" during chunk formation.

            # Re-thinking: The recursive character splitter's primary goal is to ensure chunks
            # respect logical boundaries FIRST, and THEN fit within `chunk_size`.
            # LangChain's approach is to get small pieces, then combine them.
            
            # This is a simplified version of RecursiveCharacterTextSplitter without the complex
            # character-level position tracking and merging logic, focusing on producing chunks.
            
            current_pieces: List[str] = []
            for separator in self.separators:
                if separator == "":
                    # Fallback to character-level splitting for any remaining large pieces
                    temp_pieces = []
                    for piece in current_pieces or [text]: # If current_pieces is empty, start with original text
                        if len(piece) > self.chunk_size:
                            for i in range(0, len(piece), self.chunk_size):
                                temp_pieces.append(piece[i:i + self.chunk_size])
                        else:
                            temp_pieces.append(piece)
                    current_pieces = temp_pieces
                    break

                new_pieces: List[str] = []
                # Start with text or the combined pieces from the previous separator level
                parts_to_split = [text] if not current_pieces else current_pieces
                for part in parts_to_split:
                    if len(part) > self.chunk_size:
                        sub_parts = part.split(separator)
                        new_pieces.extend(sub_parts)
                    else:
                        new_pieces.append(part)
                current_pieces = new_pieces
                
                # After splitting by a separator, try to combine small pieces
                # This is where overlap and chunk_size enforcement happens
                combined_chunks: List[str] = []
                current_chunk_start_index = 0
                while current_chunk_start_index < len(current_pieces):
                    current_chunk_text = ""
                    current_chunk_length = 0
                    piece_idx = current_chunk_start_index
                    
                    while piece_idx < len(current_pieces) and \
                          current_chunk_length + len(current_pieces[piece_idx]) + (len(separator) if current_chunk_length > 0 else 0) <= self.chunk_size:
                        
                        if current_chunk_length > 0:
                            current_chunk_text += separator
                            current_chunk_length += len(separator)
                            
                        current_chunk_text += current_pieces[piece_idx]
                        current_chunk_length += len(current_pieces[piece_idx])
                        piece_idx += 1
                    
                    if current_chunk_text:
                        combined_chunks.append(current_chunk_text)
                        # For overlap, move back
                        # This simplified overlap just moves back a certain number of pieces
                        # A more precise method would involve character positions
                        if piece_idx > current_chunk_start_index:
                            overlap_offset = 0
                            overlap_char_target = self.chunk_overlap
                            
                            # Estimate how many previous pieces sum up to the overlap amount
                            temp_overlap_len = 0
                            temp_overlap_idx = piece_idx - 1
                            while temp_overlap_idx >= current_chunk_start_index and temp_overlap_len < overlap_char_target:
                                temp_overlap_len += len(current_pieces[temp_overlap_idx]) + len(separator)
                                temp_overlap_idx -= 1
                            
                            if temp_overlap_idx < current_chunk_start_index: # Means we used all previous pieces up to current_chunk_start_index
                                current_chunk_start_index = piece_idx # No effective overlap, move to next
                            else:
                                current_chunk_start_index = temp_overlap_idx + 1 # Start from the piece that completes overlap
                                
                        else: # No pieces added, just move to next
                             current_chunk_start_index = piece_idx
                    else:
                         current_chunk_start_index += 1 # Avoid infinite loops if a piece is empty or too large


                current_pieces = combined_chunks
                if not current_pieces or all(len(p) <= self.chunk_size for p in current_pieces):
                    break # All chunks are now <= chunk_size or we exhausted separators

            final_text_chunks = current_pieces
            
            for i, chunk_text in enumerate(final_text_chunks):
                chunk_id = hashlib.md5(f"{document_id}-{i}-{chunk_text}".encode()).hexdigest()
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_idx": i,
                    "chunk_strategy": self.__class__.__name__,
                    **document_metadata if document_metadata else {}
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        _logger.info(f"Chunked document {document_id} into {len(chunks)} chunks.")
        return chunks
    
    # Helper for the simplified recursive splitting
    def _split_text_by_length_and_separator(self, text: str, separators: List[str], max_len: int) -> List[str]:
        """
        Recursively splits text using a list of separators, ensuring no resulting piece
        is longer than max_len if possible.
        """
        if not text:
            return []
        if len(text) <= max_len:
            return [text]

        for i, separator in enumerate(separators):
            if separator == "": # Fallback: character-level split
                return [text[j:j + max_len] for j in range(0, len(text), max_len)]

            parts = text.split(separator)
            if len(parts) > 1:
                result = []
                for part in parts:
                    # Recursively process each part with the same or next separator
                    # if the part is still too long.
                    # This is where overlap logic could be integrated more cleanly later.
                    sub_parts = self._split_text_by_length_and_separator(part, separators[i+1:] if i + 1 < len(separators) else [""], max_len)
                    result.extend(sub_parts)
                
                # Now, try to merge parts back together respecting max_len and overlap
                merged = []
                current_chunk = ""
                for part in result:
                    if len(current_chunk) + (len(separator) if current_chunk else 0) + len(part) <= max_len:
                        if current_chunk:
                            current_chunk += separator
                        current_chunk += part
                    else:
                        if current_chunk:
                            merged.append(current_chunk)
                        current_chunk = part
                if current_chunk:
                    merged.append(current_chunk)
                return merged
        
        # If no separators worked and text is still too long (shouldn't happen with "" as last separator)
        return [text]


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Chunks text based on semantic coherence using an embedding model.
    It splits text into smaller units (e.g., sentences), embeds them, and then groups
    them into chunks where semantic similarity is above a certain threshold.
    """
    def __init__(self, embedding_model: EmbeddingModel, threshold: float = 0.75,
                 sentence_splitter: Any = None, buffer_size: int = 1):
        """
        Initializes the semantic chunking strategy.

        Args:
            embedding_model (EmbeddingModel): An instance of the embedding model to use.
            threshold (float): Cosine similarity threshold for grouping sentences into chunks.
                               Sentences with similarity below this threshold mark a chunk boundary.
            sentence_splitter (Any): A function or object that can split text into sentences.
                                     Defaults to a simple regex-based splitter if None.
            buffer_size (int): Number of sentences to add to the beginning/end of a chunk
                               to provide more context around boundaries.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0.")
        if buffer_size < 0:
            raise ValueError("Buffer size cannot be negative.")

        self.embedding_model = embedding_model
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.sentence_splitter = sentence_splitter or self._default_sentence_splitter

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            self._np = np
            self._cosine_similarity = cosine_similarity
        except ImportError as e:
            _logger.error(f"Missing scientific computing libraries for SemanticChunkingStrategy: {e}")
            raise ImportError("numpy and scikit-learn are required for SemanticChunkingStrategy.") from e

    def _default_sentence_splitter(self, text: str) -> List[str]:
        """A simple regex-based sentence splitter."""
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document_id: str, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks the input text semantically.
        """
        if not text:
            _logger.warning(f"Document {document_id} has no text to chunk.")
            return []

        _logger.debug(f"Chunking document {document_id} with SemanticChunkingStrategy.")
        sentences = self.sentence_splitter(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            chunk_id = hashlib.md5(f"{document_id}-0-{sentences[0]}".encode()).hexdigest()
            return [{"text": sentences[0], "metadata": {
                "document_id": document_id, "chunk_id": chunk_id, "chunk_idx": 0,
                "chunk_strategy": self.__class__.__name__, **document_metadata if document_metadata else {}
            }}]

        try:
            embeddings = self.embedding_model.encode(sentences)
            if not isinstance(embeddings, list):
                embeddings = embeddings.tolist()
            embeddings_np = self._np.array(embeddings)
        except Exception as e:
            _logger.error(f"Failed to generate embeddings for document {document_id}: {e}")
            return []

        # Calculate cosine similarity between adjacent sentence embeddings
        similarities = self._cosine_similarity(embeddings_np[:-1], embeddings_np[1:])
        
        # Identify chunk boundaries where similarity drops below the threshold
        chunk_boundaries = [0] # Start of the first chunk
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                chunk_boundaries.append(i + 1)
        chunk_boundaries.append(len(sentences)) # End of the last chunk

        chunks: List[Dict[str, Any]] = []
        for i in range(len(chunk_boundaries) - 1):
            start_idx = chunk_boundaries[i]
            end_idx = chunk_boundaries[i+1]
            
            # Apply buffer for context
            buffered_start_idx = max(0, start_idx - self.buffer_size)
            buffered_end_idx = min(len(sentences), end_idx + self.buffer_size)

            chunk_text = " ".join(sentences[buffered_start_idx:buffered_end_idx])
            
            if chunk_text:
                chunk_id = hashlib.md5(f"{document_id}-{i}-{chunk_text}".encode()).hexdigest()
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_idx": i,
                    "chunk_strategy": self.__class__.__name__,
                    "original_start_sentence_idx": start_idx,
                    "original_end_sentence_idx": end_idx - 1,
                    "buffered_start_sentence_idx": buffered_start_idx,
                    "buffered_end_sentence_idx": buffered_end_idx - 1,
                    **document_metadata if document_metadata else {}
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})
        
        _logger.info(f"Chunked document {document_id} into {len(chunks)} semantic chunks.")
        return chunks


class MarkdownHeaderChunkingStrategy(ChunkingStrategy):
    """
    Chunks Markdown text based on headers (e.g., #, ##, ###).
    Each section defined by a header becomes a chunk. If a section is too large,
    it can be further split using a fallback chunking strategy.
    """
    def __init__(self, max_chunk_size: Optional[int] = 4000,
                 fallback_strategy: Optional[ChunkingStrategy] = None):
        """
        Initializes the Markdown header chunking strategy.

        Args:
            max_chunk_size (Optional[int]): If a header section exceeds this size,
                                             it will be further split by the fallback strategy.
                                             If None, sections are not sub-chunked.
            fallback_strategy (Optional[ChunkingStrategy]): A strategy to use for sub-chunking
                                                            sections that exceed `max_chunk_size`.
                                                            Defaults to RecursiveCharacterChunkingStrategy
                                                            if max_chunk_size is provided.
        """
        self.max_chunk_size = max_chunk_size
        self.fallback_strategy = fallback_strategy
        if self.max_chunk_size is not None and self.fallback_strategy is None:
            self.fallback_strategy = RecursiveCharacterChunkingStrategy(
                chunk_size=self.max_chunk_size, chunk_overlap=100, separators=["\n\n", "\n"]
            )
        
        if self.max_chunk_size is not None and self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer or None.")

    def chunk(self, document_id: str, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks the input Markdown text based on headers.
        """
        if not text:
            _logger.warning(f"Document {document_id} has no text to chunk.")
            return []

        _logger.debug(f"Chunking document {document_id} with MarkdownHeaderChunkingStrategy.")
        chunks: List[Dict[str, Any]] = []
        
        # Regex to find all headers (h1 to h6)
        # It captures the header level and the header text
        # re.DOTALL ensures '.' matches newlines
        header_pattern = re.compile(r"^(#+)\s*(.*)$", re.MULTILINE)
        
        # Find all header matches and their start/end indices
        matches = list(header_pattern.finditer(text))

        current_pos = 0
        chunk_idx = 0

        # Handle content before the first header (if any)
        if matches and matches[0].start() > 0:
            pre_header_text = text[0:matches[0].start()].strip()
            if pre_header_text:
                if self.max_chunk_size and len(pre_header_text) > self.max_chunk_size and self.fallback_strategy:
                    _logger.debug(f"Pre-header text for {document_id} exceeds max_chunk_size, applying fallback.")
                    fallback_chunks = self.fallback_strategy.chunk(document_id, pre_header_text, {
                        "section_title": "Document Start",
                        "section_level": 0,
                        **document_metadata if document_metadata else {}
                    })
                    for fc in fallback_chunks:
                        fc['metadata']['chunk_idx'] = chunk_idx
                        fc['metadata']['chunk_strategy_base'] = self.__class__.__name__
                        chunks.append(fc)
                        chunk_idx += 1
                else:
                    chunk_id = hashlib.md5(f"{document_id}-{chunk_idx}-{pre_header_text}".encode()).hexdigest()
                    chunks.append({
                        "text": pre_header_text,
                        "metadata": {
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "chunk_idx": chunk_idx,
                            "chunk_strategy": self.__class__.__name__,
                            "section_title": "Document Start",
                            "section_level": 0,
                            **document_metadata if document_metadata else {}
                        }
                    })
                    chunk_idx += 1
            current_pos = matches[0].start()

        # Iterate through headers and extract content
        for i, match in enumerate(matches):
            header_level = len(match.group(1))
            header_title = match.group(2).strip()
            header_start = match.start()
            header_end = match.end()

            # Content between this header and the next one (or end of document)
            next_header_start = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[header_end:next_header_start].strip()

            full_section_text = f"{match.group(0).strip()}\n\n{section_content}" if section_content else match.group(0).strip()
            
            if not full_section_text.strip():
                continue # Skip empty sections

            if self.max_chunk_size and len(full_section_text) > self.max_chunk_size and self.fallback_strategy:
                _logger.debug(f"Section '{header_title}' for {document_id} exceeds max_chunk_size, applying fallback.")
                fallback_chunks = self.fallback_strategy.chunk(document_id, full_section_text, {
                    "section_title": header_title,
                    "section_level": header_level,
                    **document_metadata if document_metadata else {}
                })
                for fc in fallback_chunks:
                    fc['metadata']['chunk_idx'] = chunk_idx
                    fc['metadata']['chunk_strategy_base'] = self.__class__.__name__
                    chunks.append(fc)
                    chunk_idx += 1
            else:
                chunk_id = hashlib.md5(f"{document_id}-{chunk_idx}-{header_title}-{full_section_text}".encode()).hexdigest()
                chunks.append({
                    "text": full_section_text,
                    "metadata": {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "chunk_idx": chunk_idx,
                        "chunk_strategy": self.__class__.__name__,
                        "section_title": header_title,
                        "section_level": header_level,
                        **document_metadata if document_metadata else {}
                    }
                })
                chunk_idx += 1
            
            current_pos = next_header_start
        
        # Handle any remaining content after the last header
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                if self.max_chunk_size and len(remaining_text) > self.max_chunk_size and self.fallback_strategy:
                    _logger.debug(f"Remaining text for {document_id} exceeds max_chunk_size, applying fallback.")
                    fallback_chunks = self.fallback_strategy.chunk(document_id, remaining_text, {
                        "section_title": "Document End",
                        "section_level": 0,
                        **document_metadata if document_metadata else {}
                    })
                    for fc in fallback_chunks:
                        fc['metadata']['chunk_idx'] = chunk_idx
                        fc['metadata']['chunk_strategy_base'] = self.__class__.__name__
                        chunks.append(fc)
                        chunk_idx += 1
                else:
                    chunk_id = hashlib.md5(f"{document_id}-{chunk_idx}-{remaining_text}".encode()).hexdigest()
                    chunks.append({
                        "text": remaining_text,
                        "metadata": {
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "chunk_idx": chunk_idx,
                            "chunk_strategy": self.__class__.__name__,
                            "section_title": "Document End",
                            "section_level": 0,
                            **document_metadata if document_metadata else {}
                        }
                    })
                    chunk_idx += 1

        _logger.info(f"Chunked document {document_id} into {len(chunks)} Markdown header-based chunks.")
        return chunks

class ChunkingStrategyFactory:
    """
    A factory class to create instances of different chunking strategies.
    """
    @staticmethod
    def get_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
        """
        Returns an instance of the specified chunking strategy.

        Args:
            strategy_name (str): The name of the chunking strategy (e.g., "recursive_character", "semantic", "markdown_header").
            **kwargs: Keyword arguments specific to the chosen strategy.

        Returns:
            ChunkingStrategy: An instance of the requested chunking strategy.

        Raises:
            ValueError: If an unknown strategy name is provided.
        """
        strategy_name = strategy_name.lower()
        if strategy_name == "recursive_character":
            return RecursiveCharacterChunkingStrategy(**kwargs)
        elif strategy_name == "semantic":
            # Ensure embedding_model is provided for semantic chunking
            if 'embedding_model' not in kwargs:
                _logger.error("EmbeddingModel instance is required for 'semantic' chunking strategy.")
                raise ValueError("EmbeddingModel instance is required for 'semantic' chunking strategy.")
            return SemanticChunkingStrategy(**kwargs)
        elif strategy_name == "markdown_header":
            return MarkdownHeaderChunkingStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")

if __name__ == '__main__':
    # --- Example Usage ---
    _logger.info("Running chunking strategy examples...")

    document_text_1 = """
# Introduction to AI

Artificial intelligence (AI) is a rapidly evolving field that aims to create machines capable of intelligent behavior. It encompasses various sub-fields, including machine learning, deep learning, natural language processing, and computer vision.

## Historical Overview
The concept of AI dates back to ancient times, but the modern field was founded in 1956 at a workshop at Dartmouth College. Early AI research focused on problem-solving and symbolic methods.

### The AI Winter
Periods of reduced funding and interest, known as "AI winters," occurred in the 1970s and 1980s due to unmet expectations.

## Modern AI
Recent advancements in computational power, data availability, and new algorithms have led to a resurgence of AI, particularly in machine learning. Large Language Models (LLMs) are a prime example of this new wave.

Here's a simple list:
- Item 1
- Item 2

This is a concluding paragraph about the future of AI. It's expected to revolutionize many industries, from healthcare to finance. The ethical implications are also a major concern.

"""

    document_text_2 = """
A very long paragraph about the history of the internet. The internet, a global network of interconnected computer networks, has revolutionized communication and information exchange. Its origins trace back to the 1960s with the Advanced Research Projects Agency Network (ARPANET) developed by the U.S. Department of Defense. This early network was designed to facilitate communication between researchers and to be resilient to potential disruptions. Over the decades, ARPANET evolved, incorporating new protocols and expanding its reach. In the 1980s, the Domain Name System (DNS) was introduced, making addresses more user-friendly. The World Wide Web, invented by Tim Berners-Lee in the late 1980s and early 1990s, dramatically transformed the internet's usability by introducing hypertext and a graphical interface. This period saw the explosion of commercial and public interest in the internet, leading to its rapid adoption worldwide. The 2000s ushered in the era of social media, mobile internet, and cloud computing, further integrating the internet into daily life. Today, the internet is an indispensable tool for education, business, entertainment, and personal communication, constantly evolving with new technologies like IoT and 5G. The continuous growth presents challenges related to privacy, security, and digital divide.
""" * 3 # Make it extra long

    document_text_3_short = "Short text with only one sentence. This is another sentence. And a third."

    doc_id_1 = "doc-ai-history"
    doc_id_2 = "doc-internet-history"
    doc_id_3 = "doc-short-semantic"

    # --- Recursive Character Chunking ---
    _logger.info("\n--- Recursive Character Chunking Example ---")
    rc_strategy = ChunkingStrategyFactory.get_strategy(
        "recursive_character", chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". "]
    )
    chunks_rc = rc_strategy.chunk(doc_id_1, document_text_1, {"source": "Wikipedia", "type": "markdown"})
    for i, chunk in enumerate(chunks_rc):
        _logger.info(f"Chunk RC {i}: (len={len(chunk['text'])}) {chunk['text'][:150]}...")
        if i >= 2: break # Limit output for brevity

    chunks_rc_long = rc_strategy.chunk(doc_id_2, document_text_2, {"source": "Custom Article"})
    _logger.info(f"RecursiveCharacterChunking for long doc resulted in {len(chunks_rc_long)} chunks.")
    for i, chunk in enumerate(chunks_rc_long):
        _logger.info(f"Chunk RC Long {i}: (len={len(chunk['text'])}) {chunk['text'][:150]}...")
        if i >= 2: break


    # --- Semantic Chunking ---
    _logger.info("\n--- Semantic Chunking Example ---")
    embedding_model_instance = EmbeddingModel() # Using the mock model if actual not found
    semantic_strategy = ChunkingStrategyFactory.get_strategy(
        "semantic", embedding_model=embedding_model_instance, threshold=0.7, buffer_size=1
    )
    chunks_semantic = semantic_strategy.chunk(doc_id_3, document_text_3_short, {"source": "Test", "topic": "sentences"})
    for i, chunk in enumerate(chunks_semantic):
        _logger.info(f"Chunk Semantic {i}: (len={len(chunk['text'])}) {chunk['text']}")

    chunks_semantic_ai = semantic_strategy.chunk(doc_id_1, document_text_1, {"source": "Wikipedia"})
    _logger.info(f"SemanticChunking for doc 1 resulted in {len(chunks_semantic_ai)} chunks.")
    for i, chunk in enumerate(chunks_semantic_ai):
        _logger.info(f"Chunk Semantic AI {i}: (len={len(chunk['text'])}) {chunk['text'][:150]}...")
        if i >= 2: break


    # --- Markdown Header Chunking ---
    _logger.info("\n--- Markdown Header Chunking Example ---")
    # With fallback for large sections
    md_strategy = ChunkingStrategyFactory.get_strategy(
        "markdown_header", max_chunk_size=200,
        fallback_strategy=RecursiveCharacterChunkingStrategy(chunk_size=200, chunk_overlap=20)
    )
    chunks_md = md_strategy.chunk(doc_id_1, document_text_1, {"source": "Wikipedia", "format": "markdown"})
    for i, chunk in enumerate(chunks_md):
        _logger.info(f"Chunk MD {i}: (len={len(chunk['text'])}) Title='{chunk['metadata'].get('section_title')}' Level={chunk['metadata'].get('section_level')} {chunk['text'][:150]}...")
        if i >= 5: break

    # Without fallback (larger chunks allowed)
    md_no_fallback_strategy = ChunkingStrategyFactory.get_strategy(
        "markdown_header", max_chunk_size=None
    )
    chunks_md_no_fallback = md_no_fallback_strategy.chunk(doc_id_1, document_text_1, {"source": "Wikipedia", "format": "markdown"})
    _logger.info(f"MarkdownHeaderChunking (no fallback) for doc 1 resulted in {len(chunks_md_no_fallback)} chunks.")
    for i, chunk in enumerate(chunks_md_no_fallback):
        _logger.info(f"Chunk MD No Fallback {i}: (len={len(chunk['text'])}) Title='{chunk['metadata'].get('section_title')}' Level={chunk['metadata'].get('section_level')} {chunk['text'][:150]}...")
        if i >= 2: break

    _logger.info("\nChunking strategy examples finished.")