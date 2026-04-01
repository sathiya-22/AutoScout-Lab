```python
import uuid
import re
from typing import List, Dict, Any, Optional

class TextParser:
    """
    A parser for standard text content, responsible for chunking and basic pre-processing.
    It implements a recursive character splitting strategy to maintain semantic coherence
    while adhering to specified chunk size and overlap. This approach attempts to split
    text by larger semantic units (like paragraphs) first, then progressively by smaller units
    (like sentences or words) if chunks remain too large, finally resorting to character-level splits.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        strip_whitespace: bool = True
    ):
        """
        Initializes the TextParser with chunking parameters.

        Args:
            chunk_size (int): The maximum size of each text chunk. Must be a positive integer.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.
                                 Must be less than `chunk_size` and non-negative.
            separators (Optional[List[str]]): A list of separators to use for splitting text,
                                                ordered from most semantic (e.g., paragraph breaks)
                                                to least (e.g., words, characters). The empty string `""`
                                                can be included as a last-resort character-level splitter.
                                                If None, defaults to ["\\n\\n", "\\n", " ", ""].
            strip_whitespace (bool): If True, performs robust whitespace normalization on the
                                     entire text content before chunking, and on individual chunks.
                                     This means collapsing multiple newlines/spaces, and stripping
                                     leading/trailing whitespace.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default separators prioritize larger semantic units first.
        # An empty string separator `""` is used for character-level splitting as a last resort.
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.strip_whitespace = strip_whitespace

    def _get_text_length(self, text: str) -> int:
        """Helper to get text length."""
        return len(text)

    def _split_text_recursively(self, text: str, current_offset: int) -> List[Dict[str, Any]]:
        """
        Recursively splits text using the configured separators to produce "raw pieces".
        These pieces are either small enough to fit within `chunk_size` or cannot be split
        further by the current set of separators (e.g., a very long word without spaces).
        Offsets track the position in the *processed* (normalized) document.

        Args:
            text (str): The text segment to split.
            current_offset (int): The starting character offset of this segment in the
                                  overall processed document.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a `text` piece
                                  and its `start_offset`.
        """
        if not text:
            return []

        # `segments_to_process` holds parts of the text that might still be too large
        # and need to be split by the next, finer-grained separator.
        # Each item is `{"text": string, "start_offset": int}`
        segments_to_process = [{"text": text, "start_offset": current_offset}]
        final_pieces: List[Dict[str, Any]] = []

        for separator in self.separators:
            next_iteration_segments = []
            for segment in segments_to_process:
                current_segment_text = segment["text"]
                current_segment_offset = segment["start_offset"]

                if not current_segment_text:
                    continue

                # If the segment is already small enough, or this separator is empty (char-level split),
                # we add it to final_pieces as it's either granular enough or can't be split semantically.
                if self._get_text_length(current_segment_text) <= self.chunk_size or not separator:
                    final_pieces.append({"text": current_segment_text, "start_offset": current_segment_offset})
                    continue

                # Split the current segment by the separator
                parts = current_segment_text.split(separator)
                
                temp_offset = current_segment_offset
                for part_index, part in enumerate(parts):
                    if part: # Only consider non-empty parts
                        # If this part is too long, it needs to be processed by the next separator level.
                        if self._get_text_length(part) > self.chunk_size:
                            next_iteration_segments.append({"text": part, "start_offset": temp_offset})
                        else:
                            # It's small enough, add it to final pieces
                            final_pieces.append({"text": part, "start_offset": temp_offset})
                    
                    # Update offset for the next part.
                    # If not the last part, account for the separator length that was removed by `split()`.
                    if part_index < len(parts) - 1:
                        temp_offset += self._get_text_length(part) + self._get_text_length(separator)
                    else:
                        temp_offset += self._get_text_length(part) # Only the part length

            segments_to_process = next_iteration_segments # Move to the next set of segments for the next separator

        # Any remaining segments in `segments_to_process` at the end of the loop
        # (these are segments that were too large even after trying all semantic separators)
        # must now be added as final_pieces. These will be broken into fixed-size chunks
        # by `_merge_pieces_with_overlap`.
        for segment in segments_to_process:
            if segment["text"]:
                final_pieces.append({"text": segment["text"], "start_offset": segment["start_offset"]})
        
        return final_pieces

    def _merge_pieces_with_overlap(self, pieces: List[Dict[str, Any]]) -> List[tuple[str, int, int]]:
        """
        Merges a list of raw text pieces (with their offsets) into final chunks,
        applying `chunk_size` and `chunk_overlap` constraints.

        Args:
            pieces (List[Dict[str, Any]]): A list of dictionaries, each containing a `text` piece
                                           and its `start_offset`.

        Returns:
            List[tuple[str, int, int]]: A list of tuples, where each tuple is
                                        (chunk_content, start_char_offset, end_char_offset).
        """
        final_chunks_with_offsets: List[tuple[str, int, int]] = []
        
        current_combined_text = ""
        # Offset of the first character in `current_combined_text` relative to the processed document.
        current_combined_start_offset = -1 

        for piece in pieces:
            piece_text = piece["text"]
            piece_start_offset = piece["start_offset"]

            if not piece_text:
                continue

            # Case 1: No current combined chunk, or adding the new piece keeps it within `chunk_size`.
            if not current_combined_text or \
               self._get_text_length(current_combined_text + piece_text) <= self.chunk_size:
                
                if not current_combined_text: # This is the first piece for this combined chunk
                    current_combined_start_offset = piece_start_offset
                current_combined_text += piece_text
            else:
                # Case 2: Adding the new piece makes `current_combined_text` too large.
                # 1. Emit the current combined chunk.
                final_chunks_with_offsets.append(
                    (current_combined_text, current_combined_start_offset,
                     current_combined_start_offset + self._get_text_length(current_combined_text))
                )

                # 2. Start a new combined chunk with overlap.
                # The overlap text is taken from the end of the *just-emitted* chunk.
                overlap_text = current_combined_text[max(0, self._get_text_length(current_combined_text) - self.chunk_overlap):]
                overlap_start_offset = current_combined_start_offset + max(0, self._get_text_length(current_combined_text) - self.chunk_overlap)
                
                # The new `current_combined_text` starts with the overlap, followed by the current piece.
                current_combined_text = overlap_text + piece_text
                current_combined_start_offset = overlap_start_offset

                # Edge case: If `overlap_text` was empty (e.g., previous chunk was shorter than `chunk_overlap`,
                # or `chunk_overlap` is 0), then the new chunk truly starts at the `piece_start_offset`.
                if not overlap_text:
                    current_combined_start_offset = piece_start_offset

                # If even the newly formed `current_combined_text` (overlap + new piece) is still too large
                # (e.g., if `piece_text` itself was extremely long, like a single very long word),
                # then we must break it down into fixed-size sub-chunks.
                while self._get_text_length(current_combined_text) > self.chunk_size:
                    sub_chunk_content = current_combined_text[:self.chunk_size]
                    final_chunks_with_offsets.append(
                        (sub_chunk_content, current_combined_start_offset,
                         current_combined_start_offset + self._get_text_length(sub_chunk_content))
                    )
                    
                    # Advance for the next sub-chunk, applying overlap within this very long segment.
                    current_combined_start_offset += (self.chunk_size - self.chunk_overlap)
                    current_combined_text = current_combined_text[self.chunk_size - self.chunk_overlap:]
                    
                    if not current_combined_text: # Prevent potential infinite loop if current_combined_text becomes empty
                        break

        # Emit the very last combined chunk if anything remains after the loop.
        if current_combined_text:
            final_chunks_with_offsets.append(
                (current_combined_text, current_combined_start_offset,
                 current_combined_start_offset + self._get_text_length(current_combined_text))
            )
        
        return final_chunks_with_offsets

    def parse(self, document_id: str, text_content: str) -> List[Dict[str, Any]]:
        """
        Parses the given raw text content into a list of structured text chunks.

        Args:
            document_id (str): A unique identifier for the source document.
                               Must be a non-empty string.
            text_content (str): The raw text content to parse. Must be a string.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents
                                  a chunk with its content and metadata.
                                  Each chunk includes:
                                  - "chunk_id": A unique identifier for the chunk.
                                  - "document_id": The ID of the source document.
                                  - "content": The text content of the chunk.
                                  - "start_char": The starting character offset in the processed document.
                                  - "end_char": The ending character offset in the processed document.
                                  - "chunk_index": The sequential index of the chunk within the document.
                                  - "type": "text_chunk".
        """
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string.")
        if not isinstance(text_content, str):
            raise ValueError("text_content must be a string.")

        if not text_content.strip():
            return [] # Return empty list for empty or whitespace-only content

        # Apply robust whitespace normalization to the entire text content upfront.
        # The `start_char` and `end_char` offsets in the output chunks will refer
        # to positions within this `processed_text`. This is a common and practical
        # approach for RAG, where the LLM consumes the normalized content.
        
        processed_text = text_content
        if self.strip_whitespace:
            # Standardize various newline characters to '\n'
            processed_text = processed_text.replace('\r\n', '\n').replace('\r', '\n')
            # Collapse multiple newlines to at most two, preserving paragraph breaks
            processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
            # Strip leading/trailing whitespace from each line and then join them
            processed_text = '\n'.join([line.strip() for line in processed_text.split('\n')])
            # Collapse multiple spaces into a single space
            processed_text = re.sub(r' +', ' ', processed_text)
            # Final overall strip of leading/trailing whitespace
            processed_text = processed_text.strip()

        if not processed_text:
            return []

        # Step 1: Break the `processed_text` into raw, semantically coherent pieces.
        # These pieces are either already smaller than `chunk_size` or couldn't be
        # split further by the configured separators.
        raw_pieces = self._split_text_recursively(processed_text, 0)
        
        # Step 2: Merge these raw pieces into final chunks, applying the `chunk_size`
        # and `chunk_overlap` constraints.
        chunks_with_offsets = self._merge_pieces_with_overlap(raw_pieces)

        parsed_chunks: List[Dict[str, Any]] = []
        for i, (chunk_text, start_offset_in_processed, end_offset_in_processed) in enumerate(chunks_with_offsets):
            # Final strip on the chunk content, just in case (e.g., if merge introduced spaces, unlikely with current logic).
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            if not chunk_text: # Filter out any empty chunks resulting from stripping
                continue

            parsed_chunks.append({
                "chunk_id": f"{document_id}-{i}-{uuid.uuid4().hex[:8]}", # Unique ID for each chunk
                "document_id": document_id,
                "content": chunk_text,
                "start_char": start_offset_in_processed,
                "end_char": end_offset_in_processed,
                "chunk_index": i,
                "type": "text_chunk",
            })
        return parsed_chunks
```