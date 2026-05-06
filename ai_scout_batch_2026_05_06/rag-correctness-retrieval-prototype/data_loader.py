import os
import json
from pathlib import Path
from typing import List, Dict, Union, Callable

class DataLoader:
    """
    Handles ingestion, cleaning, and chunking of raw documents from `data/raw_docs/`
    into manageable units stored in `data/processed_chunks/`.
    """

    def __init__(self,
                 raw_docs_dir: Union[str, Path] = "data/raw_docs",
                 processed_chunks_dir: Union[str, Path] = "data/processed_chunks",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_length: int = 50,
                 text_cleaner: Callable[[str], str] = None):
        """
        Initializes the DataLoader.

        Args:
            raw_docs_dir (Union[str, Path]): Path to the directory containing raw documents.
            processed_chunks_dir (Union[str, Path]): Path to the directory where processed
                                                     chunks will be stored.
            chunk_size (int): The maximum number of characters in a chunk.
            chunk_overlap (int): The number of characters to overlap between consecutive chunks.
            min_chunk_length (int): Minimum length a chunk must have to be considered valid.
            text_cleaner (Callable[[str], str], optional): A custom function to clean text.
                                                            If None, a default cleaner is used.
        """
        self.raw_docs_dir = Path(raw_docs_dir)
        self.processed_chunks_dir = Path(processed_chunks_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.text_cleaner = text_cleaner if text_cleaner else self._default_clean_text

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensures that the raw documents and processed chunks directories exist."""
        self.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_chunks_dir.mkdir(parents=True, exist_ok=True)

    def _default_clean_text(self, text: str) -> str:
        """
        Applies a default cleaning to the text:
        - Replaces multiple whitespace characters with a single space.
        - Strips leading/trailing whitespace.
        """
        text = ' '.join(text.split())
        return text.strip()

    def _load_document_content(self, file_path: Path) -> str:
        """
        Loads the content of a single document file.
        Currently supports .txt and .md.
        """
        try:
            if file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix.lower() == '.pdf':
                # Placeholder for PDF loading. In a real scenario, integrate a library like PyPDF2 or pypdf.
                print(f"Warning: PDF file '{file_path.name}' detected. PDF parsing is not fully implemented "
                      "in this prototype. Content will be skipped or handled minimally.")
                return "" # Return empty string for now
            else:
                print(f"Warning: Unsupported file type for '{file_path.name}'. Skipping.")
                return ""
        except Exception as e:
            print(f"Error loading document '{file_path.name}': {e}")
            return ""

    def _recursive_character_splitter(self, text: str, source_metadata: Dict) -> List[Dict]:
        """
        Splits text into chunks using a recursive character splitting strategy.
        This attempts to split on common delimiters (paragraphs, sentences, words)
        to maintain semantic coherence.
        """
        if not text:
            return []

        # Prioritize splitting by paragraphs, then sentences, then words
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        chunks = []
        current_chunks = [text]

        for separator in separators:
            next_chunks = []
            for chunk in current_chunks:
                if len(chunk) > self.chunk_size:
                    sub_chunks = chunk.split(separator)
                    temp_chunk = ""
                    for sub_chunk in sub_chunks:
                        if len(temp_chunk) + len(separator) + len(sub_chunk) <= self.chunk_size:
                            temp_chunk += (separator if temp_chunk else "") + sub_chunk
                        else:
                            if temp_chunk: # Add the accumulated chunk if not empty
                                next_chunks.append(temp_chunk)
                            temp_chunk = sub_chunk # Start a new chunk
                    if temp_chunk: # Add any remaining part
                        next_chunks.append(temp_chunk)
                else:
                    next_chunks.append(chunk)
            current_chunks = next_chunks
            # If all chunks are small enough, we can stop
            if all(len(c) <= self.chunk_size for c in current_chunks):
                break

        final_chunks_with_overlap = []
        for i, chunk_text in enumerate(current_chunks):
            if not chunk_text or len(chunk_text) < self.min_chunk_length:
                continue

            chunk_metadata = {
                "chunk_id": f"{source_metadata.get('source_file_id', 'unknown')}_{i}",
                "chunk_index": i,
                "text_length": len(chunk_text),
                **source_metadata
            }
            final_chunks_with_overlap.append({"text": chunk_text, "metadata": chunk_metadata})

            # Add overlap if possible (basic implementation, can be more sophisticated)
            if i < len(current_chunks) - 1 and self.chunk_overlap > 0:
                next_chunk_text = current_chunks[i+1]
                overlap_text = chunk_text[-self.chunk_overlap:]
                if next_chunk_text.startswith(overlap_text):
                    # Overlap is already implicit or naturally occurring
                    pass
                else:
                    # Append an explicit overlap if needed, but only if it fits within the next chunk
                    # For simplicity, we mostly rely on the splitting strategy itself
                    # to create some natural overlap when splitting on common separators.
                    # A more explicit overlap would involve re-slicing and re-combining,
                    # which is complex for a generic recursive char splitter.
                    pass

        return final_chunks_with_overlap


    def _save_chunk(self, chunk: Dict, original_filename_stem: str, chunk_index: int) -> Path:
        """
        Saves a single chunk dictionary to a JSON file.
        """
        output_filename = f"{original_filename_stem}_chunk_{chunk_index}.json"
        output_path = self.processed_chunks_dir / output_filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            return output_path
        except Exception as e:
            print(f"Error saving chunk {output_filename}: {e}")
            return None

    def ingest_documents(self, force_reprocess: bool = False) -> List[Dict]:
        """
        Orchestrates the loading, cleaning, chunking, and saving of all raw documents.

        Args:
            force_reprocess (bool): If True, all documents will be reprocessed even if
                                    their processed chunks already exist.

        Returns:
            List[Dict]: A list of all processed chunks, each as a dictionary.
        """
        processed_documents_count = 0
        all_processed_chunks = []
        document_files = list(self.raw_docs_dir.glob('*.*'))

        if not document_files:
            print(f"No documents found in '{self.raw_docs_dir}'.")
            return []

        print(f"Starting document ingestion from '{self.raw_docs_dir}'...")

        for file_path in document_files:
            original_filename_stem = file_path.stem
            
            # Check if processed chunks for this file already exist
            existing_chunks = list(self.processed_chunks_dir.glob(f"{original_filename_stem}_chunk_*.json"))
            if existing_chunks and not force_reprocess:
                print(f"Skipping '{file_path.name}': Processed chunks already exist. Use 'force_reprocess=True' to reprocess.")
                for chunk_file in existing_chunks:
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            all_processed_chunks.append(json.load(f))
                    except json.JSONDecodeError as e:
                        print(f"Error loading existing chunk '{chunk_file.name}': {e}. This chunk will be skipped.")
                continue

            if force_reprocess and existing_chunks:
                print(f"Reprocessing '{file_path.name}'. Deleting existing chunks...")
                for chunk_file in existing_chunks:
                    try:
                        chunk_file.unlink() # Delete the old chunk file
                    except OSError as e:
                        print(f"Error deleting old chunk file '{chunk_file.name}': {e}")

            print(f"Processing document: '{file_path.name}'...")
            raw_content = self._load_document_content(file_path)
            if not raw_content:
                continue

            cleaned_content = self.text_cleaner(raw_content)
            if not cleaned_content:
                print(f"Warning: Document '{file_path.name}' resulted in empty content after cleaning. Skipping.")
                continue

            source_metadata = {
                "source_file": file_path.name,
                "source_file_id": original_filename_stem,
                "path": str(file_path.absolute()),
                "last_modified": file_path.stat().st_mtime
            }
            chunks = self._recursive_character_splitter(cleaned_content, source_metadata)

            if not chunks:
                print(f"Warning: No valid chunks generated for document '{file_path.name}'.")
                continue

            for i, chunk in enumerate(chunks):
                saved_path = self._save_chunk(chunk, original_filename_stem, i)
                if saved_path:
                    all_processed_chunks.append(chunk)

            processed_documents_count += 1
            print(f"Successfully processed and chunked '{file_path.name}' into {len(chunks)} chunks.")

        print(f"Finished ingestion. Total documents processed: {processed_documents_count}.")
        print(f"Total chunks available: {len(all_processed_chunks)}.")
        return all_processed_chunks

    def load_processed_chunks(self) -> List[Dict]:
        """
        Loads all previously processed chunks from the `processed_chunks_dir`.

        Returns:
            List[Dict]: A list of all loaded chunks, each as a dictionary.
        """
        loaded_chunks = []
        print(f"Loading processed chunks from '{self.processed_chunks_dir}'...")
        for chunk_file in self.processed_chunks_dir.glob('*.json'):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                    loaded_chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from '{chunk_file.name}': {e}. Skipping this file.")
            except Exception as e:
                print(f"Error loading chunk file '{chunk_file.name}': {e}. Skipping this file.")
        print(f"Loaded {len(loaded_chunks)} processed chunks.")
        return loaded_chunks

# Example Usage (for testing and demonstration purposes, not part of the class)
if __name__ == '__main__':
    # Create dummy raw documents for testing
    RAW_DOCS_DIR = Path("data/raw_docs")
    PROCESSED_CHUNKS_DIR = Path("data/processed_chunks")

    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up previous test data
    for f in RAW_DOCS_DIR.glob('*'): f.unlink()
    for f in PROCESSED_CHUNKS_DIR.glob('*'): f.unlink()

    doc1_content = """
    The quick brown fox jumps over the lazy dog. This is a very common pangram used to display
    all letters of the English alphabet. It has been used for typewriters and computer fonts
    since the 19th century.

    In linguistics, a pangram (from Greek pan gramma, "every letter") or holalphabetic
    sentence is a sentence, phrase, or word sequence containing every letter of the alphabet at least once.
    The most famous example in English is "The quick brown fox jumps over the lazy dog".
    Another example is "JFK's a quaint game zygote, but is it good?".

    Pangrams are commonly used to demonstrate typefaces, test equipment, and develop typing skills.
    They are especially relevant in the field of typography and font design.
    """
    (RAW_DOCS_DIR / "doc1.txt").write_text(doc1_content, encoding='utf-8')

    doc2_content = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the
    natural intelligence displayed by humans and animals. Leading AI textbooks define the field
    as the study of "intelligent agents": any device that perceives its environment and takes
    actions that maximize its chance of successfully achieving its goals.

    Colloquially, the term "artificial intelligence" is often used to describe machines that mimic
    "cognitive" functions that humans associate with other human minds, such as "learning" and "problem-solving".
    However, many problems considered AI today have been successfully solved by techniques that do not involve
    anything resembling human cognition. For instance, sophisticated optimization algorithms or statistical models.

    The history of AI research has been marked by several "AI winters," periods of reduced funding
    and interest following overly optimistic predictions and failures to deliver. However,
    recent advancements in machine learning, particularly deep learning, have led to a resurgence
    of interest and significant progress in areas like natural language processing and computer vision.
    """
    (RAW_DOCS_DIR / "doc2.md").write_text(doc2_content, encoding='utf-8')

    # Test with a small document to ensure it's not discarded if it's very short
    doc3_content = "Short text example."
    (RAW_DOCS_DIR / "doc3.txt").write_text(doc3_content, encoding='utf-8')

    # Initialize DataLoader
    data_loader = DataLoader(
        raw_docs_dir=RAW_DOCS_DIR,
        processed_chunks_dir=PROCESSED_CHUNKS_DIR,
        chunk_size=200,  # Smaller chunk size for demonstration
        chunk_overlap=50
    )

    # Ingest documents
    print("\n--- Ingesting Documents ---")
    ingested_chunks = data_loader.ingest_documents(force_reprocess=True)
    print(f"\nTotal ingested chunks: {len(ingested_chunks)}")

    if ingested_chunks:
        print("\n--- Example of an ingested chunk ---")
        print(json.dumps(ingested_chunks[0], indent=2))

    # Test loading processed chunks
    print("\n--- Loading Processed Chunks ---")
    loaded_chunks = data_loader.load_processed_chunks()
    print(f"Total loaded chunks: {len(loaded_chunks)}")

    if loaded_chunks:
        print("\n--- Example of a loaded chunk (should be identical) ---")
        print(json.dumps(loaded_chunks[0], indent=2))

    # Test re-ingestion without force_reprocess
    print("\n--- Ingesting again without force_reprocess (should skip) ---")
    data_loader.ingest_documents()

    # Create an empty directory for testing
    empty_raw_docs_dir = Path("data/empty_raw_docs")
    empty_raw_docs_dir.mkdir(parents=True, exist_ok=True)
    empty_loader = DataLoader(raw_docs_dir=empty_raw_docs_dir)
    print("\n--- Ingesting from empty directory ---")
    empty_loader.ingest_documents()
    empty_raw_docs_dir.rmdir() # Clean up

    # Clean up test directories (optional)
    # import shutil
    # shutil.rmtree(RAW_DOCS_DIR)
    # shutil.rmtree(PROCESSED_CHUNKS_DIR)
    # print("\nCleaned up test directories.")