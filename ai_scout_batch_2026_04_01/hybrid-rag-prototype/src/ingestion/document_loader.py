import os
import mimetypes
from typing import Dict, Any, NamedTuple, Optional

# Attempt to import PDF reader library
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    # In a production system, this might log to a proper logging system
    # print("Warning: 'pypdf' library not found. PDF document loading will be disabled.")


class DocumentData(NamedTuple):
    """
    A simple data structure to hold the loaded document's content and metadata.
    """
    content: str
    metadata: Dict[str, Any]


class DocumentLoader:
    """
    Handles loading various document formats into a standardized DocumentData object.
    It primarily extracts raw text content and relevant metadata.
    Specific parsing (e.g., AST for code, tabular for CSV) is handled
    by subsequent parser components.
    """
    def __init__(self):
        # Mappings for file extensions to internal loading methods
        self.loader_map = {
            # Generic text and markdown files
            '.txt': self._load_text_file,
            '.md': self._load_text_file,
            '.json': self._load_text_file,
            '.xml': self._load_text_file,
            '.html': self._load_text_file,

            # Code files (treated as plain text initially; AST parsing is separate)
            '.py': self._load_text_file,
            '.java': self._load_text_file,
            '.c': self._load_text_file,
            '.cpp': self._load_text_file,
            '.h': self._load_text_file,
            '.hpp': self._load_text_file,
            '.js': self._load_text_file,
            '.ts': self._load_text_file,
            '.go': self._load_text_file,
            '.rs': self._load_text_file,
            '.rb': self._load_text_file,
            '.php': self._load_text_file,
            '.sh': self._load_text_file,
            '.ps1': self._load_text_file,
            '.bat': self._load_text_file,
            '.yaml': self._load_text_file,
            '.yml': self._load_text_file,
            '.toml': self._load_text_file,
            '.ini': self._load_text_file,
            '.cfg': self._load_text_file,
            '.sql': self._load_text_file,

            # Tabular data (loaded as text; tabular parser handles structure)
            '.csv': self._load_text_file,
            '.tsv': self._load_text_file,
        }

        # Add PDF support only if pypdf is available
        if PDF_SUPPORT:
            self.loader_map['.pdf'] = self._load_pdf_file
        else:
            # Optionally store unsupported types for error messaging
            self._unsupported_types = {'.pdf': "pypdf library is not installed."}

    def _get_file_extension(self, file_path: str) -> str:
        """Helper to extract and normalize file extension."""
        return os.path.splitext(file_path)[1].lower()

    def _load_text_file(self, file_path: str, encoding: str = 'utf-8') -> DocumentData:
        """
        Loads content from a generic text-based file, attempting UTF-8 and then Latin-1.
        """
        content = ""
        actual_encoding = encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except UnicodeDecodeError:
            # Fallback to Latin-1 if UTF-8 fails, as it handles most single-byte encodings
            if encoding == 'utf-8':
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                    actual_encoding = 'latin-1'
                except Exception as e:
                    raise IOError(f"Failed to read text file '{file_path}' with UTF-8 or Latin-1 encoding: {e}")
            else:
                raise IOError(f"Failed to read text file '{file_path}' with encoding '{encoding}'.")
        except Exception as e:
            raise IOError(f"Error reading text file '{file_path}': {e}")

        # Basic metadata for text files
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_extension': self._get_file_extension(file_path),
            'file_type_guessed': mimetypes.guess_type(file_path)[0] or 'text/plain',
            'encoding': actual_encoding,
            'size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        return DocumentData(content=content, metadata=metadata)

    def _load_pdf_file(self, file_path: str) -> DocumentData:
        """
        Loads content from a PDF file using the pypdf library.
        Each page's text is extracted and concatenated.
        """
        if not PDF_SUPPORT:
            raise NotImplementedError(
                f"PDF loading is not enabled. {self._unsupported_types.get('.pdf', 'Install `pypdf` to enable it.')}"
            )

        content_parts = []
        try:
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                # .extract_text() can return None for empty/scanned pages
                text = page.extract_text()
                if text:
                    content_parts.append(text)
                # Add a separator between pages for better context, especially if parsers
                # need to understand page boundaries later (though this loader primarily gets raw text)
                if i < len(reader.pages) - 1:
                    content_parts.append("\n--- PAGE_BREAK ---\n")

            content = "".join(content_parts).strip()

            # Metadata for PDF files
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_extension': '.pdf',
                'file_type_guessed': 'application/pdf',
                'total_pages': len(reader.pages),
                'size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'pdf_info': dict(reader.metadata) if reader.metadata else {}
            }
            return DocumentData(content=content, metadata=metadata)
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except Exception as e:
            # Catch general pypdf errors (e.g., malformed PDF)
            raise IOError(f"Error reading PDF file '{file_path}': {e}")

    def load_document(self, file_path: str) -> DocumentData:
        """
        Loads a document from the given file path by determining its type
        and invoking the appropriate internal loader.

        Args:
            file_path: The absolute or relative path to the document file.

        Returns:
            A DocumentData object containing the document's extracted content
            and associated metadata.

        Raises:
            ValueError: If the file path is invalid (e.g., not found, not a file)
                        or if the file type is explicitly unsupported.
            IOError: If there's an error during file reading.
            NotImplementedError: If a required external library for a file type
                                 is not installed (e.g., pypdf).
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a regular file: {file_path}")

        file_extension = self._get_file_extension(file_path)

        # Get the specific loader method for the file extension
        loader_method = self.loader_map.get(file_extension)

        if loader_method:
            return loader_method(file_path)
        else:
            # Fallback: if extension not explicitly mapped, try to load as generic text
            # if its mime type suggests it's text. This handles less common text files.
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('text/'):
                return self._load_text_file(file_path)
            else:
                # If even the mimetype doesn't suggest text, it's genuinely unsupported
                reason = self._unsupported_types.get(file_extension, "No specific loader available.")
                raise ValueError(
                    f"Unsupported document type or file extension '{file_extension}' for '{file_path}'. {reason}"
                )

# Example of how this might be used (for local testing, not part of the required file)
if __name__ == '__main__':
    loader = DocumentLoader()

    # Create dummy files for testing
    os.makedirs("temp_docs", exist_ok=True)
    with open("temp_docs/example.txt", "w", encoding='utf-8') as f:
        f.write("This is a simple text document.\nIt has multiple lines.")
    with open("temp_docs/example.md", "w", encoding='utf-8') as f:
        f.write("# Markdown Example\n\nThis is a paragraph in *markdown*.")
    with open("temp_docs/example.py", "w", encoding='utf-8') as f:
        f.write("def hello_world():\n    print('Hello, RAG!')\n")
    with open("temp_docs/example.csv", "w", encoding='utf-8') as f:
        f.write("Name,Age,City\nAlice,30,New York\nBob,24,London")

    # Test loading
    try:
        txt_doc = loader.load_document("temp_docs/example.txt")
        print(f"--- Loaded TXT ---\nContent:\n{txt_doc.content}\nMetadata: {txt_doc.metadata}\n")

        md_doc = loader.load_document("temp_docs/example.md")
        print(f"--- Loaded MD ---\nContent:\n{md_doc.content}\nMetadata: {md_doc.metadata}\n")

        py_doc = loader.load_document("temp_docs/example.py")
        print(f"--- Loaded PY ---\nContent:\n{py_doc.content}\nMetadata: {py_doc.metadata}\n")

        csv_doc = loader.load_document("temp_docs/example.csv")
        print(f"--- Loaded CSV ---\nContent:\n{csv_doc.content}\nMetadata: {csv_doc.metadata}\n")

        if PDF_SUPPORT:
            # Create a dummy PDF file (requires reportlab or similar for creation, not just pypdf for reading)
            # For actual testing, you'd need a real PDF file.
            # Example: from reportlab.pdfgen import canvas; c = canvas.Canvas("temp_docs/example.pdf"); c.drawString(100,750,"Hello PDF"); c.save()
            print("\n--- PDF Test (requires a real PDF file, generating a dummy PDF is complex) ---")
            # Assuming you have a real 'some_document.pdf' for testing
            # pdf_doc = loader.load_document("temp_docs/some_document.pdf")
            # print(f"--- Loaded PDF ---\nContent (first 200 chars):\n{pdf_doc.content[:200]}...\nMetadata: {pdf_doc.metadata}\n")
            print("To test PDF, place a PDF file named 'example.pdf' in 'temp_docs' and uncomment the lines.")
        else:
            print("\n--- PDF Test Skipped: pypdf not installed ---")
            try:
                # This should raise NotImplementedError
                loader._load_pdf_file("temp_docs/non_existent.pdf")
            except NotImplementedError as e:
                print(f"Caught expected error for PDF without pypdf: {e}")

        # Test error cases
        print("\n--- Testing Error Cases ---")
        try:
            loader.load_document("temp_docs/non_existent.xyz")
        except ValueError as e:
            print(f"Caught expected error for non-existent file type: {e}")
        try:
            loader.load_document("temp_docs/non_existent.txt")
        except ValueError as e:
            print(f"Caught expected error for non-existent text file: {e}")
        try:
            loader.load_document("temp_docs/") # A directory
        except ValueError as e:
            print(f"Caught expected error for loading a directory: {e}")

    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists("temp_docs"):
            shutil.rmtree("temp_docs")
        print("\nCleaned up temp_docs directory.")