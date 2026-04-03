import os
import requests
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# External libraries for document parsing
try:
    import pypdf
except ImportError:
    pypdf = None
    logging.warning("pypdf not installed. PDF loading functionality will be unavailable.")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup4 not installed. Web page loading functionality will be limited.")

# For structured data, we might use a simple SQLite connection for prototyping.
# For more complex databases, a dedicated ORM or connector would be used.
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a basic Document structure for output
# In a full system, this would likely be a Pydantic model from src/utils/schemas.py
class LoadedDocument:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

    def to_dict(self):
        return {"content": self.content, "metadata": self.metadata}

class DocumentLoader:
    """
    Handles loading diverse document types (PDFs, web pages, plain text, databases)
    while preserving inherent structure as much as possible.

    Returns a list of `LoadedDocument` objects, where each object represents
    a logical segment of the original document (e.g., a page, a paragraph, a record).
    """

    def load(self, source_type: str, source_identifier: Any, **kwargs) -> List[LoadedDocument]:
        """
        Loads a document based on its type and identifier.

        Args:
            source_type (str): Type of the source ('pdf', 'webpage', 'text', 'database').
            source_identifier (Any): Identifier for the source (e.g., file path, URL, connection string).
            **kwargs: Additional parameters specific to the source type (e.g., query for database).

        Returns:
            List[LoadedDocument]: A list of loaded document segments, each with content and metadata.
                                   Returns an empty list if loading fails or source type is unsupported.
        """
        try:
            if source_type == 'pdf':
                return self._load_pdf(source_identifier)
            elif source_type == 'webpage':
                return self._load_webpage(source_identifier)
            elif source_type == 'text':
                return self._load_text(source_identifier)
            elif source_type == 'database':
                query = kwargs.get('query')
                if not query:
                    logger.error("Database source type requires a 'query' argument.")
                    return []
                return self._load_database(source_identifier, query)
            else:
                logger.warning(f"Unsupported document type: {source_type}")
                return []
        except Exception as e:
            logger.error(f"Error loading document from {source_type} '{source_identifier}': {e}", exc_info=True)
            return []

    def _load_pdf(self, file_path: str) -> List[LoadedDocument]:
        """
        Loads content from a PDF file, extracting text page by page.
        Each page is treated as a separate LoadedDocument.
        """
        if pypdf is None:
            logger.error("PyPDF2 or pypdf library is not installed. Cannot load PDF files.")
            return []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents: List[LoadedDocument] = []
        try:
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            LoadedDocument(
                                content=text,
                                metadata={
                                    "source": file_path,
                                    "source_type": "pdf",
                                    "page_number": i + 1,
                                    "total_pages": len(reader.pages),
                                    "title": reader.metadata.title if reader.metadata and reader.metadata.title else os.path.basename(file_path)
                                }
                            )
                        )
        except pypdf.errors.PdfReadError as e:
            logger.error(f"Failed to read PDF file '{file_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing PDF '{file_path}': {e}")
            raise

        logger.info(f"Loaded {len(documents)} pages from PDF: {file_path}")
        return documents

    def _load_webpage(self, url: str) -> List[LoadedDocument]:
        """
        Loads content from a web page. Attempts to extract main text content.
        The entire extracted text is treated as a single LoadedDocument.
        """
        if BeautifulSoup is None:
            logger.error("BeautifulSoup4 library is not installed. Cannot load web pages effectively.")
            # Fallback to plain text if BeautifulSoup is not available
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Raise an exception for HTTP errors
                return [LoadedDocument(content=response.text, metadata={"source": url, "source_type": "webpage"})]
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch webpage '{url}': {e}")
                raise
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Attempt to find the main content. This is heuristic and can be improved.
            # Common patterns: <main>, <article>, div with specific ids/classes.
            main_content_element = soup.find('main') or soup.find('article') or soup.find(class_='post-content') or soup.find('body')

            if main_content_element:
                # Remove script and style tags
                for script_or_style in main_content_element(["script", "style"]):
                    script_or_style.extract()
                text = main_content_element.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            if not text:
                logger.warning(f"No significant text content extracted from webpage: {url}")
                return []

            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Basic metadata extraction
            title = soup.find('title').get_text(strip=True) if soup.find('title') else os.path.basename(parsed_url.path)
            # Add more metadata like author, publication date, etc. if available in meta tags

            logger.info(f"Loaded webpage: {url}")
            return [
                LoadedDocument(
                    content=text,
                    metadata={
                        "source": url,
                        "source_type": "webpage",
                        "title": title,
                        "base_url": base_url
                    }
                )
            ]
        except requests.exceptions.Timeout:
            logger.error(f"Request to '{url}' timed out after 10 seconds.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch webpage '{url}': {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing webpage '{url}': {e}")
            raise

    def _load_text(self, file_path: str) -> List[LoadedDocument]:
        """
        Loads plain text from a file. The entire file content is treated as a single LoadedDocument.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Text file '{file_path}' is empty or contains only whitespace.")
                return []

            logger.info(f"Loaded text file: {file_path}")
            return [
                LoadedDocument(
                    content=content,
                    metadata={
                        "source": file_path,
                        "source_type": "text",
                        "filename": os.path.basename(file_path)
                    }
                )
            ]
        except UnicodeDecodeError:
            logger.error(f"Could not decode text file '{file_path}' with UTF-8. Trying with a different encoding.")
            try:
                with open(file_path, 'r', encoding='latin-1') as f: # Fallback to latin-1
                    content = f.read()
                return [
                    LoadedDocument(
                        content=content,
                        metadata={
                            "source": file_path,
                            "source_type": "text",
                            "filename": os.path.basename(file_path)
                        }
                    )
                ]
            except Exception as e:
                logger.error(f"Failed to read text file '{file_path}' with fallback encoding: {e}")
                raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing text file '{file_path}': {e}")
            raise

    def _load_database(self, connection_string: str, query: str) -> List[LoadedDocument]:
        """
        Loads data from a database using a connection string and a SQL query.
        Each row returned by the query is treated as a separate LoadedDocument,
        where its content is a string representation of the row and metadata includes
        the original row data.

        This is a basic implementation using sqlite3. For production, abstract over
        different database types (PostgreSQL, MySQL, etc.) and use appropriate ORMs/libraries.
        """
        documents: List[LoadedDocument] = []
        conn: Optional[sqlite3.Connection] = None
        try:
            # Assuming connection_string is a path to a SQLite database file for simplicity.
            # In a real system, it could be a full connection string for other DBs.
            conn = sqlite3.connect(connection_string)
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                logger.warning(f"Database query returned no results for '{query}' from '{connection_string}'")
                return []

            for i, row in enumerate(rows):
                row_dict = dict(row) # Convert sqlite3.Row to a dictionary
                
                # Create a string representation of the row for 'content'
                # This can be customized based on how structured data should appear in RAG
                content_str = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])

                documents.append(
                    LoadedDocument(
                        content=content_str,
                        metadata={
                            "source": connection_string,
                            "source_type": "database",
                            "query": query,
                            "record_index": i,
                            "original_data": row_dict # Store original structured data in metadata
                        }
                    )
                )
            logger.info(f"Loaded {len(documents)} records from database query: {query} in {connection_string}")
        except sqlite3.Error as e:
            logger.error(f"Database error executing query '{query}' on '{connection_string}': {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing database query: {e}")
            raise
        finally:
            if conn:
                conn.close()
        return documents

# Example Usage (for testing purposes, not part of the file's main logic)
if __name__ == "__main__":
    loader = DocumentLoader()

    # 1. Load PDF
    # Create a dummy PDF for testing if needed, or use an existing one
    # For a real test, you'd need an actual PDF file.
    # For now, let's just illustrate.
    print("\n--- Loading PDF ---")
    dummy_pdf_path = "dummy.pdf" # Replace with actual path
    if os.path.exists(dummy_pdf_path) and pypdf:
        try:
            pdf_docs = loader.load('pdf', dummy_pdf_path)
            for doc in pdf_docs[:2]: # Print first two pages
                print(f"PDF Doc Metadata: {doc.metadata}")
                print(f"PDF Doc Content (partial): {doc.content[:200]}...")
        except Exception as e:
            print(f"PDF loading failed: {e}")
    else:
        print(f"Skipping PDF loading. '{dummy_pdf_path}' not found or pypdf not installed.")
        # Create a simple dummy PDF for future testing if it doesn't exist
        if pypdf:
            try:
                from pypdf import PdfWriter
                writer = PdfWriter()
                writer.add_blank_page(width=72, height=72)
                writer.add_blank_page(width=72, height=72)
                with open(dummy_pdf_path, "wb") as f:
                    writer.write(f)
                print(f"Created a dummy PDF at {dummy_pdf_path}. Rerun to test.")
            except Exception as e:
                print(f"Could not create dummy PDF: {e}")


    # 2. Load Web Page
    print("\n--- Loading Web Page ---")
    test_url = "https://www.example.com"
    try:
        web_docs = loader.load('webpage', test_url)
        if web_docs:
            for doc in web_docs:
                print(f"Webpage Doc Metadata: {doc.metadata}")
                print(f"Webpage Doc Content (partial): {doc.content[:500]}...")
        else:
            print(f"No content loaded from {test_url}")
    except Exception as e:
        print(f"Webpage loading failed: {e}")

    # 3. Load Text File
    print("\n--- Loading Text File ---")
    dummy_txt_path = "dummy.txt"
    with open(dummy_txt_path, "w") as f:
        f.write("This is a test text file.\nIt contains some sample data for demonstration.\n")
        f.write("Another line with important information.\n")
    try:
        text_docs = loader.load('text', dummy_txt_path)
        if text_docs:
            for doc in text_docs:
                print(f"Text Doc Metadata: {doc.metadata}")
                print(f"Text Doc Content:\n{doc.content}")
        else:
            print(f"No content loaded from {dummy_txt_path}")
    except Exception as e:
        print(f"Text file loading failed: {e}")
    finally:
        if os.path.exists(dummy_txt_path):
            os.remove(dummy_txt_path)

    # 4. Load from Database (SQLite example)
    print("\n--- Loading Database ---")
    db_path = "test_db.sqlite"
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS articles")
        cursor.execute("""
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                content TEXT,
                publication_date TEXT
            )
        """)
        cursor.execute("INSERT INTO articles VALUES (1, 'The Future of AI', 'Dr. A. Scientist', 'Artificial intelligence is rapidly evolving...', '2023-10-26')")
        cursor.execute("INSERT INTO articles VALUES (2, 'Quantum Computing Basics', 'Prof. B. Physicist', 'Quantum computing leverages quantum-mechanical phenomena...', '2022-05-15')")
        conn.commit()

        db_query = "SELECT title, author, content FROM articles WHERE id = 1"
        db_docs = loader.load('database', db_path, query=db_query)
        if db_docs:
            for doc in db_docs:
                print(f"DB Doc Metadata: {doc.metadata}")
                print(f"DB Doc Content: {doc.content}")
        else:
            print(f"No content loaded from database with query: {db_query}")

        db_query_all = "SELECT * FROM articles"
        db_docs_all = loader.load('database', db_path, query=db_query_all)
        if db_docs_all:
            print(f"\nAll records from DB:")
            for doc in db_docs_all:
                print(f"  Doc Metadata (original_data): {doc.metadata.get('original_data')}")
                print(f"  Doc Content: {doc.content}")
        
    except Exception as e:
        print(f"Database loading failed: {e}")
    finally:
        if conn:
            conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)

    print("\n--- Testing Error Handling (File Not Found) ---")
    try:
        loader.load('text', "non_existent_file.txt")
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Testing Error Handling (Unsupported Type) ---")
    unsupported_docs = loader.load('unsupported_type', 'some_id')
    print(f"Unsupported type result: {unsupported_docs}")