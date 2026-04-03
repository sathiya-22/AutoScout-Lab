```python
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import uuid

# Set up basic logging for this module
logger = logging.getLogger(__name__)
# Basic handler for local development if not configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Assumed / Minimal Schemas ---
# These schemas would ideally be imported from `src.utils.schemas`.
# They are defined here to make this file self-contained for prototyping purposes.
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata associated with a document or chunk of content."""
    source_uri: Optional[str] = None
    last_updated: Optional[datetime] = None  # When the content was last updated/published (UTC)
    source_authority_score: Optional[float] = None # A score from 0.0 (low) to 1.0 (high)
    data_type: str = "textual" # e.g., "textual", "numerical_heavy", "table", "image_description", "code"
    original_creation_date: Optional[datetime] = None # When the document was originally created (UTC)
    title: Optional[str] = None
    author: Optional[str] = None
    file_type: Optional[str] = None # e.g., "pdf", "html", "docx", "json"
    # Add other potential metadata fields as needed

class Document(BaseModel):
    """Represents a document or a significant logical chunk of content."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
# --- End of Assumed Schemas ---


class MetadataExtractor:
    """
    Extracts critical metadata from a document, such as source authority, freshness,
    and data types. This metadata is crucial for later retrieval filtering and ranking.
    """

    def __init__(self, authority_mapping: Optional[Dict[str, float]] = None):
        """
        Initializes the MetadataExtractor with an optional authority mapping.

        Args:
            authority_mapping (Optional[Dict[str, float]]): A dictionary mapping
                regular expression patterns for domains (e.g., r"\.gov$", r"wikipedia\.org")
                to authority scores (0.0-1.0). Patterns are checked against the source_uri.
        """
        self.authority_mapping = authority_mapping if authority_mapping is not None else self._default_authority_mapping()
        logger.info("MetadataExtractor initialized.")

    def _default_authority_mapping(self) -> Dict[str, float]:
        """
        Provides a default set of authority mappings for common, generally reputable domains.
        These are regex patterns to match parts of the source URI.
        """
        return {
            r"\.gov(?:\/|$)": 1.0,  # Government domains
            r"\.edu(?:\/|$)": 0.9,  # Educational institutions
            r"wikipedia\.org": 0.8,
            r"npr\.org": 0.75,
            r"nytimes\.com": 0.7,
            r"wsj\.com": 0.7,
            r"bbc\.co\.uk": 0.75,
            r"who\.int": 0.95, # World Health Organization
            r"cdc\.gov": 0.95, # Centers for Disease Control and Prevention
            # Add more patterns for reputable sources, lower scores for generic or less reliable sources
        }

    def _determine_source_authority(self, source_uri: Optional[str]) -> Optional[float]:
        """
        Determines the source authority score based on the source URI.

        Args:
            source_uri (Optional[str]): The URI of the document's source.

        Returns:
            Optional[float]: A score from 0.0 to 1.0, or None if authority cannot be determined.
        """
        if not source_uri:
            logger.debug("No source URI provided for authority determination.")
            return None

        for pattern, score in self.authority_mapping.items():
            if re.search(pattern, source_uri, re.IGNORECASE):
                logger.debug(f"Matched authority pattern '{pattern}' for URI '{source_uri}', score: {score}")
                return score

        # Default to a neutral score if no specific pattern matches
        logger.debug(f"No specific authority pattern matched for URI '{source_uri}'. Defaulting to neutral score.")
        return 0.5 # A neutral score for unknown sources

    def _determine_freshness(self, metadata: DocumentMetadata, content: str) -> Optional[datetime]:
        """
        Determines the freshness of the document.
        Prioritizes existing metadata, then tries to parse common date patterns from content.

        Args:
            metadata (DocumentMetadata): The existing metadata of the document.
            content (str): The full content of the document.

        Returns:
            Optional[datetime]: The estimated last updated/publication date (timezone-aware UTC), or None.
        """
        # 1. Prioritize existing metadata fields
        if metadata.last_updated:
            logger.debug(f"Freshness from metadata.last_updated: {metadata.last_updated}")
            # Ensure it's timezone-aware
            return metadata.last_updated if metadata.last_updated.tzinfo else metadata.last_updated.replace(tzinfo=timezone.utc)
        if metadata.original_creation_date:
            logger.debug(f"Freshness from metadata.original_creation_date: {metadata.original_creation_date}")
            # Ensure it's timezone-aware
            return metadata.original_creation_date if metadata.original_creation_date.tzinfo else metadata.original_creation_date.replace(tzinfo=timezone.utc)

        # 2. Try to parse common date patterns from content
        # This is a simplified regex-based attempt; a more robust solution might use date parsing libraries
        # like `dateutil.parser` for better flexibility.
        date_patterns = [
            r"(?:Last updated|Publication Date|Published|Date):\s*(\w+\s+\d{1,2},\s+\d{4})",  # "January 1, 2023"
            r"(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)", # YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
            r"(\d{1,2}/\d{1,2}/\d{4})", # MM/DD/YYYY or DD/MM/YYYY
            r"(\d{4}\/\d{1,2}\/\d{1,2})", # YYYY/MM/DD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                try:
                    dt: Optional[datetime] = None
                    # Attempt to parse various common date formats
                    if re.match(r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?", date_str):
                        fmts = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]
                        for fmt in fmts:
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                pass
                    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
                        # Ambiguous: could be MM/DD/YYYY or DD/MM/YYYY. Defaulting to MM/DD.
                        dt = datetime.strptime(date_str, "%m/%d/%Y")
                    elif re.match(r"\d{4}\/\d{1,2}\/\d{1,2}", date_str):
                        dt = datetime.strptime(date_str, "%Y/%m/%d")
                    else:
                        dt = datetime.strptime(date_str, "%B %d, %Y") # Month Day, Year

                    if dt:
                        logger.debug(f"Freshness parsed from content ('{date_str}'): {dt}")
                        # Assume UTC if no timezone info, or if parsing didn't provide it
                        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
                except ValueError:
                    logger.warning(f"Could not parse date '{date_str}' found with pattern '{pattern}' in content. Skipping.")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error parsing date '{date_str}' with pattern '{pattern}': {e}")
                    continue
        
        logger.debug("No freshness information found in metadata or content that could be parsed.")
        return None

    def _determine_data_type(self, content: str) -> str:
        """
        Determines the primary data type of the content (e.g., "textual", "numerical_heavy").
        This is a heuristic and can be expanded for more granular types (e.g., "code", "table").

        Args:
            content (str): The content of the document or chunk.

        Returns:
            str: The determined data type.
        """
        if not content:
            return "textual" # Default for empty content

        total_chars = len(content)
        if total_chars == 0:
            return "textual"

        # Heuristic for "numerical_heavy": Check the ratio of digits to total characters
        num_digits = sum(c.isdigit() for c in content)
        
        # Consider characters typically found in numerical data (e.g., digits, periods, commas, currency symbols, '%')
        numerical_like_chars = sum(1 for c in content if c.isdigit() or c in '.,$€£%')
        
        digit_ratio = num_digits / total_chars
        numerical_like_ratio = numerical_like_chars / total_chars

        # Define thresholds for what constitutes "numerical_heavy"
        NUMERICAL_HEAVY_DIGIT_THRESHOLD = 0.30 # If more than 30% of characters are digits
        NUMERICAL_HEAVY_TOTAL_THRESHOLD = 0.50 # If more than 50% of characters are numerical-like

        if digit_ratio >= NUMERICAL_HEAVY_DIGIT_THRESHOLD or numerical_like_ratio >= NUMERICAL_HEAVY_TOTAL_THRESHOLD:
            # We could also check for common table structures (e.g., rows/columns, CSV/TSV indicators)
            # For this prototype, a simple character ratio is sufficient for 'numerical_heavy' vs 'textual'.
            logger.debug(f"Content classified as 'numerical_heavy' (digit ratio: {digit_ratio:.2f}, numerical-like ratio: {numerical_like_ratio:.2f})")
            return "numerical_heavy"
        
        logger.debug(f"Content classified as 'textual' (digit ratio: {digit_ratio:.2f}, numerical-like ratio: {numerical_like_ratio:.2f})")
        return "textual"

    def extract(self, document: Document) -> Document:
        """
        Extracts and updates critical metadata for a given document.

        Args:
            document (Document): The document object to process.

        Returns:
            Document: The document object with updated metadata.
        """
        logger.info(f"Starting metadata extraction for document ID: {document.id[:8]}...")

        # Source Authority
        authority_score = self._determine_source_authority(document.metadata.source_uri)
        if authority_score is not None:
            document.metadata.source_authority_score = authority_score
        else:
            document.metadata.source_authority_score = 0.5 # Default to neutral if cannot determine

        # Freshness
        freshness_date = self._determine_freshness(document.metadata, document.content)
        if freshness_date:
            document.metadata.last_updated = freshness_date
        # If no freshness found, last_updated remains None or existing value.

        # Data Type
        data_type = self._determine_data_type(document.content)
        document.metadata.data_type = data_type
        
        logger.info(f"Finished metadata extraction for document ID: {document.id[:8]}. "
                    f"Authority: {document.metadata.source_authority_score}, "
                    f"Freshness: {document.metadata.last_updated.isoformat() if document.metadata.last_updated else 'N/A'}, "
                    f"DataType: {document.metadata.data_type}")
        return document

```