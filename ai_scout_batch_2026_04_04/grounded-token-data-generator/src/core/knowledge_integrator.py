import abc
import requests
from typing import Dict, Any, Optional, List
import logging

# Setup basic logging for the module
logger = logging.getLogger(__name__)
# Configure handler if not already configured by a higher-level system
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class KnowledgeSourceError(Exception):
    """Custom exception for errors related to knowledge source access."""
    pass


class BaseKnowledgeIntegrator(abc.ABC):
    """
    Abstract Base Class for all knowledge integrators.
    Defines the unified interface for querying various external knowledge sources.
    """
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def get_info(self, token: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves relevant information for a given token from the knowledge source.

        Args:
            token: The new vocabulary token to query information for.
            context: Optional contextual information to aid in disambiguation or richer queries.

        Returns:
            A dictionary containing structured information about the token, wrapped under
            the integrator's name as a key, or None if no information is found.
            Example: {self.name: {"summary": "...", "type": "..."}}
        """
        pass


class DummyKnowledgeIntegrator(BaseKnowledgeIntegrator):
    """
    A dummy integrator for testing and demonstration purposes.
    Returns mock data based on the token.
    """
    def __init__(self, name: str = "dummy_source"):
        super().__init__(name)
        self._mock_data = {
            "GTI": {"description": "Grounded Token Initialization, a method for extending LLM vocabulary.", "type": "method", "domain": "AI/NLP"},
            "PairedLinguisticSupervision": {"description": "Natural language descriptions linked to new vocabulary tokens, used for semantic grounding.", "type": "concept", "domain": "AI/NLP"},
            "NovelToken": {"description": "A new or unknown token introduced to an LLM's vocabulary.", "type": "concept", "domain": "LLM Extension"},
            "Quantium": {"description": "A hypothetical element with atomic number 130.", "type": "element", "domain": "chemistry"},
            "CyberHive": {"description": "A collaborative platform for cybersecurity threat intelligence.", "type": "platform", "domain": "cybersecurity"},
        }

    def get_info(self, token: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.debug(f"Querying dummy source for token: '{token}' with context: '{context}'")
        info = self._mock_data.get(token)
        if info:
            logger.info(f"Found dummy info for '{token}'.")
            return {self.name: info}
        logger.debug(f"No dummy info found for '{token}'.")
        return None


class WikipediaIntegrator(BaseKnowledgeIntegrator):
    """
    Integrates with the Wikipedia API to fetch summaries for tokens.
    Uses the MediaWiki API directly via requests.
    """
    def __init__(self, name: str = "wikipedia_source", lang: str = "en"):
        super().__init__(name)
        self.api_endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.timeout = 5  # seconds

    def get_info(self, token: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.debug(f"Querying Wikipedia for token: '{token}' with context: '{context}'")
        params = {
            "action": "query",
            "format": "json",
            "titles": token,
            "prop": "extracts",
            "exintro": True,  # Get only the introduction
            "explaintext": True,  # Return plain text
            "redirects": 1,       # Resolve redirects
            "converttitles": 1    # Convert titles to canonical form
        }

        try:
            response = self.session.get(self.api_endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id == "-1":  # Page not found
                    logger.debug(f"Wikipedia page not found for '{token}'.")
                    return None

                extract = page_info.get("extract")
                if extract:
                    title = page_info.get("title", token)  # Use found title if different
                    logger.info(f"Found Wikipedia info for '{token}'.")
                    return {
                        self.name: {
                            "title": title,
                            "summary": extract,
                            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        }
                    }
            logger.debug(f"No usable extract found in Wikipedia response for '{token}'.")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Wikipedia API request timed out for '{token}'.")
            raise KnowledgeSourceError(f"Wikipedia API request timed out for '{token}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Wikipedia for '{token}': {e}")
            raise KnowledgeSourceError(f"Failed to query Wikipedia for '{token}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing Wikipedia response for '{token}': {e}")
            raise KnowledgeSourceError(f"Unexpected error with Wikipedia for '{token}': {e}")


class StructuredDBIntegrator(BaseKnowledgeIntegrator):
    """
    A placeholder integrator for a structured database (e.g., a domain-specific
    SQL database or a NoSQL document store).
    For the prototype, this will simulate a lookup in an in-memory dictionary.
    """
    def __init__(self, name: str = "structured_db_source", db_config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self._db_config = db_config or {}
        # Simulate a database (e.g., loaded from a JSON file or connected to)
        self._database = {
            "AIQuant": {"definition": "An AI-powered quantitative analysis platform.", "category": "FinTech", "founder": "Jane Doe", "year": 2023},
            "NeuralSynth": {"definition": "A novel framework for synthetic data generation using neural networks.", "category": "AI/ML Research", "developed_by": "Acme Labs", "status": "alpha"},
            "EcoCycle": {"definition": "A sustainable waste management and recycling initiative.", "category": "Environmental", "location": "Global", "partners": ["UNEP", "GreenPeace"]},
        }
        logger.debug(f"Initialized StructuredDBIntegrator with config: {self._db_config}")

    def get_info(self, token: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.debug(f"Querying structured database for token: '{token}' with context: '{context}'")
        try:
            # In a real scenario, this would involve actual database queries (SQL, NoSQL client calls)
            # For simplicity in prototype, we do a direct dictionary lookup
            info = self._database.get(token)
            if info:
                logger.info(f"Found structured DB info for '{token}'.")
                return {self.name: info}
            logger.debug(f"No structured DB info found for '{token}'.")
            return None
        except Exception as e:
            logger.error(f"Error querying structured database for '{token}': {e}")
            raise KnowledgeSourceError(f"Failed to query structured database for '{token}': {e}")


class KnowledgeIntegrator:
    """
    The main KnowledgeIntegrator class, responsible for managing and querying
    multiple underlying knowledge sources. It aggregates information from various
    integrators to provide a comprehensive view.
    """
    def __init__(self, integrators: Optional[List[BaseKnowledgeIntegrator]] = None):
        self._integrators: List[BaseKnowledgeIntegrator] = []
        if integrators:
            for integrator in integrators:
                self.add_integrator(integrator)
        if not self._integrators:
            logger.warning("KnowledgeIntegrator initialized with no integrators. "
                           "Add integrators using add_integrator() to make it functional.")

    def add_integrator(self, integrator: BaseKnowledgeIntegrator):
        """Adds a new knowledge source integrator to the system."""
        if not isinstance(integrator, BaseKnowledgeIntegrator):
            raise TypeError("Provided integrator must be an instance of BaseKnowledgeIntegrator.")
        self._integrators.append(integrator)
        logger.info(f"Added knowledge integrator: '{integrator.name}'")

    def get_info(self, token: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Queries all registered knowledge sources for information about the token
        and aggregates the results.

        Args:
            token: The new vocabulary token.
            context: Optional contextual information.

        Returns:
            A dictionary where keys are source names and values are the
            information retrieved from each source. Returns an empty dict if no
            information is found from any source or if all sources fail.
        """
        aggregated_info: Dict[str, Any] = {}
        if not self._integrators:
            logger.warning(f"No integrators registered. Cannot retrieve information for '{token}'.")
            return {}

        for integrator in self._integrators:
            try:
                info_from_source = integrator.get_info(token, context)
                if info_from_source:
                    # Each integrator is expected to return {self.name: data}
                    # So we update the aggregated_info with the source's data
                    aggregated_info.update(info_from_source)
            except KnowledgeSourceError as e:
                logger.error(f"Skipping integrator '{integrator.name}' due to error: {e}")
            except Exception as e:
                # Catch any unexpected errors to prevent a single source from crashing the system
                logger.critical(f"An unhandled error occurred in integrator '{integrator.name}' for token '{token}': {e}", exc_info=True)
        
        if not aggregated_info:
            logger.info(f"No information found from any active source for token: '{token}'")
        else:
            logger.info(f"Aggregated info for '{token}' from sources: {list(aggregated_info.keys())}")

        return aggregated_info