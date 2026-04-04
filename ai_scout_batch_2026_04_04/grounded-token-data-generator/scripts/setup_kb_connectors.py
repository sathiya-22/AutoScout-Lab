import os
import sys
import abc
import logging

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mocking Architectural Components for Self-Containment ---
# In a real project, these would be imported from their respective paths:
# from config import KB_CONNECTORS_CONFIG
# from src.core.knowledge_integrator import KnowledgeBaseConnector, WikidataConnector, LocalDomainDBConnector

#region Mock config.py
# This simulates the configuration that would typically come from config.py
# It defines which knowledge base connectors are enabled and their parameters.
_MOCK_KB_CONNECTORS_CONFIG = {
    "WIKIDATA": {
        "enabled": True,
        "class": "WikidataConnector", # Name of the class to instantiate
        "params": {
            "endpoint": "https://query.wikidata.org/sparql",
            # Add any API keys or credentials here if needed for a real service
        }
    },
    "LOCAL_DOMAIN_DB": {
        "enabled": True,
        "class": "LocalDomainDBConnector",
        "params": {
            "db_path": os.path.join(os.path.dirname(__file__), "../data/local_domain.db"),
            # Placeholder for a simple local SQLite database
        }
    },
    "CUSTOM_API_KB": {
        "enabled": False, # This one is disabled for demonstration
        "class": "CustomAPIDBConnector",
        "params": {
            "api_url": "http://api.example.com/knowledge",
            "api_key_env_var": "CUSTOM_KB_API_KEY", # Assuming API key is from environment variable
        }
    }
}
#endregion

#region Mock src/core/knowledge_integrator.py
# This simulates the Knowledge Integration Layer classes.
# In a real project, these would be in src/core/knowledge_integrator.py or a subpackage.

class KnowledgeBaseConnector(abc.ABC):
    """Abstract base class for all knowledge base connectors."""

    def __init__(self, name: str):
        self.name = name
        self._is_connected = False

    @abc.abstractmethod
    def connect(self):
        """Establishes a connection to the knowledge base."""
        pass

    @abc.abstractmethod
    def query(self, query_string: str, **kwargs) -> dict:
        """Queries the knowledge base for information."""
        pass

    @abc.abstractmethod
    def disconnect(self):
        """Closes the connection to the knowledge base."""
        pass

    def is_connected(self) -> bool:
        """Returns True if the connector is currently connected."""
        return self._is_connected

class WikidataConnector(KnowledgeBaseConnector):
    """
    Concrete connector for Wikidata SPARQL endpoint.
    This is a simplified mock. A real implementation would use libraries
    like `sparqlwrapper` or `requests` for HTTP requests.
    """
    def __init__(self, endpoint: str, name: str = "Wikidata"):
        super().__init__(name)
        self.endpoint = endpoint
        self._session = None # Placeholder for an actual HTTP session

    def connect(self):
        """Simulates connecting to the Wikidata SPARQL endpoint."""
        logger.info(f"Attempting to connect to {self.name} at {self.endpoint}...")
        try:
            # In a real scenario, this might involve a simple HTTP HEAD request
            # or checking API availability. For now, just set a flag.
            self._session = True # Represents a successful connection
            self._is_connected = True
            logger.info(f"{self.name} connector successfully connected.")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to {self.name}: {e}")
            raise ConnectionError(f"Could not connect to {self.name}.") from e

    def query(self, query_string: str, **kwargs) -> dict:
        """Simulates querying Wikidata."""
        if not self.is_connected():
            raise ConnectionError(f"{self.name} connector is not connected.")
        logger.debug(f"Querying {self.name} with: '{query_string}'")
        # Mocking a response for demonstration
        mock_response = {
            "source": self.name,
            "query": query_string,
            "results": [
                {"item": "Q42", "label": "Douglas Adams", "description": "English writer"},
                {"item": "Q123", "label": "New Token Example", "description": "A new concept for GTI"}
            ],
            "note": "This is mock data from WikidataConnector"
        }
        return mock_response

    def disconnect(self):
        """Simulates disconnecting from Wikidata."""
        if self._session:
            logger.info(f"Disconnecting {self.name} connector.")
            self._session = None
            self._is_connected = False

class LocalDomainDBConnector(KnowledgeBaseConnector):
    """
    Concrete connector for a local domain-specific database (e.g., SQLite).
    This is a simplified mock. A real implementation would use `sqlite3` or an ORM.
    """
    def __init__(self, db_path: str, name: str = "LocalDomainDB"):
        super().__init__(name)
        self.db_path = db_path
        self._db_connection = None # Placeholder for actual DB connection object

    def connect(self):
        """Simulates connecting to a local SQLite database."""
        logger.info(f"Attempting to connect to {self.name} at {self.db_path}...")
        try:
            import sqlite3
            # Create a dummy database file if it doesn't exist for demonstration
            if not os.path.exists(os.path.dirname(self.db_path)):
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._db_connection = sqlite3.connect(self.db_path)
            # Create a dummy table if it doesn't exist
            cursor = self._db_connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    metadata TEXT
                )
            """)
            self._db_connection.commit()
            self._is_connected = True
            logger.info(f"{self.name} connector successfully connected to {self.db_path}.")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to {self.name}: {e}")
            raise ConnectionError(f"Could not connect to {self.name}.") from e

    def query(self, query_string: str, **kwargs) -> dict:
        """Simulates querying the local database."""
        if not self.is_connected():
            raise ConnectionError(f"{self.name} connector is not connected.")
        logger.debug(f"Querying {self.name} with: '{query_string}'")
        try:
            cursor = self._db_connection.cursor()
            # Simple query for demonstration
            cursor.execute("SELECT name, description FROM tokens WHERE name LIKE ?", (f"%{query_string}%",))
            results = cursor.fetchall()
            if not results:
                # If no existing data, insert mock data for the query_string
                cursor.execute("INSERT OR IGNORE INTO tokens (name, description, metadata) VALUES (?, ?, ?)",
                               (query_string, f"A custom description for {query_string} from local DB.", "{}"))
                self._db_connection.commit()
                cursor.execute("SELECT name, description FROM tokens WHERE name = ?", (query_string,))
                results = cursor.fetchall()

            mock_response = {
                "source": self.name,
                "query": query_string,
                "results": [{"name": r[0], "description": r[1]} for r in results],
                "note": "This is mock data from LocalDomainDBConnector"
            }
            return mock_response
        except Exception as e:
            logger.error(f"Error querying {self.name}: {e}")
            raise

    def disconnect(self):
        """Simulates disconnecting from the local database."""
        if self._db_connection:
            logger.info(f"Disconnecting {self.name} connector.")
            self._db_connection.close()
            self._db_connection = None
            self._is_connected = False

class CustomAPIDBConnector(KnowledgeBaseConnector):
    """
    Concrete connector for a custom API-based knowledge base.
    This is a simplified mock.
    """
    def __init__(self, api_url: str, api_key_env_var: str, name: str = "CustomAPIDB"):
        super().__init__(name)
        self.api_url = api_url
        self.api_key_env_var = api_key_env_var
        self._api_key = os.getenv(api_key_env_var)
        self._session = None # Placeholder for requests.Session

    def connect(self):
        logger.info(f"Attempting to connect to {self.name} at {self.api_url}...")
        if not self._api_key:
            logger.error(f"API key for {self.name} (env var: {self.api_key_env_var}) not found.")
            raise ConnectionError(f"Missing API key for {self.name}.")
        try:
            # Simulate a connection check (e.g., a simple health check endpoint)
            # In a real scenario, use `requests` to make a call.
            # import requests
            # response = requests.get(f"{self.api_url}/health", timeout=5)
            # response.raise_for_status()
            self._session = True # Represents a successful connection
            self._is_connected = True
            logger.info(f"{self.name} connector successfully connected.")
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to {self.name}: {e}")
            raise ConnectionError(f"Could not connect to {self.name}.") from e

    def query(self, query_string: str, **kwargs) -> dict:
        if not self.is_connected():
            raise ConnectionError(f"{self.name} connector is not connected.")
        logger.debug(f"Querying {self.name} with: '{query_string}'")
        # Mocking a response
        mock_response = {
            "source": self.name,
            "query": query_string,
            "results": [
                {"entity": query_string, "info": f"Details from custom API for {query_string}"}
            ],
            "note": "This is mock data from CustomAPIDBConnector"
        }
        return mock_response

    def disconnect(self):
        if self._session:
            logger.info(f"Disconnecting {self.name} connector.")
            self._session = None
            self._is_connected = False

# Mapping of class names to actual classes
_CONNECTOR_CLASSES = {
    "WikidataConnector": WikidataConnector,
    "LocalDomainDBConnector": LocalDomainDBConnector,
    "CustomAPIDBConnector": CustomAPIDBConnector,
}
#endregion
# --- End of Mocking Architectural Components ---


def setup_kb_connectors(
    config_data: dict = None
) -> dict[str, KnowledgeBaseConnector]:
    """
    Initializes and connects to various knowledge base connectors based on configuration.

    Args:
        config_data (dict, optional): A dictionary containing the configuration for KB connectors.
                                     If None, uses the internal mock configuration.

    Returns:
        dict[str, KnowledgeBaseConnector]: A dictionary where keys are connector names
                                           and values are initialized and connected
                                           KnowledgeBaseConnector instances.
                                           Only successfully connected connectors are included.
    """
    if config_data is None:
        config_data = _MOCK_KB_CONNECTORS_CONFIG

    active_connectors: dict[str, KnowledgeBaseConnector] = {}

    logger.info("Setting up knowledge base connectors...")

    for connector_name, settings in config_data.items():
        if not settings.get("enabled", False):
            logger.info(f"Connector '{connector_name}' is disabled. Skipping.")
            continue

        connector_class_name = settings.get("class")
        if not connector_class_name:
            logger.warning(f"Connector '{connector_name}' is missing a 'class' definition. Skipping.")
            continue

        connector_class = _CONNECTOR_CLASSES.get(connector_class_name)
        if not connector_class:
            logger.error(f"Unknown connector class '{connector_class_name}' for '{connector_name}'. Skipping.")
            continue

        try:
            params = settings.get("params", {})
            # Instantiate the connector
            connector_instance: KnowledgeBaseConnector = connector_class(name=connector_name, **params)
            # Attempt to connect
            connector_instance.connect()
            if connector_instance.is_connected():
                active_connectors[connector_name] = connector_instance
            else:
                logger.warning(f"Connector '{connector_name}' failed to establish a connection.")
        except ConnectionError as ce:
            logger.error(f"Failed to initialize or connect '{connector_name}': {ce}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while setting up '{connector_name}': {e}", exc_info=True)

    if not active_connectors:
        logger.warning("No knowledge base connectors were successfully set up.")
    else:
        logger.info(f"Successfully set up {len(active_connectors)} knowledge base connector(s).")
        for name in active_connectors.keys():
            logger.info(f"  - {name}")

    return active_connectors

def disconnect_all_kb_connectors(connectors: dict[str, KnowledgeBaseConnector]):
    """
    Disconnects all provided knowledge base connectors.

    Args:
        connectors (dict[str, KnowledgeBaseConnector]): A dictionary of active connectors.
    """
    logger.info("Disconnecting all knowledge base connectors...")
    for name, connector in connectors.items():
        try:
            if connector.is_connected():
                connector.disconnect()
                logger.info(f"Disconnected '{name}'.")
            else:
                logger.info(f"Connector '{name}' was not connected.")
        except Exception as e:
            logger.error(f"Error disconnecting '{name}': {e}")
    logger.info("All knowledge base connectors processed for disconnection.")


if __name__ == "__main__":
    logger.info("--- Starting Knowledge Base Connector Setup Demonstration ---")

    # Set up environment variables if needed for testing CustomAPIDBConnector
    os.environ["CUSTOM_KB_API_KEY"] = "MOCK_API_KEY_123"
    logger.info(f"Set CUSTOM_KB_API_KEY to: {os.environ['CUSTOM_KB_API_KEY']}")

    # 1. Setup all connectors based on the mock config
    active_kbs = setup_kb_connectors()

    if active_kbs:
        logger.info("\n--- Testing Active Connectors ---")
        test_query = "Grounded Token Initialization"
        for name, kb_connector in active_kbs.items():
            logger.info(f"\nQuerying '{name}' for '{test_query}':")
            try:
                results = kb_connector.query(test_query)
                logger.info(f"  Results from {name}: {results}")
            except ConnectionError as ce:
                logger.error(f"  Failed to query {name}: {ce}")
            except Exception as e:
                logger.error(f"  An unexpected error occurred while querying {name}: {e}")
    else:
        logger.warning("No active knowledge bases to test.")

    # 2. Demonstrate with a modified configuration (e.g., disabling Wikidata)
    logger.info("\n--- Demonstrating Setup with Modified Configuration (Disabling Wikidata) ---")
    modified_config = _MOCK_KB_CONNECTORS_CONFIG.copy()
    if "WIKIDATA" in modified_config:
        modified_config["WIKIDATA"]["enabled"] = False
        logger.info("Wikidata explicitly disabled in modified config.")
    if "CUSTOM_API_KB" in modified_config:
        modified_config["CUSTOM_API_KB"]["enabled"] = True # Enable custom KB for this test

    # First, disconnect any previously active connectors
    disconnect_all_kb_connectors(active_kbs)

    active_kbs_modified = setup_kb_connectors(modified_config)

    if active_kbs_modified:
        logger.info("\n--- Testing Connectors from Modified Configuration ---")
        for name, kb_connector in active_kbs_modified.items():
            logger.info(f"\nQuerying '{name}' for '{test_query}':")
            try:
                results = kb_connector.query(test_query)
                logger.info(f"  Results from {name}: {results}")
            except Exception as e:
                logger.error(f"  Error querying {name}: {e}")
    else:
        logger.warning("No active knowledge bases from modified configuration.")


    logger.info("\n--- Cleaning up: Disconnecting all active connectors ---")
    disconnect_all_kb_connectors(active_kbs_modified)

    # Clean up dummy local_domain.db if it was created
    local_db_path = os.path.join(os.path.dirname(__file__), "../data/local_domain.db")
    if os.path.exists(local_db_path):
        try:
            os.remove(local_db_path)
            logger.info(f"Cleaned up dummy database file: {local_db_path}")
        except Exception as e:
            logger.warning(f"Could not remove dummy database file {local_db_path}: {e}")

    logger.info("\n--- Knowledge Base Connector Setup Demonstration Finished ---")