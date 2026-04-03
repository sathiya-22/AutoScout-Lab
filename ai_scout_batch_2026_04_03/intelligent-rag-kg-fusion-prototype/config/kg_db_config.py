```python
import os
from typing import Optional

class KG_DB_CONFIG:
    """
    Configuration settings for the Knowledge Graph database.
    This class provides a centralized place for all KG-related database parameters,
    supporting different types of graph databases and allowing environment variable
    overrides for flexible deployment.
    """

    # --- Common Knowledge Graph Database Settings ---
    # Specifies the type of KG database being used.
    # Examples: 'NEO4J', 'VIRTUOSO', 'DGRAPH', 'AMAZON_NEPTUNE', 'RDF_TRIPLE_STORE'
    DB_TYPE: str = os.getenv("KG_DB_TYPE", "NEO4J").upper()

    # Generic host address for the KG database.
    HOST: str = os.getenv("KG_DB_HOST", "localhost")

    # Generic port number for the KG database connection.
    # Default is for Neo4j Bolt, but can be overridden.
    PORT: int = int(os.getenv("KG_DB_PORT", "7687"))

    # Generic username for database authentication.
    USERNAME: str = os.getenv("KG_DB_USERNAME", "neo4j")

    # Generic password for database authentication.
    # IMPORTANT: Use strong, secret passwords in production environments.
    # Consider using a secret management system.
    PASSWORD: str = os.getenv("KG_DB_PASSWORD", "password")

    # --- Neo4j Specific Settings (if DB_TYPE is NEO4J) ---
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    # Neo4j URI usually combines schema, host, and port.
    NEO4J_URI: str = os.getenv("NEO4J_URI", f"bolt://{HOST}:{PORT}")
    NEO4J_TIMEOUT_SECONDS: int = int(os.getenv("NEO4J_TIMEOUT_SECONDS", "30"))

    # --- RDF Triple Store / SPARQL Endpoint Specific Settings (e.g., Virtuoso, Jena Fuseki) ---
    SPARQL_QUERY_ENDPOINT: str = os.getenv("KG_SPARQL_QUERY_ENDPOINT", "http://localhost:8890/sparql")
    SPARQL_UPDATE_ENDPOINT: str = os.getenv("KG_SPARQL_UPDATE_ENDPOINT", "http://localhost:8890/sparql")
    # For named graphs in RDF stores.
    RDF_GRAPH_URI: Optional[str] = os.getenv("KG_RDF_GRAPH_URI", None)

    # --- Dgraph Specific Settings (if DB_TYPE is DGRAPH) ---
    DGRAPH_GRPC_ENDPOINT: str = os.getenv("DGRAPH_GRPC_ENDPOINT", "localhost:9080")
    DGRAPH_ALPHA_HTTP_ENDPOINT: str = os.getenv("DGRAPH_ALPHA_HTTP_ENDPOINT", "http://localhost:8080")
    DGRAPH_AUTH_TOKEN: Optional[str] = os.getenv("DGRAPH_AUTH_TOKEN", None)

    # --- Amazon Neptune Specific Settings (if DB_TYPE is AMAZON_NEPTUNE) ---
    NEPTUNE_ENDPOINT: str = os.getenv("NEPTUNE_ENDPOINT", "localhost:8182")
    NEPTUNE_PORT: int = int(os.getenv("NEPTUNE_PORT", "8182"))
    NEPTUNE_PROTOCOL: str = os.getenv("NEPTUNE_PROTOCOL", "wss") # wss for Gremlin, http for SPARQL
    NEPTUNE_REGION: Optional[str] = os.getenv("NEPTUNE_REGION", None) # e.g., 'us-east-1'

    @staticmethod
    def get_neo4j_config() -> dict:
        """Returns Neo4j specific configuration as a dictionary."""
        return {
            "uri": KG_DB_CONFIG.NEO4J_URI,
            "username": KG_DB_CONFIG.USERNAME,
            "password": KG_DB_CONFIG.PASSWORD,
            "database": KG_DB_CONFIG.NEO4J_DATABASE,
            "timeout": KG_DB_CONFIG.NEO4J_TIMEOUT_SECONDS,
        }

    @staticmethod
    def get_sparql_config() -> dict:
        """Returns SPARQL triple store specific configuration as a dictionary."""
        return {
            "query_endpoint": KG_DB_CONFIG.SPARQL_QUERY_ENDPOINT,
            "update_endpoint": KG_DB_CONFIG.SPARQL_UPDATE_ENDPOINT,
            "graph_uri": KG_DB_CONFIG.RDF_GRAPH_URI,
            # SPARQL endpoints might require HTTP Basic Auth, using generic credentials
            "username": KG_DB_CONFIG.USERNAME,
            "password": KG_DB_CONFIG.PASSWORD,
        }

    @staticmethod
    def get_dgraph_config() -> dict:
        """Returns Dgraph specific configuration as a dictionary."""
        return {
            "grpc_endpoint": KG_DB_CONFIG.DGRAPH_GRPC_ENDPOINT,
            "http_endpoint": KG_DB_CONFIG.DGRAPH_ALPHA_HTTP_ENDPOINT,
            "auth_token": KG_DB_CONFIG.DGRAPH_AUTH_TOKEN,
        }

    @staticmethod
    def get_neptune_config() -> dict:
        """Returns Amazon Neptune specific configuration as a dictionary."""
        return {
            "endpoint": KG_DB_CONFIG.NEPTUNE_ENDPOINT,
            "port": KG_DB_CONFIG.NEPTUNE_PORT,
            "protocol": KG_DB_CONFIG.NEPTUNE_PROTOCOL,
            "region": KG_DB_CONFIG.NEPTUNE_REGION,
            "username": KG_DB_CONFIG.USERNAME, # Some Neptune setups use IAM roles, others might use basic auth
            "password": KG_DB_CONFIG.PASSWORD,
        }

    @staticmethod
    def get_active_config() -> dict:
        """
        Retrieves the configuration dictionary for the currently specified KG_DB_TYPE.
        Raises a ValueError if the DB_TYPE is not supported or recognized.
        """
        if KG_DB_CONFIG.DB_TYPE == "NEO4J":
            return KG_DB_CONFIG.get_neo4j_config()
        elif KG_DB_CONFIG.DB_TYPE in ["VIRTUOSO", "JENA_FUSEKI", "RDF_TRIPLE_STORE"]:
            return KG_DB_CONFIG.get_sparql_config()
        elif KG_DB_CONFIG.DB_TYPE == "DGRAPH":
            return KG_DB_CONFIG.get_dgraph_config()
        elif KG_DB_CONFIG.DB_TYPE == "AMAZON_NEPTUNE":
            return KG_DB_CONFIG.get_neptune_config()
        else:
            raise ValueError(f"Unsupported KG_DB_TYPE specified: '{KG_DB_CONFIG.DB_TYPE}'. "
                             "Please check KG_DB_TYPE environment variable or config defaults.")

```