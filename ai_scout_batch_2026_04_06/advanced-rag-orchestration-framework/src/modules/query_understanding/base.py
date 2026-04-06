import abc
from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.models import Query, PipelineState
from src.core.interfaces import QueryProcessor


class BaseQueryUnderstandingModule(QueryProcessor):
    """
    Abstract Base Class for all Query Understanding modules.

    Query understanding modules process the initial user query to enhance it,
    rewrite it, identify intent, extract entities, or generate hypothetical documents.
    All concrete query understanding modules must inherit from this class and
    implement the `process_query` method.
    """

    @abstractmethod
    def process_query(self, query: Query, state: Optional[PipelineState] = None) -> Query:
        """
        Processes a raw user query, enhancing it with additional context,
        rewriting it, or generating alternative query forms.

        Implementations should focus on a specific aspect of query understanding,
        such as intent recognition, entity extraction, or query expansion.

        Args:
            query: The initial or partially processed Query object.
            state: Optional current pipeline state, which might contain conversational history
                   or previous processing results.

        Returns:
            An enriched or transformed Query object. This could include:
            - Original query text with added intent/entities.
            - Rewritten query text.
            - A query object indicating a specific follow-up action.
            - Potentially, a list of sub-queries for further processing (though
              the return type is a single Query, a list could be embedded in metadata
              or handled by a higher-level orchestrator if this module creates multiple).
              For now, we assume a single, refined Query is returned.

        Raises:
            QueryUnderstandingError: If an error occurs during query processing.
        """
        raise NotImplementedError("Subclasses must implement the process_query method.")

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.__class__.__name__}()"