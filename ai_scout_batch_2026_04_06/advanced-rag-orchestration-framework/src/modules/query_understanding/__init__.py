from .base import BaseQueryProcessor
from .intent_recognizer import IntentRecognizer
from .hypothetical_doc_generator import HypotheticalDocumentGenerator

__all__ = [
    "BaseQueryProcessor",
    "IntentRecognizer",
    "HypotheticalDocumentGenerator",
]

"""
The `query_understanding` module encapsulates components responsible for
sophisticated query processing within the Advanced RAG Orchestration Framework.

This includes:
- `BaseQueryProcessor`: An abstract base class defining the interface for all
  query understanding modules.
- `IntentRecognizer`: Identifies the user's intent from the query.
- `HypotheticalDocumentGenerator`: Expands the query semantically by generating
  hypothetical documents to improve retrieval recall.
"""