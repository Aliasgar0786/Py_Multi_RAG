"""
BaseRAGModule - Abstract interface for Retrieval-Augmented Generation modules.

Each RAG module implementation has its own isolated document collection
and retrieval pipeline. Consumer queries can never access Merchant knowledge,
and vice versa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalResult:
    """A single retrieval result from the knowledge base."""
    # The retrieved text content
    content: str
    # Source document identifier
    source: str
    # Relevance score (0.0 - 1.0)
    score: float


@dataclass
class RAGResponse:
    """The output of a RAG module's retrieve-and-format pipeline."""
    # The combined context string ready for LLM consumption
    context: str
    # Individual retrieval results with sources
    results: list[RetrievalResult]
    # Name of the RAG module that produced this response
    rag_module: str


class BaseRAGModule(ABC):
    """
    Abstract base for RAG modules.
    Each implementation has isolated document collections.
    """

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Unique name identifying this RAG module."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 3) -> RAGResponse:
        """
        Retrieve relevant documents from this module's isolated knowledge base
        and return formatted context for the LLM.

        Args:
            query: The user's query to search against
            top_k: Maximum number of results to return (default: 3)

        Returns:
            RAG response with context and source documents
        """
        ...
