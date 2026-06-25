"""
LocalDocumentStore - Shared utility for loading and searching local
text documents. Used internally by the concrete RAG module implementations.

This implements a keyword-based relevance search over local .txt files.
In future phases this can be replaced with vector embeddings and a
vector database without changing the RAG module interface.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .base_rag_module import RetrievalResult


@dataclass
class _Document:
    filename: str
    content: str
    # Lowercased, whitespace-normalized content for search
    normalized_content: str
    # Individual sections split by double newlines
    sections: list[str]


class LocalDocumentStore:
    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._documents: list[_Document] = []
        self._load_documents()

    def _load_documents(self) -> None:
        """Load all .txt files from the data directory into memory."""
        if not os.path.isdir(self._data_dir):
            print(f"[LocalDocumentStore] Directory not found: {self._data_dir}")
            return

        files = sorted(
            f for f in os.listdir(self._data_dir) if f.endswith(".txt")
        )

        for filename in files:
            file_path = os.path.join(self._data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()

            normalized_content = re.sub(r"\s+", " ", content.lower())

            # Split document into sections by double newlines for granular retrieval
            sections = [
                s.strip()
                for s in re.split(r"\n\n+", content)
                if s.strip()
            ]

            self._documents.append(
                _Document(
                    filename=filename,
                    content=content,
                    normalized_content=normalized_content,
                    sections=sections,
                )
            )

        print(
            f"[LocalDocumentStore] Loaded {len(self._documents)} documents "
            f"from {self._data_dir}"
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """
        Search for relevant sections across all documents using keyword
        matching with TF-based scoring.

        Args:
            query: User query to search for
            top_k: Number of top results to return

        Returns:
            Ranked retrieval results
        """
        # Tokenize query: lowercase, remove non-alphanumeric, skip short tokens
        query_terms = [
            t
            for t in re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
            if len(t) > 2
        ]

        if not query_terms:
            return []

        scored_results: list[RetrievalResult] = []

        for doc in self._documents:
            for section in doc.sections:
                normalized_section = re.sub(r"\s+", " ", section.lower())
                score = 0
                matched_terms = 0

                for term in query_terms:
                    # Count occurrences of each query term in the section
                    matches = re.findall(re.escape(term), normalized_section, re.IGNORECASE)
                    if matches:
                        matched_terms += 1
                        # Score based on frequency, capped to avoid single-term dominance
                        score += min(len(matches), 3)

                # Bonus for matching multiple distinct query terms
                if matched_terms > 1:
                    score *= 1 + matched_terms * 0.3

                # Normalize score to 0-1 range (approximate)
                normalized_score = min(score / (len(query_terms) * 3), 1.0)

                if normalized_score > 0.05:
                    scored_results.append(
                        RetrievalResult(
                            content=section,
                            source=doc.filename,
                            score=round(normalized_score * 1000) / 1000,
                        )
                    )

        # Sort by score descending and return top K
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:top_k]

    def get_document_count(self) -> int:
        """Returns the count of loaded documents."""
        return len(self._documents)
