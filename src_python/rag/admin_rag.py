"""
AdminRAG - RAG module for Admin application queries.

Retrieves information exclusively from the data/admin/ directory.
Handles: platform administration, internal documentation, monitoring, workflows.
"""

from __future__ import annotations

import os

from .base_rag_module import BaseRAGModule, RAGResponse
from .local_document_store import LocalDocumentStore


class AdminRAG(BaseRAGModule):
    def __init__(self, data_root_dir: str):
        admin_data_dir = os.path.join(data_root_dir, "admin")
        self._store = LocalDocumentStore(admin_data_dir)
        print(f"[AdminRAG] Initialized with data from: {admin_data_dir}")

    @property
    def module_name(self) -> str:
        return "admin_rag"

    async def retrieve(self, query: str, top_k: int = 3) -> RAGResponse:
        results = self._store.search(query, top_k)

        if results:
            context = "\n\n".join(
                f"[Source: {r.source} | Relevance: {r.score}]\n{r.content}"
                for r in results
            )
        else:
            context = "No relevant information found in the admin knowledge base."

        return RAGResponse(
            context=context,
            results=results,
            rag_module=self.module_name,
        )
