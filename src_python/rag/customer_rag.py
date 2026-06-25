"""
CustomerRAG - RAG module for Consumer application queries.

Retrieves information exclusively from the data/customer/ directory.
Handles: shopping help, refund questions, order guidance, consumer FAQs.
"""

from __future__ import annotations

import os

from .base_rag_module import BaseRAGModule, RAGResponse
from .local_document_store import LocalDocumentStore


class CustomerRAG(BaseRAGModule):
    def __init__(self, data_root_dir: str):
        customer_data_dir = os.path.join(data_root_dir, "customer")
        self._store = LocalDocumentStore(customer_data_dir)
        print(f"[CustomerRAG] Initialized with data from: {customer_data_dir}")

    @property
    def module_name(self) -> str:
        return "customer_rag"

    async def retrieve(self, query: str, top_k: int = 3) -> RAGResponse:
        results = self._store.search(query, top_k)

        # Format results into a combined context string for the LLM
        if results:
            context = "\n\n".join(
                f"[Source: {r.source} | Relevance: {r.score}]\n{r.content}"
                for r in results
            )
        else:
            context = "No relevant information found in the customer knowledge base."

        return RAGResponse(
            context=context,
            results=results,
            rag_module=self.module_name,
        )
