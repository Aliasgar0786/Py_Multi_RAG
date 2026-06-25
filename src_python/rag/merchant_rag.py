"""
MerchantRAG - RAG module for Merchant application queries.

Retrieves information exclusively from the data/merchant/ directory.
Handles: product management, seller onboarding, merchant FAQs, store management.
"""

from __future__ import annotations

import os

from .base_rag_module import BaseRAGModule, RAGResponse
from .local_document_store import LocalDocumentStore


class MerchantRAG(BaseRAGModule):
    def __init__(self, data_root_dir: str):
        merchant_data_dir = os.path.join(data_root_dir, "merchant")
        self._store = LocalDocumentStore(merchant_data_dir)
        print(f"[MerchantRAG] Initialized with data from: {merchant_data_dir}")

    @property
    def module_name(self) -> str:
        return "merchant_rag"

    async def retrieve(self, query: str, top_k: int = 3) -> RAGResponse:
        results = self._store.search(query, top_k)

        if results:
            context = "\n\n".join(
                f"[Source: {r.source} | Relevance: {r.score}]\n{r.content}"
                for r in results
            )
        else:
            context = "No relevant information found in the merchant knowledge base."

        return RAGResponse(
            context=context,
            results=results,
            rag_module=self.module_name,
        )
