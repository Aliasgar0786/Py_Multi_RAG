"""
ContextRouter - Routes classified queries to the correct RAG module.

Maintains a registry of RAG modules and selects the appropriate one
based on the classification result from the IntentClassifier.
"""

from __future__ import annotations

from ..rag.base_rag_module import BaseRAGModule, RAGResponse
from ..classifier.intent_classifier import ClassificationResult


class ContextRouter:
    def __init__(self):
        self._rag_modules: dict[str, BaseRAGModule] = {}

    def register_rag(self, module: BaseRAGModule) -> None:
        """Register a RAG module with the router."""
        self._rag_modules[module.module_name] = module
        print(f"[ContextRouter] Registered RAG module: {module.module_name}")

    async def route(
        self,
        classification: ClassificationResult,
        query: str,
    ) -> RAGResponse:
        """
        Route a classified query to the correct RAG module and retrieve context.

        Args:
            classification: The result from the IntentClassifier
            query: The original user query

        Returns:
            RAG response with retrieved context and sources

        Raises:
            RuntimeError: If the target RAG module is not registered
        """
        target_module = self._rag_modules.get(classification.target_rag)

        if target_module is None:
            available = ", ".join(self._rag_modules.keys())
            raise RuntimeError(
                f"[ContextRouter] RAG module not found: {classification.target_rag}. "
                f"Available modules: {available}"
            )

        print(
            f"[ContextRouter] Routing query to: {classification.target_rag} "
            f"(intent: {classification.intent}, confidence: {classification.confidence})"
        )

        rag_response = await target_module.retrieve(query)
        return rag_response

    def get_registered_modules(self) -> list[str]:
        """List all registered RAG modules."""
        return list(self._rag_modules.keys())
