"""
ShopCardPipeline - The core orchestration engine.

This is the SINGLE ENTRY POINT that any backend (Flask, FastAPI, Django,
WebSocket, CLI, cron job, etc.) calls to process a user query.

Usage from ANY backend:

    from src_python.pipeline import ShopCardPipeline

    # Initialize once at startup
    pipeline = ShopCardPipeline()

    # Call for every user query
    result = await pipeline.process_query("consumer", "How do refunds work?")
    # result is a dict with: status, platform, intent, ragSelected,
    #                         confidence, response, sources, provider, model, timestamp

That's it. The pipeline handles:
    1. Intent Classification  (which intent? refunds, orders, monitoring...)
    2. Context Routing        (which RAG module? customer, merchant, admin)
    3. Document Retrieval     (search the knowledge base for relevant context)
    4. LLM Response           (send context + query to Gemini and get answer)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .classifier import IntentClassifier
from .router import ContextRouter
from .rag import CustomerRAG, MerchantRAG, AdminRAG
from .llm import GeminiProvider, StubLLMProvider, BaseLLMProvider
from .llm.base_llm_provider import LLMGenerationOptions

VALID_PLATFORMS = ("consumer", "merchant", "admin")

PLATFORM_LABELS = {
    "consumer": "Shopcard Customer Support Assistant",
    "merchant": "Shopcard Merchant Support Assistant",
    "admin": "Shopcard Admin Support Assistant",
}


class ShopCardPipeline:
    """
    The complete AI Chat pipeline as a single callable class.

    Any backend framework just needs to:
        1. Create ONE instance:   pipeline = ShopCardPipeline()
        2. Call it per query:     result = await pipeline.process_query(platform, query)

    Everything is initialized automatically from .env and the data/ directory.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        llm_provider_type: Optional[str] = None,
    ):
        """
        Initialize the full pipeline.

        Args:
            data_dir:          Path to the data/ folder with knowledge base .txt files.
                               Defaults to the project's data/ directory.
            llm_provider:      Optionally inject a pre-built LLM provider instance.
                               If not provided, one is created from env vars.
            gemini_api_key:    Gemini API key (overrides GEMINI_API_KEY env var).
            gemini_model:      Gemini model name (default: gemini-2.0-flash).
            llm_provider_type: "gemini" or "stub" (overrides LLM_PROVIDER env var).
        """
        # ── Resolve data directory ────────────────────────────────
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data"
            )
        self._data_dir = os.path.normpath(data_dir)

        # ── Initialize LLM Provider ──────────────────────────────
        if llm_provider is not None:
            self._llm_provider = llm_provider
        else:
            provider_type = (
                llm_provider_type
                or os.getenv("LLM_PROVIDER", "gemini")
            ).lower()

            if provider_type == "stub":
                self._llm_provider = StubLLMProvider()
            else:
                api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
                if not api_key or api_key == "your_gemini_api_key_here":
                    raise ValueError(
                        "GEMINI_API_KEY is required. Set it in .env or pass gemini_api_key=."
                    )
                self._llm_provider = GeminiProvider(api_key, gemini_model)

        # ── Initialize RAG Modules ────────────────────────────────
        customer_rag = CustomerRAG(self._data_dir)
        merchant_rag = MerchantRAG(self._data_dir)
        admin_rag = AdminRAG(self._data_dir)

        # ── Initialize Context Router ─────────────────────────────
        self._router = ContextRouter()
        self._router.register_rag(customer_rag)
        self._router.register_rag(merchant_rag)
        self._router.register_rag(admin_rag)

        # ── Initialize Intent Classifier ──────────────────────────
        self._classifier = IntentClassifier()

        print(f"[ShopCardPipeline] Ready. Data: {self._data_dir} | "
              f"LLM: {self._llm_provider.provider_name}")

    async def process_query(
        self,
        platform: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Process a single user query through the full pipeline.

        This is the ONLY function a backend developer needs to call.

        Args:
            platform: One of "consumer", "merchant", "admin"
            query:    The user's question string

        Returns:
            A dict with keys:
                status      - "success" or "error"
                platform    - echo of the input platform
                intent      - detected intent (e.g. "refunds", "monitoring")
                ragSelected - which RAG module was used
                confidence  - classification confidence (0.0 - 1.0)
                response    - the AI-generated answer text
                sources     - list of source document filenames
                provider    - LLM provider name (e.g. "gemini", "stub")
                model       - LLM model name
                timestamp   - ISO timestamp of the response
        """
        start_time = time.time()

        try:
            # ── Validate input ────────────────────────────────────
            if not platform or not query:
                return self._error_result(
                    "Missing required fields: 'platform' and 'query'."
                )

            if platform not in VALID_PLATFORMS:
                return self._error_result(
                    f'Invalid platform: "{platform}". '
                    f'Must be one of: {", ".join(VALID_PLATFORMS)}'
                )

            # ── Step 1: Intent Classification ─────────────────────
            classification = self._classifier.classify(platform, query)

            # ── Step 2: Context Routing & RAG Retrieval ───────────
            rag_response = await self._router.route(classification, query)

            # ── Step 3: LLM Response Generation ───────────────────
            llm_response = await self._llm_provider.generate_response(
                query,
                rag_response.context,
                LLMGenerationOptions(
                    system_prompt=(
                        f"You are the {PLATFORM_LABELS[platform]}. "
                        f"Answer the user's question based ONLY on the provided context. "
                        f"Be helpful, concise, and professional. If the context does not "
                        f"contain enough information, clearly state that you don't have "
                        f"that information available."
                    ),
                    temperature=0.3,
                ),
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            print(
                f"[ShopCardPipeline] [OK] {platform}/{classification.intent} "
                f"in {elapsed_ms}ms ({len(rag_response.results)} sources)"
            )

            # ── Return structured result ──────────────────────────
            return {
                "status": "success",
                "platform": platform,
                "intent": classification.intent,
                "ragSelected": classification.target_rag,
                "confidence": classification.confidence,
                "response": llm_response.text,
                "sources": list(set(r.source for r in rag_response.results)),
                "provider": llm_response.provider,
                "model": llm_response.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as error:
            print(f"[ShopCardPipeline] Pipeline error: {error}")
            return self._error_result(f"Internal error: {error}")

    @staticmethod
    def _error_result(message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "error": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
