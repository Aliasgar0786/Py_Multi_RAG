"""
StubLLMProvider - A zero-dependency, offline LLM stand-in for testing.

Returns a formatted summary of the RAG context so you can verify the
full pipeline (WebSocket → Classifier → Router → RAG → response)
without burning any API quota.

Enable it by setting LLM_PROVIDER=stub in your .env file.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .base_llm_provider import BaseLLMProvider, LLMGenerationOptions, LLMResponse


class StubLLMProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        options: Optional[LLMGenerationOptions] = None,
    ) -> LLMResponse:
        # Simulate a small delay like a real LLM would have
        await asyncio.sleep(0.3)

        if context and context.strip():
            # Extract the first ~500 chars of context to build a readable reply
            snippet = context[:500] + "…" if len(context) > 500 else context
            text = (
                f'[Stub LLM] Here\'s what I found in the knowledge base for your query:\n\n'
                f'"{prompt}"\n\n'
                f'--- Retrieved Context ---\n{snippet}\n--- End Context ---\n\n'
                f'(This is a stub response. The RAG pipeline retrieved the context above. '
                f'Switch to a real LLM provider to get AI-generated answers.)'
            )
        else:
            text = (
                f'[Stub LLM] No context was retrieved for your query: "{prompt}". '
                f'The RAG module did not find relevant documents. '
                f'This may mean the query doesn\'t match any loaded data files.'
            )

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model="stub-echo-v1",
        )
