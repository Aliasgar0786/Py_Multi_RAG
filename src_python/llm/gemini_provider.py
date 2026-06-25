"""
GeminiProvider - Concrete implementation of BaseLLMProvider using
the Google GenAI SDK (google.genai).

This is the only file in the codebase that imports google.genai.
Swapping to a different provider only requires creating a new implementation
of BaseLLMProvider — no other code changes needed.

Includes automatic retry with exponential backoff for rate-limit (429) errors.
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import Optional

from google import genai
from google.genai import types

from .base_llm_provider import BaseLLMProvider, LLMGenerationOptions, LLMResponse


class GeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_retries: int = 5,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiProvider")

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        options: Optional[LLMGenerationOptions] = None,
    ) -> LLMResponse:
        system_prompt = (
            (options.system_prompt if options and options.system_prompt else None)
            or (
                "You are a helpful Shopcard assistant. Answer the user's question "
                "based ONLY on the provided context. If the context does not contain "
                "enough information to answer, say so clearly. Do not make up information."
            )
        )

        # Build the full prompt with context
        full_prompt = f"{system_prompt}\n\n"

        if context:
            full_prompt += f"--- Retrieved Context ---\n{context}\n--- End Context ---\n\n"

        full_prompt += f"User Question: {prompt}\n\nAnswer:"

        return await self._execute_with_retry(full_prompt, options)

    async def _execute_with_retry(
        self,
        full_prompt: str,
        options: Optional[LLMGenerationOptions] = None,
    ) -> LLMResponse:
        """
        Execute the LLM call with automatic retry and exponential backoff
        for rate-limit (429) errors.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                # google.genai's generate_content is synchronous;
                # run it in a thread to keep the event loop free.
                result = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=(
                            options.max_tokens if options and options.max_tokens else 1024
                        ),
                        temperature=(
                            options.temperature if options and options.temperature is not None else 0.3
                        ),
                    ),
                )

                text = result.text

                return LLMResponse(
                    text=text,
                    provider=self.provider_name,
                    model=self._model_name,
                )

            except Exception as error:
                last_error = error
                error_message = str(error)

                # Check if this is a rate-limit error (429)
                if "429" in error_message or "Too Many Requests" in error_message or "quota" in error_message:
                    if attempt < self._max_retries:
                        # Parse server-suggested retry delay, or use exponential backoff
                        server_delay = self._parse_retry_delay(error_message)
                        backoff_delay = math.pow(2, attempt + 1)  # 2s, 4s, 8s, 16s, 32s
                        wait_s = (server_delay + 1) if server_delay else backoff_delay

                        print(
                            f"[GeminiProvider] Rate limited (attempt {attempt + 1}/"
                            f"{self._max_retries}). Retrying in {wait_s:.1f}s..."
                        )

                        await asyncio.sleep(wait_s)
                        continue

                    print(
                        f"[GeminiProvider] Rate limit exceeded after {self._max_retries} "
                        f"retries. Your free-tier quota may be exhausted. "
                        f"Check: https://ai.google.dev/gemini-api/docs/rate-limits"
                    )
                else:
                    # Non-retryable error — fail immediately
                    print(f"[GeminiProvider] Error generating response: {error_message}")
                    raise RuntimeError(f"LLM generation failed: {error_message}") from error

        raise RuntimeError(
            f"LLM generation failed after {self._max_retries} retries: {last_error}"
        )

    @staticmethod
    def _parse_retry_delay(error_message: str) -> Optional[float]:
        """
        Try to extract the server-suggested retry delay (in seconds) from the error message.
        Looks for patterns like "Please retry in 39.226700788s" or "retryDelay":"39s"
        """
        patterns = [
            r"retry in ([\d.]+)s",
            r'retryDelay["\s:]+(\d+)s',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                seconds = float(match.group(1))
                if 0 < seconds < 120:
                    return math.ceil(seconds)
        return None
