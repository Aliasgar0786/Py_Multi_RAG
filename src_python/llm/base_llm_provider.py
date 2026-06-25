"""
BaseLLMProvider - Abstract interface for LLM operations.

All LLM providers (Gemini, OpenAI, Claude, Grok) must implement
this interface. The rest of the system depends ONLY on this
abstraction, never on a concrete SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMGenerationOptions:
    """Optional parameters for LLM generation."""
    # Maximum tokens in the response
    max_tokens: Optional[int] = None
    # Sampling temperature (0.0 - 1.0)
    temperature: Optional[float] = None
    # System-level instruction to guide the LLM behavior
    system_prompt: Optional[str] = None


@dataclass
class LLMResponse:
    """The result of an LLM generation call."""
    # The generated text
    text: str
    # Provider name that generated this response
    provider: str
    # Model used for generation
    model: str


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Concrete implementations must provide `provider_name` and `generate_response`.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g., "gemini", "openai", "claude")."""
        ...

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        options: Optional[LLMGenerationOptions] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM given a user prompt and optional
        retrieved context from the RAG pipeline.

        Args:
            prompt: The user's query
            context: Retrieved context from the RAG module
            options: Optional generation parameters

        Returns:
            The generated response
        """
        ...
