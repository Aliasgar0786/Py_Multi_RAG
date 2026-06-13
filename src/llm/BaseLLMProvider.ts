/**
 * BaseLLMProvider - Abstract interface for LLM operations.
 *
 * All LLM providers (Gemini, OpenAI, Claude, Grok) must implement
 * this interface. The rest of the system depends ONLY on this
 * abstraction, never on a concrete SDK.
 */

export interface LLMGenerationOptions {
  /** Maximum tokens in the response */
  maxTokens?: number;
  /** Sampling temperature (0.0 - 1.0) */
  temperature?: number;
  /** System-level instruction to guide the LLM behavior */
  systemPrompt?: string;
}

export interface LLMResponse {
  /** The generated text */
  text: string;
  /** Provider name that generated this response */
  provider: string;
  /** Model used for generation */
  model: string;
}

export interface BaseLLMProvider {
  /** Provider identifier (e.g., "gemini", "openai", "claude") */
  readonly providerName: string;

  /**
   * Generate a response from the LLM given a user prompt and optional
   * retrieved context from the RAG pipeline.
   *
   * @param prompt - The user's query
   * @param context - Retrieved context from the RAG module
   * @param options - Optional generation parameters
   * @returns The generated response
   */
  generateResponse(
    prompt: string,
    context?: string,
    options?: LLMGenerationOptions
  ): Promise<LLMResponse>;
}
