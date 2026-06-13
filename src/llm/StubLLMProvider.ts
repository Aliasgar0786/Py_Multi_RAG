/**
 * StubLLMProvider - A zero-dependency, offline LLM stand-in for testing.
 *
 * Returns a formatted summary of the RAG context so you can verify the
 * full pipeline (WebSocket → Classifier → Router → RAG → response)
 * without burning any API quota.
 *
 * Enable it by setting LLM_PROVIDER=stub in your .env file.
 */

import {
  BaseLLMProvider,
  LLMGenerationOptions,
  LLMResponse,
} from "./BaseLLMProvider";

export class StubLLMProvider implements BaseLLMProvider {
  readonly providerName = "stub";

  async generateResponse(
    prompt: string,
    context?: string,
    options?: LLMGenerationOptions
  ): Promise<LLMResponse> {
    // Simulate a small delay like a real LLM would have
    await new Promise((resolve) => setTimeout(resolve, 300));

    let text: string;

    if (context && context.trim().length > 0) {
      // Extract the first ~500 chars of context to build a readable reply
      const snippet = context.length > 500 ? context.slice(0, 500) + "…" : context;
      text =
        `[Stub LLM] Here's what I found in the knowledge base for your query:\n\n` +
        `"${prompt}"\n\n` +
        `--- Retrieved Context ---\n${snippet}\n--- End Context ---\n\n` +
        `(This is a stub response. The RAG pipeline retrieved the context above. ` +
        `Switch to a real LLM provider to get AI-generated answers.)`;
    } else {
      text =
        `[Stub LLM] No context was retrieved for your query: "${prompt}". ` +
        `The RAG module did not find relevant documents. ` +
        `This may mean the query doesn't match any loaded data files.`;
    }

    return {
      text,
      provider: this.providerName,
      model: "stub-echo-v1",
    };
  }
}
