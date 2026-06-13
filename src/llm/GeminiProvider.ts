/**
 * GeminiProvider - Concrete implementation of BaseLLMProvider using
 * the Google Generative AI (Gemini) SDK.
 *
 * This is the only file in the codebase that imports @google/generative-ai.
 * Swapping to a different provider only requires creating a new implementation
 * of BaseLLMProvider — no other code changes needed.
 *
 * Includes automatic retry with exponential backoff for rate-limit (429) errors.
 */

import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai";
import {
  BaseLLMProvider,
  LLMGenerationOptions,
  LLMResponse,
} from "./BaseLLMProvider";

export class GeminiProvider implements BaseLLMProvider {
  readonly providerName = "gemini";
  private readonly model: GenerativeModel;
  private readonly modelName: string;
  private readonly maxRetries: number;

  constructor(
    apiKey: string,
    modelName: string = "gemini-2.0-flash",
    maxRetries: number = 5
  ) {
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is required to initialize GeminiProvider");
    }
    const genAI = new GoogleGenerativeAI(apiKey);
    this.modelName = modelName;
    this.model = genAI.getGenerativeModel({ model: modelName });
    this.maxRetries = maxRetries;
  }

  async generateResponse(
    prompt: string,
    context?: string,
    options?: LLMGenerationOptions
  ): Promise<LLMResponse> {
    const systemPrompt =
      options?.systemPrompt ||
      "You are a helpful Shopcard assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information to answer, say so clearly. Do not make up information.";

    // Build the full prompt with context
    let fullPrompt = `${systemPrompt}\n\n`;

    if (context) {
      fullPrompt += `--- Retrieved Context ---\n${context}\n--- End Context ---\n\n`;
    }

    fullPrompt += `User Question: ${prompt}\n\nAnswer:`;

    return this.executeWithRetry(fullPrompt, options);
  }

  /**
   * Execute the LLM call with automatic retry and exponential backoff
   * for rate-limit (429) errors.
   */
  private async executeWithRetry(
    fullPrompt: string,
    options?: LLMGenerationOptions
  ): Promise<LLMResponse> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await this.model.generateContent({
          contents: [{ role: "user", parts: [{ text: fullPrompt }] }],
          generationConfig: {
            maxOutputTokens: options?.maxTokens ?? 1024,
            temperature: options?.temperature ?? 0.3,
          },
        });

        const response = result.response;
        const text = response.text();

        return {
          text,
          provider: this.providerName,
          model: this.modelName,
        };
      } catch (error: any) {
        lastError = error;
        const errorMessage: string = error.message || "";

        // Check if this is a rate-limit error (429)
        if (errorMessage.includes("429") || errorMessage.includes("Too Many Requests") || errorMessage.includes("quota")) {
          if (attempt < this.maxRetries) {
            // Parse server-suggested retry delay, or use exponential backoff
            const serverDelay = this.parseRetryDelay(errorMessage);
            const backoffDelay = Math.pow(2, attempt + 1) * 1000; // 2s, 4s, 8s, 16s, 32s
            const waitMs = serverDelay ? (serverDelay * 1000) + 1000 : backoffDelay;

            console.warn(
              `[GeminiProvider] Rate limited (attempt ${attempt + 1}/${this.maxRetries}). ` +
              `Retrying in ${(waitMs / 1000).toFixed(1)}s...`
            );

            await this.sleep(waitMs);
            continue;
          }

          console.error(
            `[GeminiProvider] Rate limit exceeded after ${this.maxRetries} retries. ` +
            `Your free-tier quota may be exhausted. Check: https://ai.google.dev/gemini-api/docs/rate-limits`
          );
        } else {
          // Non-retryable error — fail immediately
          console.error(`[GeminiProvider] Error generating response:`, errorMessage);
          throw new Error(`LLM generation failed: ${errorMessage}`);
        }
      }
    }

    throw new Error(
      `LLM generation failed after ${this.maxRetries} retries: ${lastError?.message}`
    );
  }

  /**
   * Try to extract the server-suggested retry delay (in seconds) from the error message.
   * Looks for patterns like "Please retry in 39.226700788s" or "retryDelay":"39s"
   */
  private parseRetryDelay(errorMessage: string): number | null {
    // Match "Please retry in <N>s" or "retryDelay":"<N>s"
    const patterns = [
      /retry in ([\d.]+)s/i,
      /retryDelay["\s:]+(\d+)s/i,
    ];

    for (const pattern of patterns) {
      const match = errorMessage.match(pattern);
      if (match) {
        const seconds = parseFloat(match[1]);
        if (!isNaN(seconds) && seconds > 0 && seconds < 120) {
          return Math.ceil(seconds);
        }
      }
    }
    return null;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
