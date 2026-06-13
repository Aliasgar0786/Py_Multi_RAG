/**
 * ContextRouter - Routes classified queries to the correct RAG module.
 *
 * Maintains a registry of RAG modules and selects the appropriate one
 * based on the classification result from the IntentClassifier.
 */

import { BaseRAGModule, RAGResponse } from "../rag/BaseRAGModule";
import { ClassificationResult } from "../classifier/IntentClassifier";

export class ContextRouter {
  private ragModules: Map<string, BaseRAGModule> = new Map();

  /**
   * Register a RAG module with the router.
   */
  registerRAG(module: BaseRAGModule): void {
    this.ragModules.set(module.moduleName, module);
    console.log(`[ContextRouter] Registered RAG module: ${module.moduleName}`);
  }

  /**
   * Route a classified query to the correct RAG module and retrieve context.
   *
   * @param classification - The result from the IntentClassifier
   * @param query - The original user query
   * @returns RAG response with retrieved context and sources
   * @throws Error if the target RAG module is not registered
   */
  async route(
    classification: ClassificationResult,
    query: string
  ): Promise<RAGResponse> {
    const targetModule = this.ragModules.get(classification.targetRAG);

    if (!targetModule) {
      throw new Error(
        `[ContextRouter] RAG module not found: ${classification.targetRAG}. ` +
          `Available modules: ${Array.from(this.ragModules.keys()).join(", ")}`
      );
    }

    console.log(
      `[ContextRouter] Routing query to: ${classification.targetRAG} ` +
        `(intent: ${classification.intent}, confidence: ${classification.confidence})`
    );

    const ragResponse = await targetModule.retrieve(query);
    return ragResponse;
  }

  /**
   * List all registered RAG modules.
   */
  getRegisteredModules(): string[] {
    return Array.from(this.ragModules.keys());
  }
}
