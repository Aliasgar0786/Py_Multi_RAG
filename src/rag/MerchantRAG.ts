/**
 * MerchantRAG - RAG module for Merchant application queries.
 *
 * Retrieves information exclusively from the data/merchant/ directory.
 * Handles: product management, seller onboarding, merchant FAQs, store management.
 */

import * as path from "path";
import { BaseRAGModule, RAGResponse } from "./BaseRAGModule";
import { LocalDocumentStore } from "./LocalDocumentStore";

export class MerchantRAG implements BaseRAGModule {
  readonly moduleName = "merchant_rag";
  private readonly store: LocalDocumentStore;

  constructor(dataRootDir: string) {
    const merchantDataDir = path.join(dataRootDir, "merchant");
    this.store = new LocalDocumentStore(merchantDataDir);
    console.log(`[MerchantRAG] Initialized with data from: ${merchantDataDir}`);
  }

  async retrieve(query: string, topK: number = 3): Promise<RAGResponse> {
    const results = this.store.search(query, topK);

    const context = results.length > 0
      ? results
          .map(
            (r, i) =>
              `[Source: ${r.source} | Relevance: ${r.score}]\n${r.content}`
          )
          .join("\n\n")
      : "No relevant information found in the merchant knowledge base.";

    return {
      context,
      results,
      ragModule: this.moduleName,
    };
  }
}
