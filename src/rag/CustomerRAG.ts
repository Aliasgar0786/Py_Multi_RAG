/**
 * CustomerRAG - RAG module for Consumer application queries.
 *
 * Retrieves information exclusively from the data/customer/ directory.
 * Handles: shopping help, refund questions, order guidance, consumer FAQs.
 */

import * as path from "path";
import { BaseRAGModule, RAGResponse } from "./BaseRAGModule";
import { LocalDocumentStore } from "./LocalDocumentStore";

export class CustomerRAG implements BaseRAGModule {
  readonly moduleName = "customer_rag";
  private readonly store: LocalDocumentStore;

  constructor(dataRootDir: string) {
    const customerDataDir = path.join(dataRootDir, "customer");
    this.store = new LocalDocumentStore(customerDataDir);
    console.log(`[CustomerRAG] Initialized with data from: ${customerDataDir}`);
  }

  async retrieve(query: string, topK: number = 3): Promise<RAGResponse> {
    const results = this.store.search(query, topK);

    // Format results into a combined context string for the LLM
    const context = results.length > 0
      ? results
          .map(
            (r, i) =>
              `[Source: ${r.source} | Relevance: ${r.score}]\n${r.content}`
          )
          .join("\n\n")
      : "No relevant information found in the customer knowledge base.";

    return {
      context,
      results,
      ragModule: this.moduleName,
    };
  }
}
