/**
 * AdminRAG - RAG module for Admin application queries.
 *
 * Retrieves information exclusively from the data/admin/ directory.
 * Handles: platform administration, internal documentation, monitoring, workflows.
 */

import * as path from "path";
import { BaseRAGModule, RAGResponse } from "./BaseRAGModule";
import { LocalDocumentStore } from "./LocalDocumentStore";

export class AdminRAG implements BaseRAGModule {
  readonly moduleName = "admin_rag";
  private readonly store: LocalDocumentStore;

  constructor(dataRootDir: string) {
    const adminDataDir = path.join(dataRootDir, "admin");
    this.store = new LocalDocumentStore(adminDataDir);
    console.log(`[AdminRAG] Initialized with data from: ${adminDataDir}`);
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
      : "No relevant information found in the admin knowledge base.";

    return {
      context,
      results,
      ragModule: this.moduleName,
    };
  }
}
