/**
 * LocalDocumentStore - Shared utility for loading and searching local
 * text documents. Used internally by the concrete RAG module implementations.
 *
 * This implements a keyword-based relevance search over local .txt files.
 * In future phases this can be replaced with vector embeddings and a
 * vector database without changing the RAG module interface.
 */

import * as fs from "fs";
import * as path from "path";
import { RetrievalResult } from "./BaseRAGModule";

interface Document {
  filename: string;
  content: string;
  /** Lowercased, whitespace-normalized content for search */
  normalizedContent: string;
  /** Individual sections split by double newlines */
  sections: string[];
}

export class LocalDocumentStore {
  private documents: Document[] = [];
  private readonly dataDir: string;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.loadDocuments();
  }

  /**
   * Load all .txt files from the data directory into memory.
   */
  private loadDocuments(): void {
    if (!fs.existsSync(this.dataDir)) {
      console.warn(`[LocalDocumentStore] Directory not found: ${this.dataDir}`);
      return;
    }

    const files = fs.readdirSync(this.dataDir).filter((f) => f.endsWith(".txt"));

    for (const file of files) {
      const filePath = path.join(this.dataDir, file);
      const content = fs.readFileSync(filePath, "utf-8");
      const normalizedContent = content.toLowerCase().replace(/\s+/g, " ");

      // Split document into sections by double newlines for granular retrieval
      const sections = content
        .split(/\n\n+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      this.documents.push({
        filename: file,
        content,
        normalizedContent,
        sections,
      });
    }

    console.log(
      `[LocalDocumentStore] Loaded ${this.documents.length} documents from ${this.dataDir}`
    );
  }

  /**
   * Search for relevant sections across all documents using keyword
   * matching with TF-based scoring.
   *
   * @param query - User query to search for
   * @param topK - Number of top results to return
   * @returns Ranked retrieval results
   */
  search(query: string, topK: number = 3): RetrievalResult[] {
    const queryTerms = query
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, "")
      .split(/\s+/)
      .filter((t) => t.length > 2); // Skip very short tokens

    if (queryTerms.length === 0) {
      return [];
    }

    const scoredResults: RetrievalResult[] = [];

    for (const doc of this.documents) {
      for (const section of doc.sections) {
        const normalizedSection = section.toLowerCase().replace(/\s+/g, " ");
        let score = 0;
        let matchedTerms = 0;

        for (const term of queryTerms) {
          // Count occurrences of each query term in the section
          const regex = new RegExp(term, "gi");
          const matches = normalizedSection.match(regex);
          if (matches) {
            matchedTerms++;
            // Score based on frequency, capped to avoid single-term dominance
            score += Math.min(matches.length, 3);
          }
        }

        // Bonus for matching multiple distinct query terms
        if (matchedTerms > 1) {
          score *= 1 + matchedTerms * 0.3;
        }

        // Normalize score to 0-1 range (approximate)
        const normalizedScore = Math.min(score / (queryTerms.length * 3), 1.0);

        if (normalizedScore > 0.05) {
          scoredResults.push({
            content: section,
            source: doc.filename,
            score: Math.round(normalizedScore * 1000) / 1000,
          });
        }
      }
    }

    // Sort by score descending and return top K
    return scoredResults
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Returns the count of loaded documents.
   */
  getDocumentCount(): number {
    return this.documents.length;
  }
}
