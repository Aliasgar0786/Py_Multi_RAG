/**
 * BaseRAGModule - Abstract interface for Retrieval-Augmented Generation modules.
 *
 * Each RAG module implementation has its own isolated document collection
 * and retrieval pipeline. Consumer queries can never access Merchant knowledge,
 * and vice versa.
 */

/** A single retrieval result from the knowledge base */
export interface RetrievalResult {
  /** The retrieved text content */
  content: string;
  /** Source document identifier */
  source: string;
  /** Relevance score (0.0 - 1.0) */
  score: number;
}

/** The output of a RAG module's retrieve-and-format pipeline */
export interface RAGResponse {
  /** The combined context string ready for LLM consumption */
  context: string;
  /** Individual retrieval results with sources */
  results: RetrievalResult[];
  /** Name of the RAG module that produced this response */
  ragModule: string;
}

export interface BaseRAGModule {
  /** Unique name identifying this RAG module */
  readonly moduleName: string;

  /**
   * Retrieve relevant documents from this module's isolated knowledge base
   * and return formatted context for the LLM.
   *
   * @param query - The user's query to search against
   * @param topK - Maximum number of results to return (default: 3)
   * @returns RAG response with context and source documents
   */
  retrieve(query: string, topK?: number): Promise<RAGResponse>;
}
