/**
 * WebSocketServer - Handles WebSocket connections and orchestrates
 * the full chat pipeline: classify → route → retrieve → generate.
 *
 * Message Protocol:
 *
 * Client sends:
 * {
 *   "platform": "consumer" | "merchant" | "admin",
 *   "query": "How do refunds work?"
 * }
 *
 * Server responds:
 * {
 *   "status": "success",
 *   "platform": "consumer",
 *   "intent": "refunds",
 *   "ragSelected": "customer_rag",
 *   "confidence": 0.75,
 *   "response": "Based on our policy...",
 *   "sources": ["refunds.txt"],
 *   "provider": "gemini",
 *   "model": "gemini-2.0-flash",
 *   "timestamp": "2026-06-12T12:00:00.000Z"
 * }
 */

import { WebSocketServer as WSServer, WebSocket } from "ws";
import { IntentClassifier, Platform } from "../classifier/IntentClassifier";
import { ContextRouter } from "../router/ContextRouter";
import { BaseLLMProvider } from "../llm/BaseLLMProvider";

interface IncomingMessage {
  platform: Platform;
  query: string;
}

interface OutgoingMessage {
  status: "success" | "error";
  platform?: string;
  intent?: string;
  ragSelected?: string;
  confidence?: number;
  response?: string;
  sources?: string[];
  provider?: string;
  model?: string;
  error?: string;
  timestamp: string;
}

export class ChatWebSocketServer {
  private wss: WSServer | null = null;
  private readonly classifier: IntentClassifier;
  private readonly router: ContextRouter;
  private readonly llmProvider: BaseLLMProvider;

  constructor(
    classifier: IntentClassifier,
    router: ContextRouter,
    llmProvider: BaseLLMProvider
  ) {
    this.classifier = classifier;
    this.router = router;
    this.llmProvider = llmProvider;
  }

  /**
   * Start the WebSocket server on the given port.
   */
  start(port: number): void {
    this.wss = new WSServer({ port });

    this.wss.on("listening", () => {
      console.log(`\n${"=".repeat(60)}`);
      console.log(`  Shopcard AI Chat Service - WebSocket Server`);
      console.log(`  Listening on ws://localhost:${port}`);
      console.log(`  LLM Provider: ${this.llmProvider.providerName}`);
      console.log(`  RAG Modules: ${this.router.getRegisteredModules().join(", ")}`);
      console.log(`${"=".repeat(60)}\n`);
    });

    this.wss.on("connection", (ws: WebSocket, req) => {
      const clientId = `client-${Date.now()}`;
      console.log(`[WebSocket] New connection: ${clientId}`);

      ws.on("message", async (data: Buffer) => {
        await this.handleMessage(ws, clientId, data);
      });

      ws.on("close", () => {
        console.log(`[WebSocket] Disconnected: ${clientId}`);
      });

      ws.on("error", (error) => {
        console.error(`[WebSocket] Error for ${clientId}:`, error.message);
      });

      // Send welcome message
      const welcome: OutgoingMessage = {
        status: "success",
        response:
          "Connected to Shopcard AI Chat Service. Send a message with { \"platform\": \"consumer|merchant|admin\", \"query\": \"your question\" }",
        timestamp: new Date().toISOString(),
      };
      ws.send(JSON.stringify(welcome));
    });

    this.wss.on("error", (error) => {
      console.error("[WebSocket] Server error:", error.message);
    });
  }

  /**
   * Process an incoming WebSocket message through the full pipeline.
   */
  private async handleMessage(
    ws: WebSocket,
    clientId: string,
    data: Buffer
  ): Promise<void> {
    const startTime = Date.now();

    try {
      // 1. Parse and validate the incoming message
      const raw = data.toString("utf-8");
      console.log(`\n[${clientId}] Received: ${raw}`);

      let message: IncomingMessage;
      try {
        message = JSON.parse(raw);
      } catch {
        this.sendError(ws, "Invalid JSON format. Expected: { \"platform\": \"...\", \"query\": \"...\" }");
        return;
      }

      if (!message.platform || !message.query) {
        this.sendError(ws, "Missing required fields: 'platform' and 'query'.");
        return;
      }

      const validPlatforms: Platform[] = ["consumer", "merchant", "admin"];
      if (!validPlatforms.includes(message.platform)) {
        this.sendError(
          ws,
          `Invalid platform: "${message.platform}". Must be one of: ${validPlatforms.join(", ")}`
        );
        return;
      }

      // 2. Intent Classification
      console.log(`[${clientId}] Step 1: Classifying intent...`);
      const classification = this.classifier.classify(
        message.platform,
        message.query
      );

      // 3. Context Routing & RAG Retrieval
      console.log(`[${clientId}] Step 2: Routing to ${classification.targetRAG}...`);
      const ragResponse = await this.router.route(classification, message.query);

      // 4. LLM Response Generation
      console.log(`[${clientId}] Step 3: Generating LLM response...`);

      // Build a platform-aware system prompt
      const platformLabels: Record<Platform, string> = {
        consumer: "Shopcard Customer Support Assistant",
        merchant: "Shopcard Merchant Support Assistant",
        admin: "Shopcard Admin Support Assistant",
      };

      const llmResponse = await this.llmProvider.generateResponse(
        message.query,
        ragResponse.context,
        {
          systemPrompt: `You are the ${platformLabels[message.platform]}. Answer the user's question based ONLY on the provided context. Be helpful, concise, and professional. If the context does not contain enough information, clearly state that you don't have that information available.`,
          temperature: 0.3,
        }
      );

      const elapsed = Date.now() - startTime;
      console.log(
        `[${clientId}] ✓ Response generated in ${elapsed}ms (${ragResponse.results.length} sources)`
      );

      // 5. Send response
      const response: OutgoingMessage = {
        status: "success",
        platform: message.platform,
        intent: classification.intent,
        ragSelected: classification.targetRAG,
        confidence: classification.confidence,
        response: llmResponse.text,
        sources: [...new Set(ragResponse.results.map((r) => r.source))],
        provider: llmResponse.provider,
        model: llmResponse.model,
        timestamp: new Date().toISOString(),
      };

      ws.send(JSON.stringify(response));
    } catch (error: any) {
      console.error(`[${clientId}] Pipeline error:`, error.message);
      this.sendError(ws, `Internal error: ${error.message}`);
    }
  }

  /**
   * Send an error response to the client.
   */
  private sendError(ws: WebSocket, errorMessage: string): void {
    const response: OutgoingMessage = {
      status: "error",
      error: errorMessage,
      timestamp: new Date().toISOString(),
    };
    ws.send(JSON.stringify(response));
  }

  /**
   * Gracefully shut down the WebSocket server.
   */
  stop(): void {
    if (this.wss) {
      this.wss.close();
      console.log("[WebSocket] Server stopped.");
    }
  }
}
