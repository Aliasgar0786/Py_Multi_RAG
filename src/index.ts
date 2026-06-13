/**
 * Shopcard AI Chat Service - Entry Point
 *
 * Initializes all components and starts the WebSocket server.
 *
 * Pipeline:
 *   WebSocket → IntentClassifier → ContextRouter → RAG Module → LLM → Response
 */

import * as path from "path";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config();

import { GeminiProvider } from "./llm/GeminiProvider";
import { StubLLMProvider } from "./llm/StubLLMProvider";
import { BaseLLMProvider } from "./llm/BaseLLMProvider";
import { IntentClassifier } from "./classifier/IntentClassifier";
import { ContextRouter } from "./router/ContextRouter";
import { CustomerRAG } from "./rag/CustomerRAG";
import { MerchantRAG } from "./rag/MerchantRAG";
import { AdminRAG } from "./rag/AdminRAG";
import { ChatWebSocketServer } from "./server/WebSocketServer";

function main(): void {
  console.log("Shopcard AI Chat Service - Phase 1 Multi-RAG Prototype");
  console.log("─".repeat(55));

  const llmProviderType = (process.env.LLM_PROVIDER || "gemini").toLowerCase();
  const modelName = process.env.GEMINI_MODEL || "gemini-2.0-flash";
  const wsPort = parseInt(process.env.WS_PORT || "8080", 10);
  const dataDir = path.resolve(__dirname, "..", "data");

  // ── 1. Initialize LLM Provider ────────────────────────────
  let llmProvider: BaseLLMProvider;

  if (llmProviderType === "stub") {
    console.log("  Provider:   stub (offline mode — no API calls)");
    llmProvider = new StubLLMProvider();
  } else {
    // ── Validate Gemini configuration ──────────────────────
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === "your_gemini_api_key_here") {
      console.error(
        "\n❌ GEMINI_API_KEY not configured!\n" +
          "   1. Copy .env.example to .env\n" +
          "   2. Set your Gemini API key in the .env file\n" +
          "   Tip: Set LLM_PROVIDER=stub in .env to test without an API key.\n"
      );
      process.exit(1);
    }
    console.log(`  Provider:   gemini (${modelName})`);
    llmProvider = new GeminiProvider(apiKey, modelName);
  }

  console.log(`  Data Dir:   ${dataDir}`);
  console.log(`  WS Port:    ${wsPort}`);
  console.log("");

  // ── 3. Initialize RAG Modules ─────────────────────────────
  console.log("Initializing RAG Modules...");
  const customerRAG = new CustomerRAG(dataDir);
  const merchantRAG = new MerchantRAG(dataDir);
  const adminRAG = new AdminRAG(dataDir);

  // ── 4. Initialize Context Router ──────────────────────────
  console.log("Initializing Context Router...");
  const router = new ContextRouter();
  router.registerRAG(customerRAG);
  router.registerRAG(merchantRAG);
  router.registerRAG(adminRAG);

  // ── 5. Initialize Intent Classifier ───────────────────────
  console.log("Initializing Intent Classifier...");
  const classifier = new IntentClassifier();

  // ── 6. Start WebSocket Server ─────────────────────────────
  const server = new ChatWebSocketServer(classifier, router, llmProvider);
  server.start(wsPort);

  // ── Graceful Shutdown ─────────────────────────────────────
  process.on("SIGINT", () => {
    console.log("\nShutting down...");
    server.stop();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    console.log("\nShutting down...");
    server.stop();
    process.exit(0);
  });
}

main();
