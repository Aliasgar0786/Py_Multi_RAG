/**
 * Test Client - Validates the full Shopcard AI Chat Service pipeline.
 *
 * Sends test queries for all three platforms (Consumer, Merchant, Admin)
 * and verifies correct routing, context isolation, and LLM response.
 *
 * Usage: Start the server first (`npm run dev`), then run:
 *   npx ts-node src/test_client.ts
 */

import WebSocket from "ws";

interface TestCase {
  name: string;
  platform: "consumer" | "merchant" | "admin";
  query: string;
  expectedRAG: string;
}

const TEST_CASES: TestCase[] = [
  // ── Consumer Tests ─────────────────────────────────────────
  {
    name: "Consumer - Refund Query",
    platform: "consumer",
    query: "How do refunds work?",
    expectedRAG: "customer_rag",
  },
  {
    name: "Consumer - Order Tracking",
    platform: "consumer",
    query: "How can I track my order?",
    expectedRAG: "customer_rag",
  },
  {
    name: "Consumer - Shopping Help",
    platform: "consumer",
    query: "What payment methods do you accept?",
    expectedRAG: "customer_rag",
  },

  // ── Merchant Tests ─────────────────────────────────────────
  {
    name: "Merchant - Onboarding",
    platform: "merchant",
    query: "How do I become a seller on Shopcard?",
    expectedRAG: "merchant_rag",
  },
  {
    name: "Merchant - Product Listing",
    platform: "merchant",
    query: "How do I add a new product listing?",
    expectedRAG: "merchant_rag",
  },
  {
    name: "Merchant - Store Dashboard",
    platform: "merchant",
    query: "Where can I see my sales analytics and performance metrics?",
    expectedRAG: "merchant_rag",
  },

  // ── Admin Tests ────────────────────────────────────────────
  {
    name: "Admin - User Suspension",
    platform: "admin",
    query: "What is the process for suspending a user account?",
    expectedRAG: "admin_rag",
  },
  {
    name: "Admin - Monitoring",
    platform: "admin",
    query: "How do I set up alerts for system monitoring?",
    expectedRAG: "admin_rag",
  },
  {
    name: "Admin - Workflows",
    platform: "admin",
    query: "What is the merchant application review workflow?",
    expectedRAG: "admin_rag",
  },
];

const WS_URL = "ws://localhost:8080";

async function runTests(): Promise<void> {
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║  Shopcard AI Chat Service - Integration Test Client     ║");
  console.log("╚══════════════════════════════════════════════════════════╝\n");

  let passed = 0;
  let failed = 0;

  for (const testCase of TEST_CASES) {
    try {
      const result = await runSingleTest(testCase);
      if (result) {
        passed++;
      } else {
        failed++;
      }
    } catch (error: any) {
      console.log(`  ❌ ${testCase.name}: ERROR - ${error.message}\n`);
      failed++;
    }

    // Generous delay between tests to respect free-tier rate limits
    await sleep(5000);
  }

  console.log("\n" + "═".repeat(60));
  console.log(`  Results: ${passed} passed, ${failed} failed, ${TEST_CASES.length} total`);
  console.log("═".repeat(60));

  process.exit(failed > 0 ? 1 : 0);
}

function runSingleTest(testCase: TestCase): Promise<boolean> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);
    let welcomeReceived = false;

    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error("Timeout after 120 seconds (rate-limit retries may be in progress)"));
    }, 120000);

    ws.on("open", () => {
      // Wait for welcome message before sending
    });

    ws.on("message", (data: Buffer) => {
      const response = JSON.parse(data.toString());

      // Skip the welcome message
      if (!welcomeReceived) {
        welcomeReceived = true;
        // Now send the test query
        const message = JSON.stringify({
          platform: testCase.platform,
          query: testCase.query,
        });
        ws.send(message);
        return;
      }

      clearTimeout(timeout);
      ws.close();

      // Validate the response
      console.log(`  📋 ${testCase.name}`);
      console.log(`     Platform: ${testCase.platform}`);
      console.log(`     Query:    "${testCase.query}"`);

      if (response.status === "error") {
        console.log(`     ❌ Error: ${response.error}\n`);
        resolve(false);
        return;
      }

      const ragCorrect = response.ragSelected === testCase.expectedRAG;
      console.log(
        `     RAG:      ${response.ragSelected} ${ragCorrect ? "✅" : `❌ (expected ${testCase.expectedRAG})`}`
      );
      console.log(`     Intent:   ${response.intent}`);
      console.log(`     Confidence: ${response.confidence}`);
      console.log(`     Sources:  ${(response.sources || []).join(", ") || "none"}`);
      console.log(
        `     Response: ${(response.response || "").substring(0, 120)}...`
      );
      console.log(`     Provider: ${response.provider}/${response.model}`);
      console.log(`     ${ragCorrect ? "✅ PASS" : "❌ FAIL"}\n`);

      resolve(ragCorrect);
    });

    ws.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

runTests().catch((error) => {
  console.error("Test runner failed:", error.message);
  process.exit(1);
});
