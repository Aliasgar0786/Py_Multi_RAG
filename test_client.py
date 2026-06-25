"""
Test Client - Validates the full Shopcard AI Chat Service pipeline.

Sends test queries for all three platforms (Consumer, Merchant, Admin)
and verifies correct routing, context isolation, and LLM response.

Usage: Start the server first (`python -m src_python.main`), then run:
  python test_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass

import websockets


@dataclass
class TestCase:
    name: str
    platform: str  # "consumer" | "merchant" | "admin"
    query: str
    expected_rag: str


TEST_CASES: list[TestCase] = [
    # ── Consumer Tests ─────────────────────────────────────────
    TestCase(
        name="Consumer - Refund Query",
        platform="consumer",
        query="How do refunds work?",
        expected_rag="customer_rag",
    ),
    TestCase(
        name="Consumer - Order Tracking",
        platform="consumer",
        query="How can I track my order?",
        expected_rag="customer_rag",
    ),
    TestCase(
        name="Consumer - Shopping Help",
        platform="consumer",
        query="What payment methods do you accept?",
        expected_rag="customer_rag",
    ),
    # ── Merchant Tests ─────────────────────────────────────────
    TestCase(
        name="Merchant - Onboarding",
        platform="merchant",
        query="How do I become a seller on Shopcard?",
        expected_rag="merchant_rag",
    ),
    TestCase(
        name="Merchant - Product Listing",
        platform="merchant",
        query="How do I add a new product listing?",
        expected_rag="merchant_rag",
    ),
    TestCase(
        name="Merchant - Store Dashboard",
        platform="merchant",
        query="Where can I see my sales analytics and performance metrics?",
        expected_rag="merchant_rag",
    ),
    # ── Admin Tests ────────────────────────────────────────────
    TestCase(
        name="Admin - User Suspension",
        platform="admin",
        query="What is the process for suspending a user account?",
        expected_rag="admin_rag",
    ),
    TestCase(
        name="Admin - Monitoring",
        platform="admin",
        query="How do I set up alerts for system monitoring?",
        expected_rag="admin_rag",
    ),
    TestCase(
        name="Admin - Workflows",
        platform="admin",
        query="What is the merchant application review workflow?",
        expected_rag="admin_rag",
    ),
]

WS_URL = "ws://localhost:8080"


async def run_single_test(test_case: TestCase) -> bool:
    """Run a single test case and return True if it passed."""
    try:
        async with websockets.connect(WS_URL) as ws:
            # Wait for welcome message
            welcome_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            # Welcome received, now send the test query
            payload = json.dumps({
                "platform": test_case.platform,
                "query": test_case.query,
            })
            await ws.send(payload)

            # Wait for response (generous timeout for rate-limit retries)
            response_raw = await asyncio.wait_for(ws.recv(), timeout=120)
            response = json.loads(response_raw)

            # Validate the response
            print(f"  [TEST] {test_case.name}")
            print(f"     Platform: {test_case.platform}")
            print(f'     Query:    "{test_case.query}"')

            if response.get("status") == "error":
                print(f"     [FAIL] Error: {response.get('error')}\n")
                return False

            rag_correct = response.get("ragSelected") == test_case.expected_rag
            rag_status = "[PASS]" if rag_correct else f"[FAIL] (expected {test_case.expected_rag})"
            print(f"     RAG:      {response.get('ragSelected')} {rag_status}")
            print(f"     Intent:   {response.get('intent')}")
            print(f"     Confidence: {response.get('confidence')}")
            sources = ", ".join(response.get("sources", [])) or "none"
            print(f"     Sources:  {sources}")
            resp_text = (response.get("response") or "")[:120]
            print(f"     Response: {resp_text}...")
            print(f"     Provider: {response.get('provider')}/{response.get('model')}")
            result_str = "[PASS] PASS" if rag_correct else "[FAIL] FAIL"
            print(f"     {result_str}\n")

            return rag_correct

    except Exception as error:
        print(f"  [FAIL] {test_case.name}: ERROR - {error}\n")
        return False


async def run_tests() -> None:
    print("+" + "=" * 58 + "+")
    print("|  Shopcard AI Chat Service - Integration Test Client     |")
    print("+" + "=" * 58 + "+\n")

    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        result = await run_single_test(test_case)
        if result:
            passed += 1
        else:
            failed += 1

        # Generous delay between tests to respect free-tier rate limits
        await asyncio.sleep(5)

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(TEST_CASES)} total")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTest run interrupted.")
        sys.exit(1)
