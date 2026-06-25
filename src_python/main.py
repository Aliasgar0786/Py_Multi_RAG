"""
Shopcard AI Chat Service - Entry Point

This is the file that starts everything.

What happens when you run this:
    1. Loads .env configuration (API key, provider, port)
    2. Creates a ShopCardPipeline (which initializes ALL components internally)
    3. Wraps it in a WebSocket server for browser/client access
    4. Listens for connections on ws://localhost:8080

The pipeline is the brain. The WebSocket server is just the ears and mouth.
A backend team could skip the WebSocket and call pipeline.process_query() directly.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .pipeline import ShopCardPipeline
from .server import ChatWebSocketServer


async def main() -> None:
    print("Shopcard AI Chat Service - Phase 1 Multi-RAG Prototype")
    print("-" * 55)

    ws_port = int(os.getenv("WS_PORT", "8080"))

    # ── 1. Create the pipeline (initializes EVERYTHING) ───────
    #    This single line sets up: classifier, router, 3 RAG modules, LLM provider
    pipeline = ShopCardPipeline()

    # ── 2. Wrap it in a WebSocket server ──────────────────────
    #    The WebSocket server is just a transport layer.
    #    It calls pipeline.process_query() for every message.
    server = ChatWebSocketServer(pipeline)
    await server.start(ws_port)

    # ── Keep running until interrupted ────────────────────────
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        print("\nShutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    # Windows fallback for signal handling
    if sys.platform == "win32":
        def _win_handler(signum, frame):
            print("\nShutting down...")
            stop_event.set()
        signal.signal(signal.SIGINT, _win_handler)

    try:
        await stop_event.wait()
    finally:
        await server.stop()


def run() -> None:
    """Convenience entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    run()
