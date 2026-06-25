"""
WebSocketServer - Thin WebSocket transport layer.

This is just ONE way to call the pipeline. The actual AI logic lives in
ShopCardPipeline. This server simply:
  1. Accepts WebSocket connections
  2. Parses incoming JSON
  3. Calls pipeline.process_query(platform, query)
  4. Sends the result back as JSON

A backend team could replace this with Flask, FastAPI, Django, gRPC, etc.
and call the same pipeline.process_query() function.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import websockets
from websockets.asyncio.server import Server, ServerConnection

from ..pipeline import ShopCardPipeline


class ChatWebSocketServer:
    """
    WebSocket transport wrapper around ShopCardPipeline.
    This is NOT the business logic -- it's just the delivery layer.
    """

    def __init__(self, pipeline: ShopCardPipeline):
        self._pipeline = pipeline
        self._server: Server | None = None

    async def start(self, port: int) -> None:
        """Start the WebSocket server on the given port."""
        self._server = await websockets.serve(
            self._handle_connection,
            "localhost",
            port,
        )

        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  Shopcard AI Chat Service - WebSocket Server")
        print(f"  Listening on ws://localhost:{port}")
        print(f"{separator}\n")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket client connection."""
        client_id = f"client-{int(time.time() * 1000)}"
        print(f"[WebSocket] New connection: {client_id}")

        # Send welcome message
        welcome = {
            "status": "success",
            "response": (
                'Connected to Shopcard AI Chat Service. Send a message with '
                '{ "platform": "consumer|merchant|admin", "query": "your question" }'
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await websocket.send(json.dumps(welcome))

        try:
            async for raw_data in websocket:
                await self._handle_message(websocket, client_id, raw_data)
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"[WebSocket] Disconnected: {client_id}")

    async def _handle_message(
        self,
        websocket: ServerConnection,
        client_id: str,
        raw_data: str | bytes,
    ) -> None:
        """Parse the incoming message and delegate to the pipeline."""
        try:
            raw = raw_data if isinstance(raw_data, str) else raw_data.decode("utf-8")
            print(f"\n[{client_id}] Received: {raw}")

            # Parse JSON
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(
                    websocket,
                    'Invalid JSON format. Expected: { "platform": "...", "query": "..." }',
                )
                return

            platform = message.get("platform", "")
            query = message.get("query", "")

            # ── Delegate to the pipeline (the ONLY line that matters) ──
            result = await self._pipeline.process_query(platform, query)

            # Send result back to client
            await websocket.send(json.dumps(result))

        except Exception as error:
            print(f"[{client_id}] Error: {error}")
            await self._send_error(websocket, f"Internal error: {error}")

    async def _send_error(self, websocket: ServerConnection, error_message: str) -> None:
        """Send an error response to the client."""
        response = {
            "status": "error",
            "error": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await websocket.send(json.dumps(response))

    async def stop(self) -> None:
        """Gracefully shut down the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            print("[WebSocket] Server stopped.")
