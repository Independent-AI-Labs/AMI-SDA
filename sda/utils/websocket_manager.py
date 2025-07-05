import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket {websocket.client} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket {websocket.client} disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: Dict[str, Any]):
        if not self.active_connections:
            # print("No active WebSocket connections to broadcast to.") # Can be noisy
            return

        # Create a list of tasks for sending messages
        tasks = []
        for ws in self.active_connections:
            tasks.append(ws.send_json(data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Iterate backwards for safe removal
        for i in range(len(self.active_connections) - 1, -1, -1):
            ws = self.active_connections[i]
            result = results[i]
            if isinstance(result, Exception):
                print(f"Error broadcasting to websocket {ws.client}: {result}. Disconnecting.")
                # This websocket is now considered dead, remove it
                # No await needed for self.disconnect as it's synchronous
                self.disconnect(ws)

# Global instance for the Control Panel
control_panel_manager = ConnectionManager()
