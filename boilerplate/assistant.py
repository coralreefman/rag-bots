from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# In-memory storage for WebSocket sessions (for demo purposes)
active_connections = {}

# Simplified Chatbot Logic
def chatbot_response(user_message: str) -> str:
    """Generate a simple response for the chatbot."""
    return f"Bot: You said '{user_message}'."

# POST Endpoint
class ChatRequest(BaseModel):
    message: str

@app.post("/chat/")
async def chat_post(request: ChatRequest):
    """
    Handle one-off chat messages via POST requests.
    """
    user_message = request.message
    bot_response = chatbot_response(user_message)
    return {"user_message": user_message, "bot_response": bot_response}

# WebSocket Endpoint
@app.websocket("/ws/")
async def websocket_chat(websocket: WebSocket):
    """
    Handle real-time chat messages via WebSocket.
    """
    await websocket.accept()
    connection_id = id(websocket)
    active_connections[connection_id] = websocket

    try:
        # Send welcome message
        await websocket.send_text(f"You connected to the WebSocket chatbot! Your ID is {connection_id}.")

        while True:
            # Receive message and respond
            user_message = await websocket.receive_text()
            bot_response = chatbot_response(user_message)
            await websocket.send_text(bot_response)
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        del active_connections[connection_id]
        await websocket.close()

# Serve Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
