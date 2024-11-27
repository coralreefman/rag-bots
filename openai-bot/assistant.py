from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

HOME0001_ASSISTANT_ID = "asst_1XlpF0EPSnscirf5ZfHWwSLc"
client = OpenAI()

app = FastAPI()

# In-memory storage for WebSocket sessions (for demo purposes)
active_connections = {}

def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def create_thread_and_run(user_input: str) -> str:
    """
    Create a new thread and submit the first user message.
    """
    # Create a new thread
    thread = client.beta.threads.create()

    # Submit the first user message and poll for the run to complete
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_input
    )
    polled_run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=HOME0001_ASSISTANT_ID,
    )

    return thread, polled_run

def get_responses(thread):
    return client.beta.threads.messages.list(thread_id=thread.id) #, order="asc"

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
        # Send message after connection
        await websocket.send_text(f"You connected to the WebSocket chatbot! Your ID is {connection_id}.")

        first_message = True

        while True:
            # user message received
            user_message = await websocket.receive_text()
            await websocket.send_text(f"message received: {user_message}")

            if first_message:

                thread, run = create_thread_and_run(user_message)
                await websocket.send_text(f"created a new thread {thread.id} \n and a new run {run.id}")
                await websocket.send_text(f"1st status: {run.status}")
     
                messages = get_responses(thread)
                answer = list(messages)[0].content[0].text.value
                await websocket.send_text(f"{answer}") 
                first_message = False
            
            else:
                run = submit_message(HOME0001_ASSISTANT_ID, thread, user_message)
                messages = get_responses(thread)
                answer = list(messages)[0].content[0].text.value
                await websocket.send_text(f"{answer}")
            
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        del active_connections[connection_id]
        await websocket.close()

# Serve Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
