from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI()

# FastAPI app instance
app = FastAPI()

# Setup OpenAI Assistant
instructions = '''
You are an expert customer service rep for the housing collective HOME0001. Use your knowledge base to answer questions about the project.
If you don't find an answer just say 'I don't know :('. Only answer questions related to the project.
Talk in a casual, pragmatic tone. Avoid marketing or corporate speak at all costs.
'''

assistant = client.beta.assistants.create(
    name="HOME0001 Customer Assistant",
    instructions=instructions,
    model="gpt-4o",
    tools=[{"type": "file_search"}],
)
assistant_id = assistant.id
print(f"Assistant created with ID: {assistant_id}")

# Create a vector store
vector_store = client.beta.vector_stores.create(name="FAQ")
vector_store_id = vector_store.id
print(f"Vector store created with ID: {vector_store_id}")

# Ready the files for upload to OpenAI
file_paths = ["data/home0001qa.json"]
file_streams = [open(path, "rb") for path in file_paths]

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# Update the assistant with vector store resources
assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# REST API Endpoint
class Query(BaseModel):
    query: str

@app.post("/chat/")
async def chat(query: Query):
    try:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": query.query
                }
            ]
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id
        )
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        message_content = messages[0].content[0].text
        return {"response": message_content.value}
    except Exception as e:
        return {"error": str(e)}

# WebSocket Endpoint
@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to HOME0001 assistant! Send your query.")
    try:
        while True:
            data = await websocket.receive_text()
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": data
                    }
                ]
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id, assistant_id=assistant.id
            )
            messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
            message_content = messages[0].content[0].text
            await websocket.send_text(message_content.value)
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        await websocket.close()

# Serve the Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
