from fastapi import FastAPI, WebSocket,  WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

import json
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict

### LANGCHAIN SETUP ###
# TO DO: Turn into class in separate script
def prepare_qa_documents(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
    
    documents = [
        Document(
            page_content=item["answer"],
            metadata={"question": item["question"]}
        )
        for item in qa_data
    ]
    
    return documents
### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def basic_retriever(documents, embeddings):

    vectorstore = Chroma.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()

    return retriever

def create_rag_chain(retriever):

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def initialize_rag_chain(document_path, embeddings):

    documents = prepare_qa_documents(document_path)
    retriever = basic_retriever(documents, embeddings)
    rag_chain = create_rag_chain(retriever)

    return rag_chain

def initialize_langchain_app():

    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)

    return app

def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

def process_user_message(user_message: str, config: dict, langchain_app):
    """Process user messages and return a response."""
    try:
        response = langchain_app.invoke({"input": user_message}, config=config)
        return response.get("answer", "Warning: Response object missing 'answer' field. Please check the invocation process.")
    except Exception as e:
        return f"Error: Unable to process the message due to {type(e).__name__}: {e}"

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()
document_path = "../data/home0001qa.json"

rag_chain = initialize_rag_chain(document_path, embeddings)
print(f"[INFO]: initializing rag chain...")
langchain_app = initialize_langchain_app()
print(f"[INFO]: initializing langchain app...")

# In-memory storage for WebSocket sessions (for demo purposes)
active_connections = {}

current_embeddings = "text-embedding-3-large (OpenAI)"
current_llm = "gpt-4o"
current_pipeline = "LangChain"

# POST Endpoint
class ChatRequest(BaseModel):
    message: str


@app.get("/config", response_model=dict)
async def get_config():
    """Returns the current config - LLMs and embeddings."""

    return {
        "llm": current_llm, 
        "embeddings": current_embeddings,
        "pipeline": current_pipeline
    }

@app.post("/chat/")
async def chat_post(request: BaseModel):
    """Handle one-off chat messages via POST requests."""
    user_message = request.message
    config = {"configurable": {"thread_id": "test123"}}
    response = process_user_message(user_message, config, langchain_app)
    return {"user_message": user_message, "bot_response": response}

# WebSocket Endpoint
@app.websocket("/ws/")
async def websocket_chat(websocket: WebSocket):
    """
    Handle real-time chat messages via WebSocket.
    """
    await websocket.accept()
    connection_id = id(websocket)
    active_connections[connection_id] = websocket
    config =  {"configurable": {"thread_id": connection_id}}

    try:
        # Send message after connection
        await websocket.send_text(f"Connected to HOME0001 assistant. Your ID is {connection_id}.")

        while True:
            # user message received
            user_message = await websocket.receive_text()
            response = process_user_message(user_message, config, langchain_app)            
            await websocket.send_text(f"{response}")
            
    except WebSocketDisconnect:
        print(f"[INFO]: WebSocket disconnected: {connection_id}")
    except Exception as e:
        await websocket.send_text(f"Error: {e}")
    finally:
        del active_connections[connection_id]
        try:
            await websocket.close()
        except RuntimeError:
            # WebSocket is already closed
            pass

# Serve Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
