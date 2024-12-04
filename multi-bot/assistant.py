from fastapi import FastAPI, WebSocket,  WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

import json

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

class ModelConfig:
    """Static configuration and instances of models and embeddings"""
    
    # Initialize model instances once
    _MODELS = {
        "openai-gpt-4o": {
            "name": "gpt-4o",
            "provider": "OpenAI",
            "description": "Most capable OpenAI model",
            "instance": ChatOpenAI(model="gpt-4o")
        },
        "ollama-llama3.1": {
            "name": "Llama 3.1",
            "provider": "Ollama",
            "description": "Local Llama 3.1 model via Ollama",
            "instance": ChatOllama(model="llama3.1")
        }
    }
    
    _EMBEDDINGS = {
        "openai-default": {
            "name": "text-embedding-3-large",
            "provider": "OpenAI",
            "description": "Default OpenAI embedding model",
            "instance": OpenAIEmbeddings()
        },
        "hf-mpnet": {
            "name": "all-mpnet-base-v2 ",
            "provider": "HuggingFace",
            "description": "small model for testing",
            "instance": HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        },
        "bge-small": {
            "name": "bge-small-en",
            "provider": "HuggingFace",
            "description": "small model for testing",
            "instance": HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True}
            )
        }

    }

    @classmethod
    def get_llm(cls, model_id):
        """Get existing LLM instance"""
        if model_id not in cls._MODELS:
            raise ValueError(f"Unknown LLM: {model_id}")
        return cls._MODELS[model_id]["instance"]
    
    @classmethod
    def get_embeddings(cls, embedding_id):
        """Get existing embeddings instance"""
        if embedding_id not in cls._EMBEDDINGS:
            raise ValueError(f"Unknown embeddings: {embedding_id}")
        return cls._EMBEDDINGS[embedding_id]["instance"]

    @classmethod
    def get_frontend_config(cls):
        """Get configuration formatted for frontend dropdowns"""
        return {
            "llms": [
                {
                    "id": model_id,
                    "name": config["name"],
                    "provider": config["provider"],
                    "description": config["description"]
                }
                for model_id, config in cls._MODELS.items()
            ],
            "embeddings": [
                {
                    "id": embed_id,
                    "name": config["name"],
                    "provider": config["provider"],
                    "description": config["description"]
                }
                for embed_id, config in cls._EMBEDDINGS.items()
            ]
        }
        
class ChatBot:
    def __init__(self, llm_id="openai-gpt-4o", embedding_id="openai-default"):
        """Initialize chatbot with specified models"""
        self.llm = ModelConfig.get_llm(llm_id)
        self.embeddings = ModelConfig.get_embeddings(embedding_id)
        self.current_llm_id = llm_id
        self.current_embedding_id = embedding_id
        self.active_connections = {}
        self.setup_prompts()
    
    def _get_llm(self, model_id):
        """Get LLM instance from config"""
        if model_id not in ModelConfig.LLM_CONFIGS:
            raise ValueError(f"Unknown LLM: {model_id}")
        return ModelConfig.LLM_CONFIGS[model_id]["create"]()
    
    def _get_embeddings(self, embedding_id):
        """Get embeddings instance from config"""
        if embedding_id not in ModelConfig.EMBEDDING_CONFIGS:
            raise ValueError(f"Unknown embeddings: {embedding_id}")
        return ModelConfig.EMBEDDING_CONFIGS[embedding_id]["create"]()
    
    def load_documents(self, file_path):
        """Load QA documents"""
        with open(file_path, 'r') as f:
            return [Document(page_content=item["answer"], 
                           metadata={"question": item["question"]}) 
                   for item in json.load(f)]

    def setup_prompts(self):
        """Setup chat prompts"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def initialize_rag_chain(self, document_path):
        """Initialize RAG chain"""
 
        documents = self.load_documents(document_path)
        vectorstore = FAISS.from_documents(
            documents, 
            self.embeddings,
            # persist_directory=chroma_db_path
        )
        retriever = vectorstore.as_retriever()
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.context_prompt
        )
        qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def call_model(self, state: State):
        response = self.rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
    
    def initialize_app(self):

        self.workflow = StateGraph(state_schema=State)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.call_model)

        memory = MemorySaver()

        self.app = self.workflow.compile(checkpointer=memory)

    async def get_response(self, message: str, config: dict) -> str:
        """Get response for a message"""
        try:
            response = self.app.invoke(
                {"input": message},
                config=config
            )
            return response.get("answer", "Warning: Response object missing 'answer' field. Please check the invocation process.")
        except Exception as e:
            return f"Error: {str(e)}"
        
# Initialize FastAPI and first instance of Chatbot
app = FastAPI()
bot = ChatBot()
bot.initialize_rag_chain("../data/home0001qa.json")
bot.initialize_app()

# API Models
class ChatRequest(BaseModel):
    message: str

class ConfigRequest(BaseModel):
    llm: str
    embeddings: str

@app.get("/config/options")
async def get_config_options():
    """Get available model options for frontend"""
    return ModelConfig.get_frontend_config()

@app.get("/config/current")
async def get_current_config():
    """Get current model configuration"""
    return {
        "llm": bot.current_llm_id,
        "embeddings": bot.current_embedding_id
    }

@app.post("/config")
async def update_config(config: ConfigRequest):
    """Update models configuration and reinitialize the entire chain"""
    global bot
    try:
        # Create new bot instance with requested models
        bot = ChatBot(config.llm, config.embeddings)
        bot.initialize_rag_chain("../data/home0001qa.json")
        bot.initialize_app()
        return {
            "status": "success",
            "config": {
                "llm": bot.current_llm_id,
                "embeddings": bot.current_embedding_id
            }
        }
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    
@app.websocket("/ws/")
async def websocket_chat(websocket: WebSocket):
    """Handle WebSocket connections"""
    await websocket.accept()
    connection_id = str(id(websocket))
    bot.active_connections[connection_id] = websocket
    config =  {"configurable": {"thread_id": connection_id}}
    
    try:
        await websocket.send_text(f"Connected. ID: {connection_id}")
        while True:
            message = await websocket.receive_text()
            response = await bot.get_response(message, config)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        del bot.active_connections[connection_id]
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass

# Serve Frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")