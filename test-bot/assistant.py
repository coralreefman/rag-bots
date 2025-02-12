from fastapi import FastAPI, WebSocket,  WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import json
from typing import Sequence
from typing_extensions import Annotated, TypedDict

# Load the .env file
load_dotenv()

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
            "provider": "OpenAI API",
            "description": "Most capable OpenAI model",
            "instance": ChatOpenAI(model="gpt-4o")
        },
        "llama3.3": {
            "name": "llama3.3:70b",
            "provider": "Ollama",
            "description": "Local Llama 3.3 model via Ollama",
            "instance": ChatOllama(model="llama3.3:70b-instruct-q8_0")
        },
                "claude-3-opus": {
            "name": "claude-3-opus-20240229",
            "provider": "Anthropic API",
            "description": "Most capable Anthropic model",
            "instance": ChatAnthropic(model='claude-3-opus-20240229')
        }
    }

    _EMBEDDINGS = {
        "openai-default": {
            "name": "text-embedding-3-large",
            "provider": "OpenAI",
            "description": "Default OpenAI embedding model",
            "instance": OpenAIEmbeddings()
        },
                "nomic-1.5": {
            "name": "nomic-embed-text-v1.5",
            "provider": "HuggingFace",
            "description": "decent model",
            "instance": NomicEmbeddings(
                model='nomic-embed-text-v1.5', 
                inference_mode='local'
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
    
class ChatBot:

    def __init__(self, llm_id="openai-gpt-4o", embedding_id="openai-default"):
        """Initialize chatbot with specified models"""
        self.llm = ModelConfig.get_llm(llm_id)
        self.embeddings = ModelConfig.get_embeddings(embedding_id)
        self.current_llm_id = llm_id
        self.current_embedding_id = embedding_id
        self.active_connections = {}
        self.setup_prompts()

    # Dumb Document Loader - To be replaced
    def load_documents(self, file_path):
            """Load QA documents"""
            with open(file_path, 'r') as f:
                return [Document(page_content=item["answer"], 
                            metadata={"question": item["question"]}) 
                    for item in json.load(f)]
            
    def setup_prompts(self):
        """Setup chat prompts"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Your name is Talin. You're part of the HOME0001 team. You are an expert customer service rep for the housing collective HOME0001. Your goal is to help customers who are looking to buy a home at one of our locations. Use the following pieces of retrieved context to answer any question. If you don't find an answer just admit that you don't know. Only answer questions related to HOME0001. Talk in a pragmatic tone. don't get cute, keep it professional but avoid marketing or corporate speak at all costs. Talk like you're texting. Use three sentences maximum and keep the answer concise. Do not admit that you're a bot and do not give it away. Always act human. Talk about HOME0001 as 'us' and 'we'. \n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        self.context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def setup_vector_db(self, document_path):

        if os.path.isfile(f"./FAISS/{self.current_embedding_id}.faiss"):
            
            self.vectorstore = FAISS.load_local(
                folder_path="./FAISS",
                embeddings=self.embeddings,
                index_name=self.current_embedding_id,
                allow_dangerous_deserialization=True
            )

        else:

            documents = self.load_documents(document_path)

            self.vectorstore = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            self.vectorstore.save_local("./FAISS", self.current_embedding_id)                  
        
    def initialize_rag_chain(self):
        """Initialize RAG chain""" 
        
        retriever = self.vectorstore.as_retriever()
        
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

        # TO DO: add document grade to check for retrieved doc relevance AND 
        # add case for when no doc is relevant
        
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
bot.setup_vector_db("../data/home0001qa.json")
bot.initialize_rag_chain()
bot.initialize_app()

# API Models
class ChatRequest(BaseModel):
    message: str

class ConfigRequest(BaseModel):
    llm: str
    embeddings: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        config =  {"configurable": {"thread_id": "test-id"}}
        response = await bot.get_response(request.message, config=config)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
        bot.setup_vector_db("../data/home0001qa.json")
        bot.initialize_rag_chain()
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
    
