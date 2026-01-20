#!/usr/bin/env python3
"""
FastAPI server for RAG (Retrieval-Augmented Generation)
Uses CPU-based HuggingFace embeddings + Ollama LLM
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import os

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Initialize FastAPI
app = FastAPI(title="RAG API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION - Customize these for your setup
# ============================================================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

# ============================================================================
# INITIALIZE MODELS
# ============================================================================
print("Starting RAG API...")
print(f"Loading Ollama model: {OLLAMA_MODEL}")

model = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    keep_alive=-1,
    temperature=0.2
)
print("LLM loaded successfully!")

# Initialize CPU-based embeddings
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embeddings loaded!")

# Load vector store
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH
)
print(f"Loaded vector database from {CHROMA_DB_PATH}")

# Create retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
print("Retriever ready\n")

# ============================================================================
# PROMPT TEMPLATE - Customize for your use case
# ============================================================================
template = """You are a helpful assistant. Answer questions based on the provided context.

RULES:
1. Only answer based on the context provided below
2. If the context doesn't contain the answer, say "I don't have information about that in my knowledge base."
3. Be concise and accurate
4. Never make up information

CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_history}

QUESTION: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    stream: Optional[bool] = True

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "RAG API",
        "model": OLLAMA_MODEL,
        "embedding": EMBEDDING_MODEL
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG."""

    print(f"\n{'='*60}")
    print(f"Query: {request.message}")

    try:
        # Step 1: Retrieve relevant documents
        print("Retrieving documents...")
        docs = retriever.invoke(request.message)
        print(f"   Retrieved {len(docs)} documents")

        # Step 2: Format context from retrieved documents
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"\n--- Document {i} ---\n"
            context += doc.page_content
            context += "\n"

        # Step 3: Format conversation history
        conversation_history = ""
        if request.history:
            recent_history = request.history[-6:] if len(request.history) > 6 else request.history
            if recent_history:
                conversation_history = "Previous conversation:\n"
                for msg in recent_history:
                    role_label = "User" if msg.role == "user" else "Assistant"
                    conversation_history += f"{role_label}: {msg.content}\n"
                conversation_history += "\n"
                print(f"Including {len(recent_history)} previous messages")

        # Step 4: Generate response
        print("Generating response...")

        if request.stream:
            async def generate():
                try:
                    for chunk in chain.stream({
                        "context": context,
                        "question": request.message,
                        "conversation_history": conversation_history
                    }):
                        yield f"data: {json.dumps({'token': chunk, 'type': 'content'})}\n\n"
                        await asyncio.sleep(0)

                    yield "data: [DONE]\n\n"
                    print("Response generated and sent")

                except Exception as e:
                    print(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            response = chain.invoke({
                "context": context,
                "question": request.message,
                "conversation_history": conversation_history
            })

            print("Response generated")
            return {
                "success": True,
                "response": response,
                "sources": [
                    {"content": doc.page_content[:200] + "..."}
                    for doc in docs
                ]
            }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("PORT", 8001))

    print("\n" + "="*60)
    print("RAG API Server")
    print("="*60)
    print(f"Running on: http://127.0.0.1:{PORT}")
    print(f"Vector DB: {CHROMA_DB_PATH}")
    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info"
    )
