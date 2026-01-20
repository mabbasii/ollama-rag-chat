# Ollama RAG Chat

A web-based chat interface for [Ollama](https://ollama.ai/) with RAG (Retrieval-Augmented Generation) support. This project allows you to chat with local LLMs and query your own knowledge base using vector similarity search.

![Architecture](docs/architecture.png)

## Features

- **Web UI** - Clean, modern chat interface with streaming responses
- **Direct Ollama Chat** - Chat with any Ollama model directly
- **RAG Support** - Query your own documents/knowledge base
- **Multiple RAG Endpoints** - Support for multiple knowledge bases
- **Streaming Responses** - Real-time token streaming for better UX
- **Thinking Display** - Shows model's thinking process (for supported models)
- **Markdown Rendering** - Properly formatted responses with code blocks, lists, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      BROWSER                                │
│                   (index.html + script.js)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP requests
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 NODE.JS SERVER (server.js)                  │
│                     Port 3001                               │
│                                                             │
│  • Serves static files (HTML, CSS, JS)                      │
│  • Proxies requests to backend services                     │
└────────┬───────────────────┬───────────────────┬────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FASTAPI PYTHON  │  │ FASTAPI PYTHON  │  │     OLLAMA      │
│   (rag_api.py)  │  │  (rag_api_2.py) │  │   (LLM Server)  │
│   Port 8001     │  │   Port 8002     │  │   Port 11434    │
│                 │  │                 │  │                 │
│  RAG Endpoint 1 │  │  RAG Endpoint 2 │  │ Direct LLM chat │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running
- Node.js 18+
- Python 3.10+

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ollama-rag-chat.git
cd ollama-rag-chat
```

### 2. Start Ollama

```bash
ollama serve
# In another terminal, pull a model:
ollama pull llama3.1:8b
```

### 3. Set up the frontend (Node.js)

```bash
cd frontend
npm install
npm start
```

### 4. Set up the RAG API (Python)

```bash
cd rag-api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build vector database from sample data
python vector_builder.py ../sample-data/sample_knowledge_base.csv \
    --content-column content \
    --metadata title category keywords

# Start the RAG API
python rag_api.py
```

### 5. Open the UI

Navigate to `http://localhost:3001` in your browser.

## Project Structure

```
ollama-rag-chat/
├── frontend/                 # Node.js web server
│   ├── server.js            # Express server (proxy + static files)
│   ├── package.json
│   └── public/
│       ├── index.html       # Chat UI
│       └── script.js        # Frontend JavaScript
│
├── rag-api/                  # Python RAG backend
│   ├── rag_api.py           # FastAPI server with RAG
│   ├── vector_builder.py    # Script to build vector DB from CSV
│   └── requirements.txt
│
├── sample-data/              # Example data files
│   └── sample_knowledge_base.csv
│
├── .gitignore
└── README.md
```

## Configuration

### Environment Variables

**Frontend (Node.js):**
```bash
PORT=3001                              # Web server port
OLLAMA_API=http://localhost:11434/api  # Ollama API URL
RAG_API_1=http://localhost:8001/api    # First RAG endpoint
RAG_API_2=http://localhost:8002/api    # Second RAG endpoint
```

**RAG API (Python):**
```bash
PORT=8001                              # RAG API port
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=knowledge_base
```

## Building Your Own Knowledge Base

1. Prepare your data as a CSV file with at least a `content` column
2. Run the vector builder:

```bash
python vector_builder.py your_data.csv \
    --db-path ./your_chroma_db \
    --collection your_collection \
    --content-column content \
    --metadata title category  # optional metadata columns
```

3. Update `CHROMA_DB_PATH` and `COLLECTION_NAME` in the RAG API

## Adding Models to the UI

Edit `frontend/public/index.html` to add more models to the dropdown:

```html
<select id="model-select">
    <!-- RAG options -->
    <option value="rag-knowledge-base">Knowledge Base (RAG)</option>

    <!-- Direct Ollama models -->
    <option value="llama3.1:8b">Llama 3.1 8B</option>
    <option value="mistral:7b">Mistral 7B</option>
    <option value="your-model:tag">Your Model</option>
</select>
```

## Tech Stack

- **Frontend**: HTML, TailwindCSS, JavaScript, Marked.js
- **Web Server**: Node.js, Express
- **RAG Backend**: Python, FastAPI, LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (BAAI/bge-large-en-v1.5)
- **LLM**: Ollama (any supported model)

## License

MIT License - feel free to use this project for any purpose.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
