## Nova Przestrzeń Chatbot (Flask + React)

An end-to-end retrieval-augmented chatbot for Nova Przestrzeń. The backend (Flask) loads a PDF brochure, builds embeddings, and serves streaming answers over Server-Sent Events. The frontend (React) provides a simple chat UI. You can switch between OpenAI and a local Ollama model.

### Features
- Streaming chat responses (SSE)
- Retrieval-Augmented Generation (FAISS + HuggingFace embeddings)
- OpenAI or local Ollama LLM
- Simple session handling with system prompt

### Project Structure
```
flask-app/
  app/
    __init__.py          # Flask app factory and blueprint registration
    config.py            # Env-driven configuration
    state.py             # In-memory chat state
    routes/
      chat.py            # Routes: home, new-session, chat-stream endpoints
    services/
      rag.py             # RAG chain construction
      streaming.py       # Streaming SSE helpers
  data/
    nova_przestrzen.pdf # Source brochure/content
  templates/
    index.html          # Minimal web UI (if not using React)
  my-react-app/         # React frontend (Create React App)
  run.py                # Entry point using the app factory
  requirements.txt
  .gitignore
  README.md
```

### Prerequisites
- Python 3.10+
- Node 18+

### Setup (Backend)
1. Create a virtual environment and install deps:
   ```bash
   cd flask-app
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create `.env` and set keys:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   OPENAI_API_KEY=...
   OPENAI_MODEL=gpt-3.5-turbo
   PDF_PATH=data/nova_przestrzen.pdf
   FRONTEND_ORIGIN=http://localhost:3000
   ```
3. Run the backend:
   ```bash
   python run.py
   ```

### Setup (Frontend)
```bash
cd my-react-app
npm install
npm start
```

By default, the React app runs on `http://localhost:3000` and the Flask app on `http://localhost:5000`.

### Using the App
- Visit the React UI at `http://localhost:3000` or the basic template at `http://localhost:5000/`.
- Use the model selector to switch between OpenAI and a local Ollama model.

### Configuration
Environment variables (with defaults):
- `OPENAI_API_KEY` (no default)
- `OPENAI_MODEL` (default: `gpt-3.5-turbo`)
- `PDF_PATH` (default: `data/nova_przestrzen.pdf`)
- `FRONTEND_ORIGIN` (default: `http://localhost:3000`)

### Notes
- This project uses in-memory chat history; for production, persist sessions (e.g., Redis/Postgres).
- FAISS index is built at startup. For larger corpora, pre-build and persist the index.

### License
MIT


