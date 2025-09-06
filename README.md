# AI Prompt Application with Conversation Memory

A web-based AI chat application powered by LLaMA with intelligent conversation memory and session-based context awareness.

## Features

- 🤖 **LLaMA-powered AI responses** with local model execution
- 🧠 **Intelligent conversation memory** that learns from user interactions
- 📝 **AI-generated session summaries** (up to 300 words) capturing user identity, topics, and communication style
- 🎯 **Session-based context** for natural, continuous conversations
- 🌐 **Modern web interface** with glassmorphism design
- ⚡ **Real-time responses** via FastAPI backend

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Download LLaMA Model
```bash
mkdir -p models
# Download a 4GB model (recommended for testing)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/llama-2-7b-chat.q4.gguf
```

### 3. Configure Environment
```bash
# Create .env file in backend directory
echo "MODEL_PATH=/absolute/path/to/your/models/llama-2-7b-chat.q4.gguf" > .env
```

### 4. Start the Server
```bash
cd backend
uvicorn app:app --reload
```

### 5. Open Web Interface
Navigate to `http://localhost:8000` in your browser.

## How It Works

### Conversation Memory System
- **Session Isolation**: Each chat session maintains its own context
- **AI-Generated Summaries**: Comprehensive 300-word summaries capture:
  - User identity (name, profession, personal details)
  - Topics discussed and interests
  - Communication style and personality
  - Important context for future responses
- **Smart Context Injection**: Relevant conversation history is automatically included in prompts

### API Endpoints
- `POST /api/prompt` - Send prompts and receive AI responses
- `GET /api/session/{session_id}/summary` - View session summary
- `GET /api/stats` - System statistics

## Project Structure
```
ai-prompt/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── model.py              # LLaMA model handler
│   ├── conversation_store.py  # Memory & summary system
│   ├── config.py             # Configuration
│   ├── static/
│   │   └── index.html        # Web interface
│   └── .env                  # Environment variables
├── requirements.txt          # Dependencies
└── README.md
```

## Configuration

Environment variables in `.env`:
```bash
MODEL_PATH=/path/to/your/llama-model.gguf
MAX_TOKENS=2000
TEMPERATURE=0.7
TOP_P=0.95
```

## Requirements

- Python 3.8+
- 8GB+ RAM (for LLaMA model)
- Modern web browser

## License

MIT License