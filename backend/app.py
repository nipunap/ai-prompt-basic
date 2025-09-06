from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from model import llm_handler
from config import settings
from conversation_store import conversation_store
import uuid

app = FastAPI(title="LLaMA Prompt API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    use_context: bool = True

class PromptResponse(BaseModel):
    response: str
    conversation_id: Optional[int] = None
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: str  # "positive", "negative", "neutral"
    feedback_text: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the LLaMA model on startup."""
    print("🚀 Starting AI Prompt Server with Learning Capabilities")
    print(f"📁 Model path: {settings.MODEL_PATH}")

    if not settings.MODEL_PATH:
        print("⚠️  Warning: MODEL_PATH not set in environment variables")
        print("The server will start but AI responses will not be available.")
        return

    print("📥 Loading LLaMA model... This may take a few moments...")
    try:
        success = llm_handler.load_model()
        if success:
            print("✅ LLaMA model loaded successfully!")
            print("🧠 Learning system is ready to collect conversations and feedback.")
        else:
            print("❌ Failed to load LLaMA model - check model path and file permissions")
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        print("The server will start but AI responses may not work properly.")

@app.post("/api/prompt", response_model=PromptResponse)
async def generate_prompt(request: PromptRequest):
    """Generate a response for the given prompt with learning capabilities."""
    if not llm_handler.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    response, conversation_id = llm_handler.generate_response(
        request.prompt,
        session_id=session_id,
        use_context=request.use_context
    )

    return PromptResponse(
        response=response,
        conversation_id=conversation_id,
        session_id=session_id
    )

# Feedback endpoint (optional - uncomment if you want to collect explicit feedback)
# @app.post("/api/feedback")
# async def submit_feedback(request: FeedbackRequest):
#     """Submit user feedback for a conversation to improve learning."""
#     try:
#         conversation_store.update_feedback(
#             request.conversation_id,
#             request.feedback,
#             request.feedback_text
#         )
#         return {"status": "success", "message": "Feedback recorded"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error recording feedback: {e}")

@app.get("/api/stats")
async def get_conversation_stats():
    """Get statistics about conversation history and learning."""
    try:
        stats = conversation_store.get_conversation_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {e}")

@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get the session summary for debugging."""
    try:
        summary = conversation_store.get_conversation_summary(session_id)
        messages = conversation_store.get_session_context(session_id)
        return {
            "session_id": session_id,
            "summary": summary,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session summary: {e}")

@app.get("/api/health")
async def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": llm_handler.model is not None
    }

@app.get("/")
async def root():
    """Serve the test interface."""
    return FileResponse('static/index.html')
