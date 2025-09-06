import os
from llama_cpp import Llama
import uuid
from conversation_store import conversation_store, Conversation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLaMAHandler:
    def __init__(self):
        self.model = None
        self.model_path = os.getenv('MODEL_PATH', './models/llama-2-7b-chat.gguf')

    def load_model(self):
        """Load the LLaMA model."""
        try:
            print(f"Loading LLaMA model from: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of CPU threads
                verbose=False
            )
            print("✅ LLaMA model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_response(self, prompt: str, session_id: str = None, use_context: bool = True) -> tuple[str, str]:
        """Generate response using LLaMA model with optional conversation context."""
        try:
            if not self.model:
                return "Model not loaded. Please check the model path.", None

            # Build context-aware prompt if session_id provided and use_context is True
            if use_context and session_id:
                enhanced_prompt, context_used = self._build_context_aware_prompt(prompt, session_id)
            else:
                enhanced_prompt = f"Human: {prompt}\nAssistant:"
                context_used = None

            # Generate response
            response = self.model(
                enhanced_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["Human:", "\n\n"]
            )

            response_text = response['choices'][0]['text'].strip()

            # Store conversation if session_id provided
            conversation_id = None
            if session_id:
                conversation = Conversation(
                    user_prompt=prompt,
                    ai_response=response_text,
                    session_id=session_id,
                    context_used=context_used
                )
                conversation_id = conversation_store.store_conversation(conversation)

            return response_text, conversation_id

        except Exception as e:
            return f"Error generating response: {e}", None

    def _build_context_aware_prompt(self, prompt: str, session_id: str = None) -> tuple[str, str]:
        """Build context-aware prompt using AI-generated natural language summary."""
        try:
            if session_id:
                session_summary = conversation_store.get_conversation_summary(session_id)

                if session_summary and session_summary.strip():
                    # Use the AI-generated natural language summary directly
                    enhanced_prompt = f"""[Previous conversation context: {session_summary}]

Human: {prompt}
Assistant:"""
                    return enhanced_prompt, session_summary
                else:
                    # No session history - new conversation
                    return f"Human: {prompt}\nAssistant:", None
            else:
                # No session ID provided
                return f"Human: {prompt}\nAssistant:", None

        except Exception as e:
            print(f"Error building context: {e}")
            return f"Human: {prompt}\nAssistant:", None

# Create a global instance
llm_handler = LLaMAHandler()