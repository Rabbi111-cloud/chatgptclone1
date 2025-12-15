from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Allow Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mistral-7b-instruct"  # free, reliable model

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    # Keep last 5 messages in history
    for user, bot in req.history[-5:]:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": req.message})

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Safely parse AI reply
        reply = None
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"]
            elif "text" in choice:
                reply = choice["text"]

        if not reply:
            reply = "⚠️ AI temporarily unavailable. Please try again."

        return {"reply": reply}

    except Exception as e:
        return {"reply": "⚠️ AI temporarily unavailable. Please try again."}
