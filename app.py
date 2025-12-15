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
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("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")
MODEL = "mistralai/mistral-7b-instruct"

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for user, bot in req.history[-5:]:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

    messages.append({"role": "user", "content": req.message})

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

    if response.status_code != 200:
        return {"reply": "⚠️ AI temporarily unavailable. Please try again."}

    data = response.json()
    return {"reply": data["choices"][0]["message"]["content"]}

