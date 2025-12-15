import os
import time
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- CONFIG ----------------
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

# ---------------- HF CALL ----------------
def call_hf(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    for i in range(20):
        r = requests.post(HF_API_URL, headers=HEADERS, json=payload)

        # SUCCESS
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]

        # MODEL LOADING
        time.sleep(3)

    # NEVER RETURN EMPTY
    return "⚠️ The AI is busy right now. Please send your message again."

# ---------------- ROUTES ----------------
@app.get("/")
def health():
    return {"status": "OK"}

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = "You are a helpful assistant.\n\n"

    for u, a in req.history[-3:]:
        prompt += f"User: {u}\nAssistant: {a}\n"

    prompt += f"User: {req.message}\nAssistant:"

    reply = call_hf(prompt)

    return {
        "reply": reply
    }

# ---------------- WARMUP ----------------
@app.on_event("startup")
def warmup():
    print("Warming model...")
    call_hf("Hello")

