import os
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL = "HuggingFaceH4/zephyr-7b-beta"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class ChatRequest(BaseModel):
    message: str
    history: list = []

def query(payload, retries=5):
    for _ in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()
        time.sleep(5)
    return None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = "You are a helpful AI assistant.\n\n"

    for user, bot in req.history[-3:]:
        prompt += f"User: {user}\nAssistant: {bot}\n"

    prompt += f"User: {req.message}\nAssistant:"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    result = query(payload)

    if not result or not isinstance(result, list):
        return {"reply": "Model is warming up. Please try again."}

    return {"reply": result[0]["generated_text"]}
