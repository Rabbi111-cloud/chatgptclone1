import os
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so Netlify frontend can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or list your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face model settings
MODEL = "HuggingFaceH4/zephyr-7b-beta"  # free model
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Must set in Render env variables

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Request body
class ChatRequest(BaseModel):
    message: str

# Query HF Inference API
def query(payload, retries=5):
    for _ in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()
        time.sleep(5)
    return None

# Health check
@app.get("/")
def health():
    return {"status": "ok"}

# Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"""
You are ChatGPT, a helpful and friendly AI assistant.
Answer in plain English. 
Do NOT write code unless asked.
Do NOT output JSON, JavaScript, or HTML.

User: {req.message}
Assistant:
""".strip()

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    result = query(payload)

    if not result or not isinstance(result, list):
        return {"reply": "The model is waking up. Please try again in a few seconds."}

    return {"reply": result[0]["generated_text"].strip()}
