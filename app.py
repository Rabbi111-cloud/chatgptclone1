import os
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=["*"],  # Netlify access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMA ----------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ---------------- HF QUERY ----------------
def query_hf(prompt, retries=15, delay=5):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    for attempt in range(retries):
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data[0]["generated_text"]

        print(f"Model sleeping... retry {attempt+1}/{retries}")
        time.sleep(delay)

    return "⚠️ Model is still waking up. Please try again."

# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"status": "Backend running"}

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = "You are a helpful AI assistant.\n"

    for user, bot in req.history[-3:]:
        prompt += f"User: {user}\nAssistant: {bot}\n"

    prompt += f"User: {req.message}\nAssistant:"

    reply = query_hf(prompt)
    return {"reply": reply}

# ---------------- WAKE MODEL ----------------
@app.on_event("startup")
def wake_model():
    print("Waking Hugging Face model...")
    try:
        query_hf("Hello", retries=5, delay=3)
    except:
        pass

