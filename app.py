import os
import time
import requests
import gradio as gr

MODEL = "HuggingFaceH4/zephyr-7b-beta"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query(payload, retries=6):
    for _ in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json()
        time.sleep(5)
    return None

def chat(message, history):
    prompt = "You are a helpful AI assistant.\n\n"
    for user, bot in history[-3:]:
        prompt += f"User: {user}\nAssistant: {bot}\n"
    prompt += f"User: {message}\nAssistant:"

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
        return "⏳ Model is waking up. Please try again."

    return result[0]["generated_text"]

gr.ChatInterface(
    fn=chat,
    title="ChatGPT Clone (HF Inference API)"
).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
    inbrowser=False
)

