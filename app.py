import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# ==============================
# Load model (TinyLlama 1.1B)
# ==============================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()
print("Model loaded successfully.")

# ==============================
# Parameters
# ==============================
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 100
MAX_HISTORY_TURNS = 3  # limit chat history

SYSTEM_PROMPT = "You are a helpful AI assistant."

# ==============================
# Build prompt safely
# ==============================
def build_prompt(message, history):
    prompt = SYSTEM_PROMPT + "\n\n"
    history = history[-MAX_HISTORY_TURNS:]
    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"
    prompt += f"User: {message}\nAssistant:"
    return prompt

# ==============================
# Chat function
# ==============================
def chat(message, history):
    prompt = build_prompt(message, history)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the assistant reply
    if "Assistant:" in decoded:
        reply = decoded.split("Assistant:")[-1].strip()
    else:
        reply = decoded.strip()

    return reply

# ==============================
# Gradio interface
# ==============================
gr.ChatInterface(
    fn=chat,
    title="ChatGPT Clone (TinyLlama on Render)",
    description="Stable backend using TinyLlama 1.1B. Works on Render free/paid tier."
).launch()
