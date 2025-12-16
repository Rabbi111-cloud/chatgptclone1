# chatgptclone1

# ChatGPT Clone (Free Tier)

A lightweight ChatGPT clone built with **FastAPI** (backend) and a simple **HTML/JS frontend** hosted on **Netlify**.  
This project uses **OpenRouter’s free AI models** for AI responses, so it’s fully serverless and free to run.

---

## Features

- Chat interface with AI replies
- Keeps last 5 messages in conversation context
- Fully free OpenRouter AI model (`mistralai/mistral-7b-instruct`)
- Serverless backend on Render
- Frontend hosted on Netlify
- Handles AI downtime gracefully

---

## Tech Stack

- **Backend:** FastAPI, Requests
- **Frontend:** HTML, JavaScript
- **AI Model:** OpenRouter (Mistral 7B)
- **Hosting:** Render (backend), Netlify (frontend)

---

## Demo

**Backend URL:** `https://your-backend.onrender.com`  
**Frontend URL:** `https://your-frontend.netlify.app`  

---

## Setup & Deployment

### Backend (Render)

1. Fork or clone this repo.
2. Add `OPENROUTER_API_KEY` as an environment variable in Render.
3. Start command:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
