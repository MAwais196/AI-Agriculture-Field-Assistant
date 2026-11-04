"""
AI Agriculture Field Assistant - Main Entrypoint
Starts FastAPI web and API server.
"""
import sys
import uvicorn
from app.api import app

if __name__ == "__main__":
    print("Starting AI Agriculture Field Assistant (Punjab Agriculture Department)...")
    print("Server running at: http://127.0.0.1:8000")
    print("Open browser: http://127.0.0.1:8000")
    print("API Docs:     http://127.0.0.1:8000/docs")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

