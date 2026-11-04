"""
Speech Service: STT (Speech to Text) and Audio Voice Handling
Supports Urdu, Roman Urdu, and English voice notes sent by farmers.
Powered by Groq Whisper Large V3 Turbo with resilient offline fallback.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_STT_MODEL, AUDIO_OUTPUT_DIR

class SpeechService:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.client = Groq(api_key=self.api_key, timeout=30.0) if self.api_key else None

    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribes a voice note in Urdu, Roman Urdu, Punjabi, or English.
        """
        if not os.path.exists(audio_file_path):
            return {"text": "", "error": f"Audio file not found: {audio_file_path}"}

        if not self.client:
            # Fallback simulated response if no API key
            return {
                "text": "Meri wheat ki fasal ke pattay peelay ho rahe hain, kya karun?",
                "language": "ur",
                "simulated": True
            }

        try:
            with open(audio_file_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), f),
                    model=GROQ_STT_MODEL,
                    response_format="verbose_json",
                    temperature=0.0
                )
                
                # Check detected language and text
                text = transcription.text.strip()
                lang = getattr(transcription, "language", "ur")
                return {
                    "text": text,
                    "language": lang,
                    "duration": getattr(transcription, "duration", None),
                    "simulated": False
                }
        except Exception as e:
            # Graceful error handling
            return {
                "text": "Meri wheat ki fasal ke pattay peelay ho rahe hain, kya karun?",
                "error": str(e),
                "simulated": True
            }

speech_service = SpeechService()

