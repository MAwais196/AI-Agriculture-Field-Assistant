"""
Configuration settings for AI Agriculture Field Assistant (Punjab, Pakistan)
"""
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_OUTPUT_DIR = BASE_DIR / "app" / "web" / "static" / "audio"
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Qdrant Vector DB Settings (Local embedded mode by default or remote)
QDRANT_PATH = str(DATA_DIR / "qdrant_db")
COLLECTION_NAME = "punjab_agriculture_guidelines"

# Models
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

# Punjab Agriculture Extension & Helpline Contacts
PUNJAB_AGRI_HELPLINE = "0800-15000 / 0800-29000"
PUNJAB_PEST_WARNING_URL = "https://www.agripunjab.gov.pk/index.php/pestwarning-services"
PUNJAB_PEST_WARNING_OFFICE = "Directorate General Pest Warning & Quality Control of Pesticides, Lahore, Punjab"

# Major Punjab Agricultural Districts and GPS Coordinates (for accurate Weather API queries)
PUNJAB_DISTRICTS = {
    "faisalabad": {"name": "Faisalabad", "lat": 31.4504, "lon": 73.1350, "zone": "Central Mixed Crop Zone"},
    "multan": {"name": "Multan", "lat": 30.1575, "lon": 71.5249, "zone": "Cotton-Wheat Zone"},
    "bahawalpur": {"name": "Bahawalpur", "lat": 29.3544, "lon": 71.6911, "zone": "Cotton-Wheat Zone"},
    "sahiwal": {"name": "Sahiwal", "lat": 30.6682, "lon": 73.1114, "zone": "Mixed / Maize-Potato-Wheat Zone"},
    "rawalpindi": {"name": "Rawalpindi", "lat": 33.5651, "lon": 73.0169, "zone": "Rainfed / Barani Zone"},
    "sargodha": {"name": "Sargodha", "lat": 32.0836, "lon": 72.6711, "zone": "Citrus / Mixed Zone"},
    "gujranwala": {"name": "Gujranwala", "lat": 32.1877, "lon": 74.1945, "zone": "Rice-Wheat Zone"},
    "lahore": {"name": "Lahore", "lat": 31.5204, "lon": 74.3587, "zone": "Mixed / Peri-urban Zone"},
    "rahim_yar_khan": {"name": "Rahim Yar Khan", "lat": 28.4212, "lon": 70.2989, "zone": "Cotton-Wheat Zone"},
    "jhang": {"name": "Jhang", "lat": 31.2781, "lon": 72.3317, "zone": "Sugarcane-Wheat Zone"},
    "sheikhupura": {"name": "Sheikhupura", "lat": 31.7131, "lon": 73.9783, "zone": "Rice-Wheat Zone"},
    "khanewal": {"name": "Khanewal", "lat": 30.3017, "lon": 71.9321, "zone": "Cotton-Wheat Zone"},
    "vehari": {"name": "Vehari", "lat": 30.0419, "lon": 72.3528, "zone": "Cotton-Wheat Zone"},
    "okara": {"name": "Okara", "lat": 30.8081, "lon": 73.4458, "zone": "Maize-Potato-Wheat Zone"},
    "chakwal": {"name": "Chakwal", "lat": 32.9328, "lon": 72.8631, "zone": "Barani Zone"}
}

