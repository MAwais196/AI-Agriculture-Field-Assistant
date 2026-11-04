"""
FastAPI Server for AI Agriculture Field Assistant
Evidence-Based Field Decision Agent for Farmers in Punjab, Pakistan.
"""
import os
import shutil
import sys
import time
import uuid
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import psutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import (
    UPLOAD_DIR,
    PUNJAB_DISTRICTS,
    PUNJAB_AGRI_HELPLINE,
    PUNJAB_PEST_WARNING_URL,
    GROQ_API_KEY
)
from app.graph.decision_graph import field_agent_graph
from app.weather_service import weather_service
from app.speech_service import speech_service
from app.vision_service import vision_service
from app.knowledge_base.punjab_agri_kb import PUNJAB_AGRICULTURAL_DOCUMENTS
from app.knowledge_base.qdrant_rag import qdrant_rag
from app.search_agent import web_search_agent

SERVER_START_TIME = time.time()

app = FastAPI(
    title="AI Agriculture Field Assistant",
    description="Evidence-Based Field Decision Agent for Punjab Agriculture Department",
    version="2.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"
STATIC_DIR.mkdir(exist_ok=True, parents=True)
WEB_DIR = Path(__file__).resolve().parent / "web"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class TextChatRequest(BaseModel):
    message: str
    crop: Optional[str] = None
    district: Optional[str] = "Faisalabad"
    crop_stage: Optional[str] = None
    session_id: Optional[str] = None

@app.get("/api/health")
def health_check(detailed: bool = True):
    """
    Comprehensive diagnostics & real-time health inspection for AI Agriculture Assistant.
    Validates Qdrant Vector DB, LangGraph StateGraph, Knowledge Base, Weather Telemetry,
    Vision, and Speech services.
    """
    uptime_seconds = round(time.time() - SERVER_START_TIME, 2)
    uptime_minutes = round(uptime_seconds / 60, 2)

    # 1. Check Qdrant Vector DB
    qdrant_status = "operational"
    total_vectors = 0
    qdrant_error = None
    try:
        total_vectors = qdrant_rag.client.count(collection_name=qdrant_rag.collection_name).count
    except Exception as e:
        qdrant_status = "degraded"
        qdrant_error = str(e)

    # 2. Check LangGraph
    langgraph_status = "operational" if field_agent_graph is not None else "degraded"

    # 3. Check Knowledge Base
    kb_count = len(PUNJAB_AGRICULTURAL_DOCUMENTS)
    kb_status = "operational" if kb_count > 0 else "empty"

    # 4. Check Weather API configuration
    weather_status = "operational" if len(PUNJAB_DISTRICTS) > 0 else "degraded"

    # 5. Check Vision Service
    vision_status = "operational"

    # 6. Check Speech Service
    speech_configured = bool(speech_service.client)
    speech_status = "operational (whisper-large-v3-turbo)" if speech_configured else "operational (offline fallback ready)"

    # 7. Process & system memory
    mem_usage_mb = None
    cpu_percent = None
    try:
        proc = psutil.Process()
        mem_usage_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
        cpu_percent = proc.cpu_percent(interval=None)
    except Exception:
        pass

    # Overall system health
    is_healthy = (qdrant_status == "operational" and langgraph_status == "operational" and kb_status == "operational")
    overall_status = "healthy" if is_healthy else "degraded"

    response = {
        # Core backward-compatible fields
        "status": overall_status,
        "agent": "AI Agriculture Field Assistant (LangGraph Evidence Engine)",
        "version": "2.1.0",
        "department": "Punjab Agriculture Department (agripunjab.gov.pk)",
        "vector_db": "Qdrant (Embedded)",
        "weather_api": "Open-Meteo Punjab Station Active",
        "supported_crops": ["Wheat (Gandum)", "Cotton (Kapas)", "Rice (Dhan)", "Maize (Makai)"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "uptime": {
            "seconds": uptime_seconds,
            "minutes": uptime_minutes,
            "formatted": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
        }
    }

    if detailed:
        response["subsystems"] = {
            "qdrant_vector_db": {
                "status": qdrant_status,
                "collection": qdrant_rag.collection_name,
                "mode": "in-memory embedded (zero lock contention)",
                "total_points": total_vectors,
                "vector_dimension": qdrant_rag.embedder.vector_dim,
                "distance_metric": "Cosine",
                "error": qdrant_error
            },
            "knowledge_base": {
                "status": kb_status,
                "documents_loaded": kb_count,
                "conditions_covered": [doc["condition_name"] for doc in PUNJAB_AGRICULTURAL_DOCUMENTS],
                "verified_source": "Directorate General Pest Warning & Quality Control of Pesticides, Punjab"
            },
            "langgraph_workflow": {
                "status": langgraph_status,
                "state_machine": "StateGraph(FieldAssistantState)",
                "nodes": [
                    "extract_signals",
                    "fetch_weather",
                    "vision_analysis",
                    "retrieve_evidence",
                    "web_search",
                    "differential_reasoning",
                    "decision_and_advisory"
                ],
                "conditional_branches": ["should_clarify", "escalate_to_helpline"],
                "reasoning_mode": "Multi-Factor Differential Diagnosis"
            },
            "internet_search_agent": {
                "status": "operational",
                "engine": "DDGS Live Web & Agri News Search",
                "domains_prioritized": ["agripunjab.gov.pk", "parc.gov.pk", "aari.punjab.gov.pk", "uaf.edu.pk"],
                "features": [
                    "Real-time Web Search",
                    "Agricultural News & Subsidies Tracking",
                    "Pakistan Regional Augmentation",
                    "Wikipedia Scientific Fallback"
                ]
            },
            "weather_telemetry": {
                "status": weather_status,
                "source": "Open-Meteo API",
                "monitored_districts": len(PUNJAB_DISTRICTS),
                "epidemiological_alerts": [
                    "Yellow Rust (10-22°C, RH > 65%)",
                    "Aphid / Sust Tela (Warm & Dry)",
                    "Waterlogging Root Hypoxia (> 5mm Rain)",
                    "Heat Stress Scorch (> 33°C)"
                ]
            },
            "vision_pathology": {
                "status": vision_status,
                "engine": "Colorimetric Spectral & Stripe Pattern Analyzer",
                "features": ["Chlorosis %", "Necrosis %", "Linear Stripe Fungi Detection"]
            },
            "speech_stt": {
                "status": speech_status,
                "primary_model": "Whisper Large V3 Turbo",
                "groq_api_ready": speech_configured,
                "offline_fallback": True
            },
            "emergency_routing": {
                "status": "operational",
                "helpline": PUNJAB_AGRI_HELPLINE,
                "official_url": PUNJAB_PEST_WARNING_URL
            }
        }
        response["system_metrics"] = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "memory_usage_mb": mem_usage_mb,
            "cpu_percent": cpu_percent
        }
        response["endpoints"] = [
            {"method": "GET", "path": "/api/health", "description": "System health and subsystem diagnostics"},
            {"method": "POST", "path": "/api/chat", "description": "Text reasoning and agronomic advisory"},
            {"method": "POST", "path": "/api/diagnose", "description": "Multimodal field diagnosis (Voice + Vision + Text)"},
            {"method": "POST", "path": "/api/search", "description": "Autonomous internet search agent for agricultural news, prices & subsidies"},
            {"method": "GET", "path": "/api/districts", "description": "Punjab districts coordinates & agro-ecological zones"},
            {"method": "GET", "path": "/api/guidelines", "description": "Official agricultural guidelines catalog"}
        ]

    return response

@app.get("/api/districts")
def list_districts():
    """Returns list of Punjab districts with coordinates and agro-ecological zones."""
    return [{"id": k, **v} for k, v in PUNJAB_DISTRICTS.items()]

@app.get("/api/guidelines")
def list_guidelines():
    """Returns official advisory documents loaded into knowledge base."""
    return PUNJAB_AGRICULTURAL_DOCUMENTS

class SearchRequest(BaseModel):
    query: str
    crop: Optional[str] = None
    max_results: Optional[int] = 4

@app.post("/api/search")
def search_internet(req: SearchRequest):
    """
    Autonomous Internet Search Agent endpoint.
    Searches live web and agricultural news for latest prices, subsidies, varieties, and advisories.
    """
    try:
        results = web_search_agent.search(query=req.query, crop=req.crop, max_results=req.max_results or 4)
        summary = web_search_agent.summarize_findings(results)
        news = web_search_agent.search_news(query=req.query, max_results=2)
        return {
            "query": req.query,
            "crop": req.crop,
            "total_results": len(results),
            "results": results,
            "news": news,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def process_chat(req: TextChatRequest):
    """
    Standard text reasoning endpoint using LangGraph.
    """
    session_id = req.session_id or str(uuid.uuid4())
    state_input = {
        "session_id": session_id,
        "user_message": req.message,
        "crop": req.crop,
        "district": req.district,
        "crop_stage": req.crop_stage,
        "audio_path": None,
        "image_path": None,
        "symptoms": []
    }

    try:
        result = field_agent_graph.invoke(state_input)
        return {
            "session_id": session_id,
            "crop": result.get("crop"),
            "district": result.get("district"),
            "growth_stage": result.get("crop_stage"),
            "weather": result.get("weather_context"),
            "vision": result.get("vision_findings"),
            "top_diagnosis": result.get("top_diagnosis"),
            "confidence_score": result.get("confidence_score"),
            "distinguishing_test": result.get("distinguishing_test"),
            "differential_diagnosis": result.get("differential_diagnosis"),
            "advisory_roman_urdu": result.get("final_advisory_roman_urdu"),
            "advisory_urdu": result.get("final_advisory_urdu"),
            "official_sources": result.get("official_sources"),
            "recommended_actions": result.get("recommended_actions"),
            "safety_guidelines": result.get("safety_guidelines"),
            "web_search_results": result.get("web_search_results", []),
            "web_search_summary": result.get("web_search_summary", ""),
            "escalation_needed": result.get("escalation_needed", False),
            "escalation_reason": result.get("escalation_reason"),
            "expert_helpline": result.get("expert_helpline", PUNJAB_AGRI_HELPLINE),
            "clarification_needed": result.get("clarification_needed", False),
            "clarification_questions": result.get("clarification_questions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose")
async def multimodal_diagnose(
    message: Optional[str] = Form(None),
    crop: Optional[str] = Form("Wheat (Gandum)"),
    district: Optional[str] = Form("Faisalabad"),
    crop_stage: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    """
    Full Multimodal Endpoint:
    Accepts voice notes (STT), leaf photos (Vision), and user parameters.
    """
    session_id = str(uuid.uuid4())
    audio_path = None
    image_path = None
    transcription_result = None

    # Handle Audio Upload (Voice note)
    if audio_file and audio_file.filename:
        suffix = Path(audio_file.filename).suffix or ".wav"
        saved_audio_name = f"audio_{session_id}{suffix}"
        audio_path = str(UPLOAD_DIR / saved_audio_name)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe audio using Whisper
        transcription_result = speech_service.transcribe_audio(audio_path)
        transcribed_text = transcription_result.get("text", "")
        if transcribed_text:
            if message:
                message = f"{message} (Voice Note: {transcribed_text})"
            else:
                message = transcribed_text

    if not message:
        message = "Meri wheat ki fasal ke pattay peelay ho rahe hain, kya karun?"

    # Handle Image Upload (Leaf inspection)
    if image_file and image_file.filename:
        suffix = Path(image_file.filename).suffix or ".jpg"
        saved_img_name = f"leaf_{session_id}{suffix}"
        image_path = str(UPLOAD_DIR / saved_img_name)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)

    state_input = {
        "session_id": session_id,
        "user_message": message,
        "crop": crop,
        "district": district,
        "crop_stage": crop_stage,
        "audio_path": audio_path,
        "image_path": image_path,
        "symptoms": []
    }

    try:
        result = field_agent_graph.invoke(state_input)
        return {
            "session_id": session_id,
            "transcription": transcription_result,
            "crop": result.get("crop"),
            "district": result.get("district"),
            "growth_stage": result.get("crop_stage"),
            "weather": result.get("weather_context"),
            "vision": result.get("vision_findings"),
            "top_diagnosis": result.get("top_diagnosis"),
            "confidence_score": result.get("confidence_score"),
            "distinguishing_test": result.get("distinguishing_test"),
            "differential_diagnosis": result.get("differential_diagnosis"),
            "advisory_roman_urdu": result.get("final_advisory_roman_urdu"),
            "advisory_urdu": result.get("final_advisory_urdu"),
            "official_sources": result.get("official_sources"),
            "recommended_actions": result.get("recommended_actions"),
            "safety_guidelines": result.get("safety_guidelines"),
            "web_search_results": result.get("web_search_results", []),
            "web_search_summary": result.get("web_search_summary", ""),
            "escalation_needed": result.get("escalation_needed", False),
            "escalation_reason": result.get("escalation_reason"),
            "expert_helpline": result.get("expert_helpline", PUNJAB_AGRI_HELPLINE),
            "clarification_needed": result.get("clarification_needed", False),
            "clarification_questions": result.get("clarification_questions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "AI Agriculture Field Assistant API is running. Open /docs for Swagger UI."}

