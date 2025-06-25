#!/usr/bin/env python3
"""
Simple Malaysia AI Travel Guide - Guaranteed to Work Version
"""

import os
import sys
import logging
from typing import Dict, Any
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# Initialize FastAPI app
app = FastAPI(
    title="Malaysia AI Travel Guide - Simple",
    description="A simple travel guide for Malaysia using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI model
gemini_model = None

def setup_ai():
    """Setup Gemini AI model"""
    global gemini_model
    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found, AI features disabled")
            return False
            
        genai.configure(api_key=gemini_api_key)
        
        # Try different model options
        model_options = [
            "TourismMalaysiaAI",  # Your fine-tuned model
            "gemini-1.5-flash",
            "gemini-pro"
        ]
        
        for model_name in model_options:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                logger.info(f"✅ AI model connected: {model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to connect to {model_name}: {e}")
                continue
                
        logger.error("❌ Failed to connect to any AI model")
        return False
        
    except Exception as e:
        logger.error(f"AI setup failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting Malaysia AI Travel Guide...")
    setup_ai()
    logger.info("✅ Application startup complete!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Malaysia AI Travel Guide!",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "status": "/api/status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "malaysia-ai-guide",
        "version": "1.0.0",
        "ai_enabled": gemini_model is not None
    }

@app.get("/api/status")
async def get_status():
    """Get application status"""
    return {
        "is_ready": True,
        "ai_model_available": gemini_model is not None,
        "service": "malaysia-ai-travel-guide",
        "version": "1.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI travel guide"""
    try:
        if not gemini_model:
            # Fallback response without AI
            return ChatResponse(
                response="Welcome to Malaysia! I'm here to help you explore this beautiful country. "
                       "Malaysia offers amazing destinations like Kuala Lumpur, Penang, Langkawi, "
                       "and the Cameron Highlands. What would you like to know about?",
                status="fallback"
            )
        
        # Prepare prompt for Malaysia tourism
        system_prompt = """You are a knowledgeable Malaysia travel guide. Help users discover the best of Malaysia including:
        - Tourist attractions and destinations
        - Local culture and traditions
        - Food recommendations
        - Travel tips and advice
        - Transportation options
        
        Keep responses helpful, friendly, and focused on Malaysia tourism."""
        
        full_prompt = f"{system_prompt}\n\nUser: {request.message}\n\nAssistant:"
        
        # Generate AI response
        response = gemini_model.generate_content(full_prompt)
        
        return ChatResponse(
            response=response.text,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

@app.get("/api/destinations")
async def get_destinations():
    """Get popular Malaysia destinations"""
    destinations = [
        {
            "name": "Kuala Lumpur",
            "description": "Capital city with iconic Petronas Twin Towers",
            "category": "city",
            "coordinates": [3.1319, 101.6841]
        },
        {
            "name": "Penang (George Town)",
            "description": "UNESCO World Heritage site with rich culture",
            "category": "heritage",
            "coordinates": [5.4164, 100.3327]
        },
        {
            "name": "Langkawi",
            "description": "Tropical paradise with beautiful beaches",
            "category": "beach",
            "coordinates": [6.3500, 99.8000]
        },
        {
            "name": "Cameron Highlands",
            "description": "Cool mountain retreat with tea plantations",
            "category": "nature",
            "coordinates": [4.4956, 101.3781]
        },
        {
            "name": "Batu Caves",
            "description": "Sacred Hindu temple in limestone caves",
            "category": "religious",
            "coordinates": [3.2379, 101.6840]
        }
    ]
    
    return {
        "destinations": destinations,
        "total": len(destinations)
    }

if __name__ == "__main__":
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🌐 Starting server on port {port}")
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=30,
        access_log=False  # Reduce log noise
    ) 