"""
🚀 AI Chat Backend using Google Gen AI SDK
FastAPI backend that connects to fine-tuned Gemini models on Vertex AI
using the unified Google Gen AI SDK for better compatibility.
"""

import logging
import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server_genai")

# Import Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
    logger.info("✅ Google Gen AI SDK imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Google Gen AI SDK: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="AI Chat Backend - Gen AI SDK",
    description="FastAPI backend using Google Gen AI SDK for fine-tuned Vertex AI models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192  # Increased for detailed responses

class ChatResponse(BaseModel):
    response: str
    model_used: Optional[str] = "vertex-ai-endpoint"

# Minimal text cleaning function to preserve content quality
def clean_response_text(text: str) -> str:
    """Minimal cleaning to preserve AI response quality"""
    if not text:
        return text
    
    # Only remove leading/trailing whitespace and normalize line endings
    cleaned = text.strip()
    
    # Normalize line endings but preserve formatting
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    
    return cleaned

# Global variables
client = None
model_endpoint = None
project_id = None
location = None

@app.on_event("startup")
async def startup_event():
    """Initialize the backend configuration on startup"""
    global project_id, location, model_endpoint
    
    logger.info("🚀 Starting AI Chat Backend with Google Gen AI SDK...")
    
    try:
        # Set configuration directly - consistent with successful test script
        project_id = "bright-coyote-463315-q8"
        location = "us-west1" 
        model_endpoint = "projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824"
        
        logger.info(f"🔧 Project: {project_id}")
        logger.info(f"🔧 Location: {location}")
        logger.info(f"🔧 Endpoint: {model_endpoint}")
        logger.info("🔐 Using Application Default Credentials (like successful test)")
        
        # Test client creation - following successful test script format exactly
        test_client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        logger.info("✅ Google Gen AI client initialized successfully")
        logger.info(f"✅ Using fine-tuned model endpoint: {model_endpoint}")
        logger.info("✅ Backend initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize backend: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Chat Backend (Google Gen AI SDK) is running",
        "model_endpoint": model_endpoint,
        "backend_version": "2.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using the correct Google Gen AI SDK approach"""
    logger.info(f"📨 Received chat request: {request.message[:50]}...")
    
    try:
        # Create client - following successful test script format exactly
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        
        # Use endpoint path as model name
        model = model_endpoint
        
        # Create content - following successful test script format exactly
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=request.message)
                ]
            ),
        ]
        
        # Enhanced generation config to match Google Cloud Console settings
        generate_content_config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=0.95,  # More conservative for better quality
            top_k=40,    # Add top_k for better control
            max_output_tokens=request.max_tokens,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
        )
        
        logger.info(f"🚀 Calling fine-tuned model: {model}")
        logger.info(f"🔧 Config: temp={request.temperature}, max_tokens={request.max_tokens}, top_p=0.95, top_k=40")
        
        # Call model - following successful test script format exactly
        response_text = ""
        chunk_count = 0
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
                chunk_count += 1
        
        # Minimal cleaning to preserve content quality
        cleaned_response = clean_response_text(response_text)
        
        logger.info(f"✅ Response generated: {chunk_count} chunks, {len(response_text)} chars -> {len(cleaned_response)} chars")
        logger.info(f"📄 Response preview: {cleaned_response[:100]}..." if len(cleaned_response) > 100 else f"📄 Full response: {cleaned_response}")
        
        return ChatResponse(
            response=cleaned_response,
            model_used=f"vertex-ai-endpoint-{model_endpoint.split('/')[-1]}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate response: {str(e)}"
        )


@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint"""
    logger.info(f"📨 Received streaming chat request: {request.message[:50]}...")
    
    def generate():
        try:
            # Create client
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            
            # Use the endpoint path as model name
            model = model_endpoint
            
            # Create contents
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=request.message)
                    ]
                ),
            ]
            
            # Create config
            generate_content_config = types.GenerateContentConfig(
                temperature=request.temperature,
                top_p=0.95,
                seed=0,
                max_output_tokens=request.max_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ],
            )
            
            # Stream response with text cleaning
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    # Clean each chunk before sending
                    cleaned_chunk = clean_response_text(chunk.text)
                    if cleaned_chunk:  # Only send if there's content after cleaning
                        yield f"data: {json.dumps({'text': cleaned_chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Error in streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/test-chat", response_model=ChatResponse)
async def test_chat_endpoint(request: ChatRequest):
    """Test chat endpoint using regular Gemini API"""
    logger.info(f"📨 Testing with regular Gemini API: {request.message[:50]}...")
    
    try:
        # Create a client for regular Gemini API
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found")
            
        test_client = genai.Client(api_key=gemini_api_key)
        
        # Create content for the request
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.message)]
            )
        ]
        
        # Configure generation parameters
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens
        )
        
        # Generate content using regular Gemini
        response = test_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config=config
        )
        
        # Extract and clean the response text
        response_text = response.text if response.text else "No response generated"
        cleaned_response = clean_response_text(response_text)
        
        logger.info(f"✅ Generated test response: {len(response_text)} chars -> {len(cleaned_response)} chars (cleaned)")
        
        return ChatResponse(
            response=cleaned_response,
            model_used="gemini-1.5-flash (test)"
        )
        
    except Exception as e:
        logger.error(f"❌ Error in test endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in test endpoint: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 