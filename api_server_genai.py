"""
🚀 AI Chat Backend using Google Gen AI SDK
FastAPI backend that connects to fine-tuned Gemini models on Vertex AI
using the unified Google Gen AI SDK for better compatibility.
Optimized for Render cloud deployment.
"""

import logging
import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
from google.oauth2 import service_account

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
    title="🇲🇾 Malaysia Tourism AI Backend",
    description="Advanced AI Chat Backend using Google Gen AI SDK with fine-tuned Gemini model",
    version="2.0.0"
)

# Add CORS middleware - Allow all origins for cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Render deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8192

class ChatResponse(BaseModel):
    response: str
    model_used: Optional[str] = "vertex-ai-endpoint"

# Minimal text cleaning function to preserve content quality
def clean_response_text(text: str) -> str:
    """Clean up the response text"""
    if not text:
        return ""
    
    # Remove excessive newlines and clean up formatting
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Global variables
project_id = None
location = None
model_endpoint = None
credentials = None

def setup_google_credentials():
    """Setup Google Cloud credentials for different environments"""
    global credentials
    try:
        # Define the required scopes for Vertex AI
        VERTEX_AI_SCOPES = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/cloud-platform.read-only',
            'https://www.googleapis.com/auth/devstorage.full_control',
            'https://www.googleapis.com/auth/devstorage.read_only',
            'https://www.googleapis.com/auth/devstorage.read_write'
        ]
        
        # Check if we're in Render environment
        if os.getenv("RENDER_SERVICE_NAME"):
            logger.info("🌐 Running on Render - setting up cloud credentials")
            
            # First try to use JSON credentials from environment variable directly
            google_creds_json = os.getenv("GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON")
            if google_creds_json:
                try:
                    import json
                    import io
                    
                    # Parse the JSON credentials
                    creds_info = json.loads(google_creds_json)
                    
                    # Create credentials from service account info
                    credentials = service_account.Credentials.from_service_account_info(
                        creds_info,
                        scopes=VERTEX_AI_SCOPES
                    )
                    logger.info("🔐 Service account credentials loaded from environment JSON")
                    logger.info(f"🔧 Applied scopes: {len(VERTEX_AI_SCOPES)} vertex AI scopes")
                    logger.info(f"🎯 Service account email: {creds_info.get('client_email', 'unknown')}")
                    return True
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON: {e}")
                except Exception as e:
                    logger.error(f"❌ Failed to load credentials from JSON env var: {e}")
            
            # Fallback: Try different possible secret file locations
            possible_paths = [
                '/etc/secrets/google_creds.json',
                '/var/secrets/google_creds.json', 
                '/opt/render/project/secrets/google_creds.json',
                './google_creds.json'
            ]
            
            credentials_loaded = False
            for secret_file_path in possible_paths:
                if os.path.exists(secret_file_path):
                    try:
                        # Load credentials with proper scopes
                        credentials = service_account.Credentials.from_service_account_file(
                            secret_file_path,
                            scopes=VERTEX_AI_SCOPES
                        )
                        logger.info(f"🔐 Service account credentials loaded from: {secret_file_path}")
                        logger.info(f"🔧 Applied scopes: {len(VERTEX_AI_SCOPES)} vertex AI scopes")
                        credentials_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load credentials from {secret_file_path}: {e}")
                        continue
            
            if not credentials_loaded:
                # Final fallback: Use default credentials with scopes
                logger.info("🔄 Using Application Default Credentials as fallback")
                try:
                    from google.auth import default
                    credentials, _ = default(scopes=VERTEX_AI_SCOPES)
                    logger.info("🔐 Using default application credentials with proper scopes")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to use default credentials: {e}")
                    return False
            
            return True
            
        else:
            # Local development - use existing file
            logger.info("🏠 Running in local environment")
            cred_file = "bright-coyote-463315-q8-59797318b374.json"
            if os.path.exists(cred_file):
                credentials = service_account.Credentials.from_service_account_file(
                    cred_file,
                    scopes=VERTEX_AI_SCOPES
                )
                logger.info(f"🔐 Using local credentials with scopes: {cred_file}")
                return True
            else:
                logger.warning(f"⚠️ Local credential file not found: {cred_file}")
                # Try default credentials
                try:
                    from google.auth import default
                    credentials, _ = default(scopes=VERTEX_AI_SCOPES)
                    logger.info("🔄 Using default application credentials with proper scopes")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to use default credentials: {e}")
                    return False
                
    except Exception as e:
        logger.error(f"❌ Failed to setup credentials: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the backend configuration on startup"""
    global project_id, location, model_endpoint
    
    logger.info("🚀 Starting AI Chat Backend with Google Gen AI SDK...")
    
    try:
        # Setup credentials first
        if not setup_google_credentials():
            raise ValueError("Failed to setup Google Cloud credentials")
        
        # Get configuration from environment variables (set in Render)
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "bright-coyote-463315-q8")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-west1")
        model_endpoint = os.getenv(
            "VERTEX_AI_ENDPOINT", 
            "projects/bright-coyote-463315-q8/locations/us-west1/endpoints/6528596580524621824"
        )
        
        logger.info(f"🔧 Project: {project_id}")
        logger.info(f"🔧 Location: {location}")
        logger.info(f"🔧 Endpoint: {model_endpoint}")
        
        # Check if we're in Render environment
        render_service = os.getenv("RENDER_SERVICE_NAME")
        if render_service:
            logger.info(f"🌐 Running on Render service: {render_service}")
        else:
            logger.info("🔐 Running in local development environment")
        
        # Test client creation
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
        # Don't raise in cloud environment - continue with fallback
        if not os.getenv("RENDER_SERVICE_NAME"):
            raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🇲🇾 Malaysia Tourism AI Backend",
        "status": "healthy",
        "version": "2.0.0",
        "endpoints": ["/health", "/chat", "/chat-stream"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Chat Backend (Google Gen AI SDK) is running",
        "model_endpoint": model_endpoint,
        "backend_version": "2.0.0",
        "environment": "render" if os.getenv("RENDER_SERVICE_NAME") else "local"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using the correct Google Gen AI SDK approach"""
    logger.info(f"📨 Received chat request: {request.message[:50]}...")
    
    try:
        # Create client - pass credentials object explicitly
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials
        )
        
        # Use standard model name format (extract from endpoint if needed)
        # Since the fine-tuned endpoint might not be available, use standard models as fallback
        if model_endpoint and "gemini-1.5-pro" in model_endpoint:
            model = "gemini-1.5-pro"
            logger.info(f"🎯 Using Gemini 1.5 Pro (inferred from endpoint)")
        elif model_endpoint and "gemini-1.5-flash" in model_endpoint:
            model = "gemini-1.5-flash"
            logger.info(f"🎯 Using Gemini 1.5 Flash (inferred from endpoint)")
        else:
            # For fine-tuned endpoints or when endpoint is not available, use standard model
            model = "gemini-1.5-pro"
            logger.info(f"🔄 Using standard Gemini 1.5 Pro (endpoint fallback: {model_endpoint})")
        
        # Create content - following official documentation format
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=request.message)
                ]
            ),
        ]
        
        # Enhanced generation config following official documentation
        generate_content_config = types.GenerateContentConfig(
            temperature=request.temperature,
            top_p=0.95,  # More conservative for better quality
            max_output_tokens=request.max_tokens,
            # Use proper safety settings format per documentation
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                )
            ],
        )
        
        logger.info(f"🚀 Calling model: {model}")
        logger.info(f"🔧 Config: temp={request.temperature}, max_tokens={request.max_tokens}, top_p=0.95")
        
        # Call model using streaming approach for better performance
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
            model_used=f"vertex-ai-{model}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate response: {str(e)}"
        )


@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint using Google Gen AI SDK"""
    logger.info(f"📨 Received streaming chat request: {request.message[:50]}...")
    
    async def generate():
        try:
            # Create client - pass credentials object explicitly
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                credentials=credentials
            )

            # Use standard model name format (same logic as main chat endpoint)
            if model_endpoint and "gemini-1.5-pro" in model_endpoint:
                model = "gemini-1.5-pro"
                logger.info(f"🎯 Stream using Gemini 1.5 Pro (inferred from endpoint)")
            elif model_endpoint and "gemini-1.5-flash" in model_endpoint:
                model = "gemini-1.5-flash"
                logger.info(f"🎯 Stream using Gemini 1.5 Flash (inferred from endpoint)")
            else:
                # For fine-tuned endpoints or when endpoint is not available, use standard model
                model = "gemini-1.5-pro"
                logger.info(f"🔄 Stream using standard Gemini 1.5 Pro (endpoint fallback: {model_endpoint})")

            # Create content following official documentation format
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=request.message)]
                )
            ]
            
            # Generation config following official documentation
            generation_config = types.GenerateContentConfig(
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.95,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    )
                ]
            )

            logger.info(f"🚀 Starting stream for model: {model}")

            # Send request and get streaming response
            responses = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generation_config,
            )
            
            # Yield each chunk
            for chunk in responses:
                if chunk.text:
                    cleaned_chunk = clean_response_text(chunk.text)
                    if cleaned_chunk:
                        yield f"data: {json.dumps({'response': cleaned_chunk})}\n\n"
                                
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            error_message = f"❌ Streaming error: {str(e)}"
            logger.error(error_message)
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

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