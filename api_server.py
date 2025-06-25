import os
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import traceback

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Google Cloud and AI imports
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import aiplatform
import google.generativeai as genai

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Data processing
import pandas as pd
import numpy as np

# Google services
import gspread
from google.auth import default

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]

class StatusResponse(BaseModel):
    service: str
    status: str
    version: str
    model_status: str
    database_status: str

# Global variables for services
embedding_model = None
chroma_client = None
collection = None
vertex_model = None
genai_model = None
google_sheets_client = None
conversation_memory = {}

# Initialize FastAPI app
app = FastAPI(
    title="Malaysia AI Travel Guide",
    description="Complete RAG-powered travel guide for Malaysia",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_embedding_model():
    """Initialize sentence transformer model for embeddings"""
    global embedding_model
    try:
        logger.info("Initializing embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return False

def initialize_chromadb():
    """Initialize ChromaDB client and collection"""
    global chroma_client, collection
    try:
        logger.info("Initializing ChromaDB...")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path="./rag_database",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        collection_name = "malaysia_tourism"
        try:
            collection = chroma_client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            collection = chroma_client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
            
            # Load data if collection is new
            load_tourism_data()
        
        logger.info("ChromaDB initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return False

def load_tourism_data():
    """Load tourism data into ChromaDB"""
    try:
        logger.info("Loading tourism data...")
        
        # Sample tourism data
        tourism_data = [
            {
                "id": "dest_kl",
                "content": "Kuala Lumpur is Malaysia's capital and largest city. Key attractions include the Petronas Twin Towers, KL Tower, Batu Caves, Central Market, and Chinatown. The city offers excellent shopping at Bukit Bintang and diverse street food.",
                "category": "destination",
                "location": "Kuala Lumpur"
            },
            {
                "id": "dest_penang",
                "content": "Penang is known as the Pearl of the Orient. George Town is a UNESCO World Heritage site with colonial architecture, street art, and incredible food scene. Must-visit: Penang Hill, Kek Lok Si Temple, and local hawker centers.",
                "category": "destination",
                "location": "Penang"
            },
            {
                "id": "food_nasi_lemak",
                "content": "Nasi Lemak is Malaysia's national dish. Coconut rice served with sambal, fried anchovies, peanuts, cucumber, and boiled egg. Often accompanied by rendang, fried chicken, or curry. Available from street stalls to high-end restaurants.",
                "category": "food",
                "location": "Malaysia"
            },
            {
                "id": "food_char_kway_teow",
                "content": "Char Kway Teow is a popular stir-fried noodle dish from Penang. Flat rice noodles wok-fried with prawns, cockles, Chinese sausage, eggs, bean sprouts, and chives in dark soy sauce. Best enjoyed from street vendors.",
                "category": "food",
                "location": "Penang"
            },
            {
                "id": "culture_festivals",
                "content": "Malaysia celebrates diverse festivals: Chinese New Year, Hari Raya, Deepavali, Christmas. Each celebration features unique traditions, foods, and decorations. Many festivals are public holidays with special events nationwide.",
                "category": "culture",
                "location": "Malaysia"
            }
        ]
        
        # Generate embeddings and add to collection
        for data in tourism_data:
            if embedding_model:
                embedding = embedding_model.encode(data["content"]).tolist()
                collection.add(
                    embeddings=[embedding],
                    documents=[data["content"]],
                    metadatas=[{
                        "category": data["category"],
                        "location": data["location"],
                        "id": data["id"]
                    }],
                    ids=[data["id"]]
                )
        
        logger.info(f"Loaded {len(tourism_data)} tourism entries")
        return True
    except Exception as e:
        logger.error(f"Failed to load tourism data: {e}")
        return False

def initialize_vertex_ai():
    """Initialize Vertex AI model"""
    global vertex_model
    try:
        logger.info("Initializing Vertex AI...")
        
        # Check for service account key
        service_account_path = "bright-coyote-463315-q8-59797318b374.json"
        if os.path.exists(service_account_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        
        # Initialize Vertex AI
        project_id = "bright-coyote-463315-q8"
        location = "us-west1"
        
        vertexai.init(project=project_id, location=location)
        
        # Initialize your fine-tuned model endpoint (TourismMalaysiaAI)
        endpoint_id = "4352232060597829632"
        
        # Create endpoint connection for your fine-tuned model
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        )
        
        # Use the endpoint for predictions
        vertex_model = endpoint
        
        logger.info("Vertex AI fine-tuned model (TourismMalaysiaAI) initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        return False

def initialize_genai_fallback():
    """Initialize Gemini API as fallback"""
    global genai_model
    try:
        logger.info("Initializing Gemini fallback...")
        
        # Configure Gemini API (you'll need to set your API key)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini fallback initialized successfully")
            return True
        else:
            logger.warning("GEMINI_API_KEY not found, fallback unavailable")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize Gemini fallback: {e}")
        return False

def initialize_google_sheets():
    """Initialize Google Sheets client for feedback"""
    global google_sheets_client
    try:
        logger.info("Initializing Google Sheets...")
        
        # Use service account for authentication
        google_sheets_client = gspread.service_account(filename="bright-coyote-463315-q8-59797318b374.json")
        
        logger.info("Google Sheets initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Sheets: {e}")
        return False

def search_knowledge_base(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search the knowledge base using vector similarity"""
    try:
        if not collection or not embedding_model:
            logger.warning("ChromaDB or embedding model not available")
            return []
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        # Format results
        sources = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                
                sources.append({
                    "content": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
        
        return sources
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return []

def is_malaysia_travel_related(user_message: str) -> bool:
    """Check if the user message is related to Malaysia travel"""
    try:
        # Convert to lowercase for easier matching
        message_lower = user_message.lower()
        
        # Malaysia-related keywords
        malaysia_keywords = [
            'malaysia', 'malaysian', 'kuala lumpur', 'kl', 'penang', 'johor', 'sabah', 'sarawak',
            'langkawi', 'malacca', 'melaka', 'ipoh', 'cameron highlands', 'genting', 'putrajaya',
            'cyberjaya', 'shah alam', 'petaling jaya', 'klang', 'seremban', 'nilai', 'kedah',
            'kelantan', 'terengganu', 'pahang', 'perak', 'negeri sembilan', 'selangor',
            'kuching', 'kota kinabalu', 'sandakan', 'miri', 'sibu', 'bintulu', 'tawau'
        ]
        
        # Travel-related keywords
        travel_keywords = [
            'travel', 'trip', 'visit', 'tour', 'tourism', 'vacation', 'holiday', 'destination',
            'attraction', 'place', 'hotel', 'accommodation', 'restaurant', 'food', 'eat',
            'culture', 'festival', 'temple', 'mosque', 'beach', 'island', 'mountain', 'park',
            'shopping', 'mall', 'market', 'transportation', 'flight', 'bus', 'train', 'taxi',
            'grab', 'weather', 'climate', 'currency', 'ringgit', 'language', 'malay', 'english',
            'chinese', 'tamil', 'heritage', 'history', 'museum', 'gallery', 'activity',
            'things to do', 'where to go', 'how to get', 'best time', 'recommendation',
            'suggest', 'advice', 'guide', 'itinerary', 'budget', 'cost', 'price'
        ]
        
        # Food-related keywords (Malaysian cuisine)
        food_keywords = [
            'nasi lemak', 'char kway teow', 'rendang', 'satay', 'laksa', 'roti canai',
            'bak kut teh', 'cendol', 'durian', 'teh tarik', 'mamak', 'hawker', 'kopitiam',
            'dim sum', 'curry', 'sambal', 'coconut', 'rice', 'noodles', 'seafood', 'halal'
        ]
        
        # Check if message contains Malaysia-related terms
        has_malaysia_keyword = any(keyword in message_lower for keyword in malaysia_keywords)
        
        # Check if message contains travel-related terms
        has_travel_keyword = any(keyword in message_lower for keyword in travel_keywords)
        
        # Check if message contains food-related terms
        has_food_keyword = any(keyword in message_lower for keyword in food_keywords)
        
        # Message is relevant if it has Malaysia keywords OR (travel/food keywords that could be about Malaysia)
        if has_malaysia_keyword:
            return True
        elif has_travel_keyword or has_food_keyword:
            # Additional context check - if no Malaysia keyword but has travel/food terms,
            # we'll be more lenient and let the AI decide with context
            return True
        
        # Common question patterns that might be Malaysia-related even without keywords
        question_patterns = [
            'where', 'what', 'how', 'when', 'which', 'recommend', 'suggest', 'best',
            'good', 'popular', 'famous', 'must visit', 'should i', 'can i', 'is it'
        ]
        
        has_question_pattern = any(pattern in message_lower for pattern in question_patterns)
        
        # If it's a question, we'll allow it but the AI will redirect if not Malaysia-related
        if has_question_pattern:
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error checking Malaysia travel relevance: {e}")
        # If there's an error, be permissive and let the AI handle it
        return True

def generate_ai_response(user_message: str, context: str, conversation_history: List[str]) -> str:
    """Generate AI response using available models, restricted to Malaysia travel topics"""
    try:
        # Check if the question is related to Malaysia travel
        if not is_malaysia_travel_related(user_message):
            return """I'm TourismMalaysiaAI, specifically designed to help with Malaysia travel questions. 

I can assist you with:
🏛️ Malaysian destinations and attractions
🍜 Local food and dining recommendations  
🎭 Cultural experiences and festivals
🏨 Accommodation and transportation
🗺️ Travel itineraries and tips
💰 Budget and cost information

Please ask me anything about traveling in Malaysia! For example:
- "What are the best places to visit in Kuala Lumpur?"
- "Where can I find the best nasi lemak?"
- "What's the weather like in Penang?"
- "How do I get from KLIA to the city center?"

How can I help you explore Malaysia? 🇲🇾"""

        # Prepare enhanced prompt for TourismMalaysiaAI with strict focus
        system_prompt = """You are TourismMalaysiaAI, a specialized AI assistant exclusively for Malaysia travel and tourism. 

STRICT GUIDELINES:
- ONLY answer questions related to Malaysia travel, tourism, destinations, food, culture, and experiences
- If asked about other countries, politely redirect to Malaysia
- If asked non-travel questions, redirect to Malaysia travel topics
- Use the provided context from the knowledge base when available
- Be enthusiastic, helpful, and knowledgeable about Malaysia
- Include practical travel tips and local insights
- Mention specific places, foods, and experiences in Malaysia

ALWAYS stay focused on Malaysia travel topics only."""
        
        # Format conversation history
        history_text = "\n".join(conversation_history[-6:]) if conversation_history else ""
        
        prompt = f"""{system_prompt}

Context from Malaysia travel knowledge base:
{context}

Recent conversation:
{history_text}

User Question: {user_message}

TourismMalaysiaAI Response (Malaysia travel focused):"""

        # Try your fine-tuned TourismMalaysiaAI model first
        if vertex_model:
            try:
                # Prepare input for your fine-tuned model
                instances = [{"content": prompt}]
                
                # Call your fine-tuned model endpoint
                response = vertex_model.predict(instances=instances)
                
                # Extract the response text
                if response.predictions and len(response.predictions) > 0:
                    prediction = response.predictions[0]
                    if isinstance(prediction, dict) and 'content' in prediction:
                        ai_response = prediction['content']
                    elif isinstance(prediction, str):
                        ai_response = prediction
                    else:
                        ai_response = str(prediction)
                    
                    # Add Malaysia focus reminder if response seems too general
                    if ai_response and len(ai_response) > 50:
                        return ai_response
                    
                logger.warning("No valid prediction from fine-tuned model")
                
            except Exception as e:
                logger.warning(f"Fine-tuned TourismMalaysiaAI model failed: {e}")
        
        # Try Gemini fallback with Malaysia focus
        if genai_model:
            try:
                response = genai_model.generate_content(prompt)
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini fallback failed: {e}")
        
        # Fallback response - still Malaysia focused
        return """I'm TourismMalaysiaAI, your dedicated Malaysia travel guide! 🇲🇾

Even though I'm having technical difficulties right now, I'd love to help you discover Malaysia's amazing destinations:

🏙️ **Cities**: Kuala Lumpur (Petronas Towers, Bukit Bintang), Penang (George Town UNESCO site), Malacca (historic city)
🏝️ **Islands**: Langkawi (beaches & cable car), Tioman (diving paradise), Redang (crystal waters)
🍜 **Food**: Nasi lemak, char kway teow, rendang, laksa, satay, roti canai
🎭 **Culture**: Diverse festivals (Chinese New Year, Hari Raya, Deepavali), temples, mosques
🌿 **Nature**: Cameron Highlands (tea plantations), Taman Negara (rainforest), Mount Kinabalu

What specific aspect of Malaysia travel interests you most? I'll do my best to help despite the technical issues!"""
        
    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        return "I apologize for the technical difficulties. As TourismMalaysiaAI, I'm here to help with your Malaysia travel questions. Please try asking about Malaysian destinations, food, or travel tips, and I'll do my best to assist you! 🇲🇾"

# Initialize all services at startup
def initialize_services():
    """Initialize all services"""
    logger.info("Starting service initialization...")
    
    # Initialize components
    embedding_success = initialize_embedding_model()
    chromadb_success = initialize_chromadb()
    vertex_success = initialize_vertex_ai()
    genai_success = initialize_genai_fallback()
    sheets_success = initialize_google_sheets()
    
    logger.info(f"Service initialization complete:")
    logger.info(f"  - Embedding model: {'✓' if embedding_success else '✗'}")
    logger.info(f"  - ChromaDB: {'✓' if chromadb_success else '✗'}")
    logger.info(f"  - Vertex AI: {'✓' if vertex_success else '✗'}")
    logger.info(f"  - Gemini fallback: {'✓' if genai_success else '✗'}")
    logger.info(f"  - Google Sheets: {'✓' if sheets_success else '✗'}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services when FastAPI starts"""
    initialize_services()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Malaysia AI Travel Guide",
        "status": "operational",
        "version": "2.0.0",
        "description": "Complete RAG-powered travel guide for Malaysia"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "embedding_model": "healthy" if embedding_model else "unavailable",
        "chromadb": "healthy" if collection else "unavailable",
        "vertex_ai": "healthy" if vertex_model else "unavailable",
        "genai_fallback": "healthy" if genai_model else "unavailable",
        "google_sheets": "healthy" if google_sheets_client else "unavailable"
    }
    
    overall_status = "healthy" if any(v == "healthy" for v in [components["vertex_ai"], components["genai_fallback"]]) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get service status"""
    return StatusResponse(
        service="Malaysia AI Travel Guide",
        status="operational",
        version="2.0.0",
        model_status="available" if vertex_model or genai_model else "unavailable",
        database_status="available" if collection else "unavailable"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        user_message = request.message.strip()
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
        
        # Initialize conversation history if needed
        if conversation_id not in conversation_memory:
            conversation_memory[conversation_id] = []
        
        # Search knowledge base
        sources = search_knowledge_base(user_message)
        
        # Prepare context
        context = "\n\n".join([source["content"] for source in sources[:3]])
        
        # Generate AI response
        ai_response = generate_ai_response(
            user_message=user_message,
            context=context,
            conversation_history=conversation_memory[conversation_id]
        )
        
        # Update conversation memory
        conversation_memory[conversation_id].append(f"User: {user_message}")
        conversation_memory[conversation_id].append(f"AI: {ai_response}")
        
        # Keep only last 10 exchanges
        if len(conversation_memory[conversation_id]) > 20:
            conversation_memory[conversation_id] = conversation_memory[conversation_id][-20:]
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            sources=sources[:3]
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )

@app.get("/api/map")
async def get_map():
    """Get interactive map HTML"""
    try:
        map_html_path = "../malaysia-ai-map/map.html"
        if os.path.exists(map_html_path):
            with open(map_html_path, 'r', encoding='utf-8') as f:
                map_html = f.read()
            return {"map_html": map_html}
        else:
            return {"map_html": "<p>Map not available</p>"}
    except Exception as e:
        logger.error(f"Map endpoint error: {e}")
        return {"map_html": "<p>Error loading map</p>"}

# Additional utility endpoints
@app.get("/api/destinations")
async def get_destinations():
    """Get list of popular destinations"""
    destinations = [
        {"name": "Kuala Lumpur", "description": "Malaysia's vibrant capital city"},
        {"name": "Penang", "description": "UNESCO World Heritage site with amazing food"},
        {"name": "Malacca", "description": "Historic city with colonial architecture"},
        {"name": "Langkawi", "description": "Tropical island paradise"},
        {"name": "Cameron Highlands", "description": "Cool hill station with tea plantations"}
    ]
    return {"destinations": destinations}

@app.get("/api/food")
async def get_popular_foods():
    """Get list of popular Malaysian foods"""
    foods = [
        {"name": "Nasi Lemak", "description": "Malaysia's national dish with coconut rice"},
        {"name": "Char Kway Teow", "description": "Stir-fried flat rice noodles"},
        {"name": "Rendang", "description": "Slow-cooked meat in coconut curry"},
        {"name": "Laksa", "description": "Spicy noodle soup with various regional variations"},
        {"name": "Satay", "description": "Grilled meat skewers with peanut sauce"}
    ]
    return {"foods": foods}

@app.post("/api/check-relevance")
async def check_relevance(request: ChatRequest):
    """Test endpoint to check if a question is Malaysia travel related"""
    try:
        user_message = request.message.strip()
        is_relevant = is_malaysia_travel_related(user_message)
        
        return {
            "message": user_message,
            "is_malaysia_travel_related": is_relevant,
            "explanation": "This question is about Malaysia travel" if is_relevant else "This question is not about Malaysia travel"
        }
    except Exception as e:
        logger.error(f"Relevance check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking question relevance"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 