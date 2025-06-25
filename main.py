#!/usr/bin/env python3
"""
Malaysia AI Travel Guide - Enhanced FastAPI Backend
Modern async API with full RAG capabilities and Vertex AI integration
Optimized for Google Cloud Run deployment
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# FastAPI and async
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Google Cloud and AI
from google.cloud import aiplatform
import google.generativeai as genai

# Vector Database and ML
import chromadb
from sentence_transformers import SentenceTransformer

# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    location: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    location: Optional[str] = None
    timestamp: str
    success: bool

class FeedbackRequest(BaseModel):
    rating: int
    feedback_text: str
    user_query: Optional[str] = None
    ai_response: Optional[str] = None
    location: Optional[str] = None

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    radius: Optional[float] = 10.0  # km

# Malaysia destinations with coordinates for map integration
MALAYSIA_DESTINATIONS = [
    {
        "name": "Petronas Twin Towers",
        "location": "Kuala Lumpur",
        "latitude": 3.1579,
        "longitude": 101.7116,
        "category": "landmark",
        "description": "Iconic twin towers and symbol of Malaysia"
    },
    {
        "name": "George Town",
        "location": "Penang",
        "latitude": 5.4164,
        "longitude": 100.3327,
        "category": "heritage",
        "description": "UNESCO World Heritage site with rich history"
    },
    {
        "name": "Langkawi Cable Car",
        "location": "Langkawi",
        "latitude": 6.3833,
        "longitude": 99.7167,
        "category": "attraction",
        "description": "Stunning cable car ride with panoramic views"
    },
    {
        "name": "Batu Caves",
        "location": "Selangor",
        "latitude": 3.2379,
        "longitude": 101.6840,
        "category": "religious",
        "description": "Hindu temple complex in limestone caves"
    },
    {
        "name": "Cameron Highlands",
        "location": "Pahang",
        "latitude": 4.4698,
        "longitude": 101.3831,
        "category": "nature",
        "description": "Cool hill station with tea plantations"
    }
]

class MalaysiaAIBackend:
    """Enhanced FastAPI backend with full RAG and map capabilities"""
    
    def __init__(self):
        # Configuration
        self.project_id = "bright-coyote-463315-q8"
        self.location = "us-west1"
        self.endpoint_id = "4352232060597829632"
        self.sheet_id = "1tE80wYY5yqEW0uR553RcnM7IkKqhQF1INtmgf54aRp4"
        self.maps_api_key = "AIzaSyCS__n781EsrrX80XVVcTf2biRdMaftsK4"
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.vertex_ai_endpoint = None
        self.gemini_model = None
        self.fallback_model = None
        self.fine_tuned_model_name = None
        self.google_sheet = None
        self.is_ready = False
        self.conversation_memory = {}
        
        # Initialize on startup
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Async initialization of all components"""
        try:
            logger.info("🚀 Initializing Malaysia AI Backend (FastAPI + Cloud Run)...")
            
            # 1. Initialize embedding model
            await self._setup_embeddings()
            
            # 2. Initialize ChromaDB
            await self._setup_chromadb()
            
            # 3. Initialize AI services
            await self._setup_ai_services()
            
            # 4. Initialize Google Sheets
            await self._setup_google_sheets()
            
            # 5. Load and process data
            await self._load_and_process_data()
            
            self.is_ready = True
            logger.info("🎉 Malaysia AI Backend initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {str(e)}")
            # Continue with degraded functionality
            self.is_ready = False
    
    async def _setup_embeddings(self):
        """Setup sentence transformer model"""
        try:
            logger.info("📥 Loading embedding model...")
            # Use lighter model for Cloud Run
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Embedding model loaded!")
        except Exception as e:
            logger.error(f"❌ Embedding model error: {e}")
            self.embedding_model = None
    
    async def _setup_chromadb(self):
        """Setup ChromaDB for vector storage"""
        try:
            logger.info("🗄️ Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path="./vector_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="malaysia_travel_rag",
                metadata={"description": "Malaysia travel and food RAG database"}
            )
            logger.info("✅ ChromaDB initialized!")
        except Exception as e:
            logger.error(f"❌ ChromaDB error: {e}")
            self.chroma_client = None
    
    async def _setup_ai_services(self):
        """Setup fine-tuned Gemini model via Vertex AI"""
        try:
            # Setup Vertex AI for fine-tuned Gemini model
            logger.info("🔐 Setting up Vertex AI for fine-tuned Gemini...")
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Your fine-tuned Gemini 2.5 Flash model name
            self.fine_tuned_model_name = f"projects/{self.project_id}/locations/{self.location}/models/TourismMalaysiaAI"
            
            # Setup Gemini API with your fine-tuned model
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                # Use your fine-tuned model
                self.gemini_model = genai.GenerativeModel(self.fine_tuned_model_name)
                logger.info("✅ Fine-tuned Gemini 2.5 Flash model connected!")
                
                # Fallback to standard Gemini if fine-tuned fails
                self.fallback_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Fallback Gemini model configured!")
            
        except Exception as e:
            logger.error(f"❌ AI services error: {e}")
            # Try standard Vertex AI endpoint as ultimate fallback
            try:
                endpoint_name = f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
                self.vertex_ai_endpoint = aiplatform.Endpoint(endpoint_name)
                logger.info("✅ Vertex AI endpoint fallback connected!")
            except:
                self.vertex_ai_endpoint = None
    
    async def _setup_google_sheets(self):
        """Setup Google Sheets for feedback"""
        try:
            logger.info("📊 Setting up Google Sheets...")
            scopes = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Use service account credentials
            if os.path.exists('bright-coyote-463315-q8-59797318b374.json'):
                creds = Credentials.from_service_account_file(
                    'bright-coyote-463315-q8-59797318b374.json',
                    scopes=scopes
                )
                client = gspread.authorize(creds)
                self.google_sheet = client.open_by_key(self.sheet_id).sheet1
                logger.info("✅ Google Sheets connected!")
            
        except Exception as e:
            logger.error(f"❌ Google Sheets error: {e}")
            self.google_sheet = None
    
    async def _load_and_process_data(self):
        """Load and process data files into vector database"""
        try:
            logger.info("📂 Loading and processing data files...")
            
            data_files = [
                "RestaurantOriginalCSV.csv",
                "vertex_ai_training_data.jsonl",
                "Curation Queue (3).xlsx"
            ]
            
            all_documents = []
            all_metadatas = []
            all_ids = []
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    logger.info(f"📄 Processing {file_path}...")
                    documents, metadatas, ids = await self._process_file(file_path)
                    all_documents.extend(documents)
                    all_metadatas.extend(metadatas)
                    all_ids.extend(ids)
            
            # Generate embeddings and store in ChromaDB
            if self.embedding_model and self.collection and all_documents:
                logger.info(f"🔮 Generating embeddings for {len(all_documents)} documents...")
                embeddings = self.embedding_model.encode(all_documents)
                
                # Store in ChromaDB
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=all_documents,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                
                logger.info(f"✅ Stored {len(all_documents)} documents in vector database!")
            
        except Exception as e:
            logger.error(f"❌ Data processing error: {e}")
    
    async def _process_file(self, file_path: str):
        """Process individual data file"""
        documents = []
        metadatas = []
        ids = []
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                for idx, row in df.iterrows():
                    content = self._create_document_content(row)
                    documents.append(content)
                    metadatas.append({
                        "source_file": file_path,
                        "row_id": idx,
                        "type": "restaurant" if "restaurant" in file_path.lower() else "general"
                    })
                    ids.append(f"{file_path}_{idx}")
            
            elif file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                content = json.dumps(data, ensure_ascii=False)
                                documents.append(content)
                                metadatas.append({
                                    "source_file": file_path,
                                    "row_id": idx,
                                    "type": "training_data"
                                })
                                ids.append(f"{file_path}_{idx}")
                            except:
                                continue
            
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                for idx, row in df.iterrows():
                    content = self._create_document_content(row)
                    documents.append(content)
                    metadatas.append({
                        "source_file": file_path,
                        "row_id": idx,
                        "type": "curation"
                    })
                    ids.append(f"{file_path}_{idx}")
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
        
        return documents, metadatas, ids
    
    def _create_document_content(self, row):
        """Create searchable content from data row"""
        content_parts = []
        
        # Common text columns
        text_columns = ['name', 'title', 'description', 'content', 'location', 'address', 
                       'cuisine', 'category', 'type', 'price', 'rating', 'review']
        
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        # If no specific columns found, use all string columns
        if not content_parts:
            for col, val in row.items():
                if isinstance(val, str) and len(str(val).strip()) > 3:
                    content_parts.append(f"{col}: {val}")
        
        return " | ".join(content_parts[:10])  # Limit to prevent too long documents
    
    async def search_vector_database(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search in vector database"""
        try:
            if not self.embedding_model or not self.collection:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max_results
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - results['distances'][0][i],
                        "source_file": results['metadatas'][0][i].get('source_file', 'unknown')
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []
    
    async def generate_ai_response(self, query: str, context: List[Dict], 
                                 location: Optional[str] = None, user_id: str = "anonymous") -> str:
        """Generate AI response using Vertex AI or Gemini fallback"""
        try:
            # Build context from RAG results
            context_text = ""
            if context:
                context_text = "\n\nRelevant information from knowledge base:\n"
                for i, item in enumerate(context[:3], 1):
                    context_text += f"{i}. {item['content'][:200]}...\n"
            
            # Add conversation history
            history = self.conversation_memory.get(user_id, [])
            history_text = ""
            if history:
                recent_history = history[-2:]
                history_text = "\n\nRecent conversation:\n"
                for h in recent_history:
                    history_text += f"User: {h['query']}\nAssistant: {h['response'][:150]}...\n"
            
            location_text = f"\nUser's location: {location}" if location else ""
            
            # Create comprehensive prompt
            prompt = f"""You are a knowledgeable and enthusiastic Malaysia travel guide AI. 
Help users discover the best food, destinations, and experiences in Malaysia with accurate, engaging recommendations.

User question: {query}{location_text}{history_text}{context_text}

Provide a helpful, accurate response about Malaysia in 150-200 words. Include specific recommendations with practical details like locations, pricing ranges, and unique features when possible.
Be conversational and enthusiastic about Malaysia's diversity."""

            # Try your fine-tuned Gemini 2.5 Flash model first
            if self.gemini_model:
                try:
                    response = self.gemini_model.generate_content(prompt)
                    if response and response.text:
                        logger.info("✅ Response from fine-tuned Gemini 2.5 Flash")
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"Fine-tuned Gemini error, trying fallback: {e}")
            
            # Fallback to standard Gemini model
            if hasattr(self, 'fallback_model') and self.fallback_model:
                try:
                    response = self.fallback_model.generate_content(prompt)
                    if response and response.text:
                        logger.info("✅ Response from fallback Gemini model")
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"Fallback Gemini error, trying Vertex AI: {e}")
            
            # Ultimate fallback to Vertex AI endpoint
            if hasattr(self, 'vertex_ai_endpoint') and self.vertex_ai_endpoint:
                try:
                    instances = [{"content": prompt}]
                    response = self.vertex_ai_endpoint.predict(instances=instances)
                    
                    if response and response.predictions:
                        logger.info("✅ Response from Vertex AI endpoint")
                        return response.predictions[0].get('content', 
                            "I'd love to help you explore Malaysia! Could you tell me more about what you're looking for?")
                except Exception as e:
                    logger.warning(f"Vertex AI endpoint error: {e}")
            
            # Ultimate fallback
            return "I'm here to help you discover amazing Malaysia! Please ask me about food, destinations, or travel experiences you're interested in."
            
        except Exception as e:
            logger.error(f"❌ AI response error: {e}")
            return "I'm experiencing technical difficulties, but I'd love to help you explore Malaysia! Please try your question again."
    
    async def get_nearby_destinations(self, latitude: float, longitude: float, radius: float = 10.0) -> List[Dict[str, Any]]:
        """Get destinations near specified coordinates"""
        nearby = []
        
        for dest in MALAYSIA_DESTINATIONS:
            # Simple distance calculation (for demo purposes)
            lat_diff = abs(dest['latitude'] - latitude)
            lon_diff = abs(dest['longitude'] - longitude)
            distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111  # Rough km conversion
            
            if distance <= radius:
                dest_copy = dest.copy()
                dest_copy['distance_km'] = round(distance, 2)
                nearby.append(dest_copy)
        
        return sorted(nearby, key=lambda x: x['distance_km'])
    
    def update_conversation_memory(self, user_id: str, query: str, response: str):
        """Update conversation memory"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "query": query[:200],
            "response": response[:300],
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep last 5 exchanges
        self.conversation_memory[user_id] = self.conversation_memory[user_id][-5:]
    
    async def save_feedback(self, feedback_data: Dict[str, Any]):
        """Save feedback to Google Sheets"""
        try:
            if not self.google_sheet:
                return False
            
            row_data = [
                datetime.now().isoformat(),
                feedback_data.get('user_query', ''),
                feedback_data.get('ai_response', ''),
                feedback_data.get('rating', ''),
                feedback_data.get('feedback_text', ''),
                feedback_data.get('location', '')
            ]
            
            self.google_sheet.append_row(row_data)
            logger.info("✅ Feedback saved to Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feedback save error: {e}")
            return False

# Initialize backend
backend = MalaysiaAIBackend()

# Create FastAPI app
app = FastAPI(
    title="Malaysia AI Travel Guide",
    description="Enhanced AI-powered travel guide for Malaysia with RAG and map integration",
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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Malaysia AI Travel Guide",
        "version": "2.0.0",
        "description": "Enhanced AI-powered travel guide with RAG and map integration",
        "status": "running",
        "endpoints": [
            "/health",
            "/api/status", 
            "/api/chat",
            "/api/knowledge",
            "/api/map/destinations",
            "/api/map/nearby",
            "/api/feedback"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "backend_ready": backend.is_ready,
        "components": {
            "embeddings": backend.embedding_model is not None,
            "vector_db": backend.collection is not None,
            "vertex_ai": backend.vertex_ai_endpoint is not None,
            "gemini": backend.gemini_model is not None,
            "sheets": backend.google_sheet is not None
        }
    }

@app.get("/api/status")
async def get_status():
    """Get detailed system status"""
    return {
        "is_ready": backend.is_ready,
        "has_ai": backend.vertex_ai_endpoint is not None or backend.gemini_model is not None,
        "has_rag": backend.collection is not None and backend.embedding_model is not None,
        "has_sheets": backend.google_sheet is not None,
        "version": "2.0.0",
        "features": ["rag_search", "ai_chat", "map_integration", "feedback_system"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with RAG"""
    try:
        # Perform RAG search
        search_results = await backend.search_vector_database(request.message, max_results=5)
        
        # Generate AI response
        ai_response = await backend.generate_ai_response(
            query=request.message,
            context=search_results,
            location=request.location,
            user_id=request.user_id
        )
        
        # Update conversation memory
        backend.update_conversation_memory(request.user_id, request.message, ai_response)
        
        # Generate suggestions based on query
        suggestions = []
        query_lower = request.message.lower()
        if 'food' in query_lower or 'eat' in query_lower:
            suggestions = [
                "What's the best street food in Penang?",
                "Where to find authentic Nasi Lemak?",
                "Best hawker centers in KL"
            ]
        elif any(word in query_lower for word in ['place', 'visit', 'destination']):
            suggestions = [
                "Things to do in Cameron Highlands",
                "Best beaches in Langkawi",
                "Cultural sites in George Town"
            ]
        
        return ChatResponse(
            message=ai_response,
            sources=search_results[:3],
            suggestions=suggestions,
            location=request.location,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/knowledge")
async def get_knowledge():
    """Get knowledge base information"""
    try:
        if not backend.collection:
            return {"message": "Knowledge base not available"}
        
        # Get collection info
        count = backend.collection.count()
        
        return {
            "total_documents": count,
            "categories": ["destinations", "food", "culture", "activities"],
            "data_sources": ["RestaurantOriginalCSV.csv", "vertex_ai_training_data.jsonl", "Curation Queue (3).xlsx"],
            "search_type": "semantic_vector_search",
            "model": "paraphrase-multilingual-MiniLM-L12-v2"
        }
        
    except Exception as e:
        logger.error(f"❌ Knowledge endpoint error: {e}")
        return {"error": str(e)}

@app.get("/api/map/destinations")
async def get_all_destinations():
    """Get all Malaysia destinations with coordinates"""
    return {
        "destinations": MALAYSIA_DESTINATIONS,
        "total_count": len(MALAYSIA_DESTINATIONS),
        "maps_api_key": backend.maps_api_key  # For frontend map integration
    }

@app.post("/api/map/nearby")
async def get_nearby_destinations(request: LocationRequest):
    """Get destinations near specified coordinates"""
    try:
        nearby = await backend.get_nearby_destinations(
            request.latitude, 
            request.longitude, 
            request.radius
        )
        
        return {
            "query_location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "radius_km": request.radius
            },
            "nearby_destinations": nearby,
            "count": len(nearby)
        }
        
    except Exception as e:
        logger.error(f"❌ Nearby destinations error: {e}")
        raise HTTPException(status_code=500, detail="Error finding nearby destinations")

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit user feedback"""
    try:
        feedback_data = {
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "user_query": request.user_query,
            "ai_response": request.ai_response,
            "location": request.location,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save feedback in background
        background_tasks.add_task(backend.save_feedback, feedback_data)
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Error saving feedback")

# For Cloud Run deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 