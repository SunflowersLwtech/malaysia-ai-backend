#!/usr/bin/env python3
"""
Malaysia AI Travel Guide - Working FastAPI Version
Based on successful minimal approach with complete RAG features
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Google AI
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
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

# Initialize FastAPI
app = FastAPI(
    title="Malaysia AI Travel Guide",
    description="Complete travel guide for Malaysia using AI",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MalaysiaAIBackend:
    """Working Malaysia AI Backend with complete features"""
    
    def __init__(self):
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.is_ready = False
        self.conversation_memory = {}
        self.knowledge_base = []
        self.model = None
        self.destinations = []
        
        # Initialize immediately
        self._initialize_sync()
    
    def _initialize_sync(self):
        """Synchronous initialization"""
        try:
            logger.info("🚀 Initializing Malaysia AI Backend...")
            
            # Setup AI
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                
                # Try fine-tuned model first, then fallback
                model_options = [
                    "TourismMalaysiaAI",
                    "gemini-1.5-flash",
                    "gemini-pro"
                ]
                
                for model_name in model_options:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        logger.info(f"✅ AI model connected: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to connect to {model_name}: {e}")
                        continue
            
            # Load knowledge and destinations
            self._load_knowledge_base()
            self._load_destinations()
            
            self.is_ready = True
            logger.info("🎉 Malaysia AI Backend ready!")
            
        except Exception as e:
            logger.warning(f"⚠️ Initialization warning: {e}, continuing with basic features")
            self.is_ready = True
    
    def _load_knowledge_base(self):
        """Load comprehensive Malaysia knowledge"""
        self.knowledge_base = [
            # Destinations
            {
                "content": "Kuala Lumpur: Malaysia's capital with iconic Petronas Twin Towers, vibrant street food, KL Tower, Batu Caves nearby, and modern shopping districts like Bukit Bintang",
                "category": "destination",
                "region": "central",
                "location": "Kuala Lumpur"
            },
            {
                "content": "Penang: UNESCO World Heritage site in George Town with colonial architecture, famous street art murals, incredible hawker food scene, and Clan Houses",
                "category": "destination", 
                "region": "north",
                "location": "Penang"
            },
            {
                "content": "Langkawi: Duty-free tropical island with beautiful beaches, Langkawi Cable Car, mangrove tours, and island hopping activities",
                "category": "destination",
                "region": "north",
                "location": "Langkawi"
            },
            {
                "content": "Malacca: Historic city with Portuguese, Dutch, and British influences, Jonker Street night market, Nyonya Peranakan culture, and river cruise",
                "category": "destination",
                "region": "south",
                "location": "Malacca"
            },
            {
                "content": "Cameron Highlands: Cool hill station with tea plantations, strawberry farms, hiking trails, and escape from tropical heat",
                "category": "destination",
                "region": "central",
                "location": "Cameron Highlands"
            },
            {
                "content": "Batu Caves: Hindu temple complex in limestone caves near KL, featuring 272 colorful steps and giant golden statue of Lord Murugan",
                "category": "destination",
                "region": "central",
                "location": "Selangor"
            },
            
            # Foods
            {
                "content": "Nasi Lemak: Malaysia's national dish with fragrant coconut rice, spicy sambal, crispy anchovies, roasted peanuts, cucumber, and boiled egg",
                "category": "food",
                "type": "main_dish",
                "origin": "Malay"
            },
            {
                "content": "Char Kway Teow: Stir-fried flat rice noodles with prawns, Chinese sausage, eggs, bean sprouts, and dark soy sauce, especially famous in Penang",
                "category": "food",
                "type": "noodles",
                "origin": "Chinese"
            },
            {
                "content": "Rendang: Rich and spicy slow-cooked meat curry with coconut milk and aromatic spices, tender and flavorful",
                "category": "food",
                "type": "curry",
                "origin": "Malay"
            },
            {
                "content": "Roti Canai: Flaky, crispy flatbread served with curry dipping sauce (dal or chicken curry), popular breakfast item",
                "category": "food",
                "type": "bread",
                "origin": "Indian"
            },
            {
                "content": "Laksa: Spicy noodle soup with variations like Asam Laksa (sour fish-based) and Curry Laksa (coconut curry-based)",
                "category": "food",
                "type": "noodles",
                "origin": "Peranakan"
            },
            {
                "content": "Cendol: Traditional dessert with green pandan jelly noodles, coconut milk, palm sugar syrup, and shaved ice",
                "category": "food",
                "type": "dessert",
                "origin": "Malay"
            },
            {
                "content": "Hainanese Chicken Rice: Tender poached chicken served with fragrant rice cooked in chicken stock, cucumber, and dipping sauces",
                "category": "food",
                "type": "main_dish",
                "origin": "Chinese"
            },
            
            # Culture & Tips
            {
                "content": "Malaysia is multicultural with Malay, Chinese, Indian, and indigenous communities. Be respectful of religious sites and dress modestly",
                "category": "culture",
                "type": "etiquette"
            },
            {
                "content": "Malaysian Ringgit (MYR) is the currency. Credit cards accepted in cities, but carry cash for street food and rural areas",
                "category": "travel_tips",
                "type": "money"
            },
            {
                "content": "Best time to visit is March-October (dry season). December-February is monsoon season on east coast",
                "category": "travel_tips",
                "type": "weather"
            },
            {
                "content": "Grab is the main ride-hailing app. Public transport includes LRT, MRT, and buses in KL. Rent a car for rural areas",
                "category": "travel_tips",
                "type": "transport"
            }
        ]
        logger.info(f"📚 Loaded {len(self.knowledge_base)} knowledge items")
    
    def _load_destinations(self):
        """Load destination coordinates for map integration"""
        self.destinations = [
            {
                "name": "Petronas Twin Towers",
                "location": "Kuala Lumpur",
                "latitude": 3.1579,
                "longitude": 101.7116,
                "category": "landmark"
            },
            {
                "name": "George Town",
                "location": "Penang", 
                "latitude": 5.4164,
                "longitude": 100.3327,
                "category": "heritage"
            },
            {
                "name": "Langkawi Cable Car",
                "location": "Langkawi",
                "latitude": 6.3833,
                "longitude": 99.7167,
                "category": "attraction"
            },
            {
                "name": "Batu Caves",
                "location": "Selangor",
                "latitude": 3.2379,
                "longitude": 101.6840,
                "category": "religious"
            },
            {
                "name": "Cameron Highlands",
                "location": "Pahang",
                "latitude": 4.4698,
                "longitude": 101.3831,
                "category": "nature"
            }
        ]
    
    def search_knowledge(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword-based knowledge search"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_base:
            content_lower = item['content'].lower()
            # Simple scoring based on keyword matches
            score = 0
            for word in query_lower.split():
                if word in content_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "content": item['content'],
                    "category": item['category'],
                    "score": score,
                    "metadata": {k: v for k, v in item.items() if k != 'content'}
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    async def generate_response(self, query: str, context: List[Dict], user_id: str = "anonymous") -> str:
        """Generate AI response with context"""
        try:
            if not self.model:
                return self._get_fallback_response(query, context)
            
            # Build context from search results
            context_text = ""
            if context:
                context_text = "\n".join([item['content'] for item in context])
            
            # Get conversation history
            memory = self.conversation_memory.get(user_id, [])
            memory_text = ""
            if memory:
                recent_memory = memory[-2:]  # Last 2 exchanges
                memory_text = "\n".join([f"User: {m['query']}\nAssistant: {m['response']}" for m in recent_memory])
            
            prompt = f"""You are a knowledgeable Malaysia travel guide. Help users discover Malaysia's destinations, food, culture, and travel tips.

Context from knowledge base:
{context_text}

Previous conversation:
{memory_text}

User question: {query}

Provide a helpful, friendly response about Malaysia tourism. Include specific recommendations when relevant."""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return self._get_fallback_response(query, context)
    
    def _get_fallback_response(self, query: str, context: List[Dict] = None) -> str:
        """Fallback response when AI is unavailable"""
        query_lower = query.lower()
        
        # Destination queries
        if any(word in query_lower for word in ['where', 'visit', 'destination', 'place', 'go']):
            return """Malaysia has amazing destinations! 🇲🇾
            
Popular places to visit:
• **Kuala Lumpur** - See the iconic Petronas Twin Towers and vibrant street food
• **Penang** - UNESCO heritage site with incredible food and street art  
• **Langkawi** - Beautiful tropical island with beaches and cable car
• **Cameron Highlands** - Cool hill station with tea plantations
• **Malacca** - Historic city with rich cultural heritage

What type of destination interests you most?"""
        
        # Food queries
        elif any(word in query_lower for word in ['food', 'eat', 'dish', 'cuisine', 'hungry']):
            return """Malaysian food is incredible! 🍜
            
Must-try dishes:
• **Nasi Lemak** - National dish with coconut rice and spicy sambal
• **Char Kway Teow** - Delicious stir-fried noodles (try it in Penang!)
• **Rendang** - Rich and spicy slow-cooked curry
• **Roti Canai** - Flaky bread with curry sauce, perfect for breakfast
• **Laksa** - Spicy noodle soup with many regional variations

Are you looking for specific types of food or restaurants?"""
        
        # General greeting
        else:
            return """Welcome to Malaysia! 🇲🇾 I'm here to help you discover this beautiful country.

I can help you with:
• **Destinations** - Where to go and what to see
• **Food** - Must-try dishes and where to find them  
• **Culture** - Local customs and traditions
• **Travel Tips** - Transportation, weather, and practical advice

What would you like to know about Malaysia?"""
    
    def update_memory(self, user_id: str, query: str, response: str):
        """Update conversation memory"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_memory[user_id]) > 10:
            self.conversation_memory[user_id] = self.conversation_memory[user_id][-10:]

# Initialize the backend
backend = MalaysiaAIBackend()

# FastAPI Endpoints
@app.get("/")
async def root():
    return {
        "service": "Malaysia AI Travel Guide",
        "status": "running",
        "version": "2.0.0",
        "description": "Complete travel guide for Malaysia using AI",
        "endpoints": {
            "health": "/health",
            "status": "/api/status", 
            "chat": "/api/chat",
            "knowledge": "/api/knowledge",
            "destinations": "/api/destinations",
            "feedback": "/api/feedback"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "is_ready": backend.is_ready,
        "has_ai": backend.model is not None
    }

@app.get("/api/status")
async def get_status():
    return {
        "is_ready": backend.is_ready,
        "ai_model_available": backend.model is not None,
        "knowledge_items": len(backend.knowledge_base),
        "destinations": len(backend.destinations),
        "features": ["chat", "knowledge_search", "destinations", "feedback"],
        "version": "2.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Search knowledge base
        context = backend.search_knowledge(request.message)
        
        # Generate AI response
        response_text = await backend.generate_response(
            request.message, context, request.user_id
        )
        
        # Update memory
        backend.update_memory(request.user_id, request.message, response_text)
        
        # Generate suggestions
        suggestions = [
            "Tell me about Malaysian food",
            "What are the best places to visit?",
            "How do I get around Malaysia?",
            "What's the weather like?"
        ]
        
        return ChatResponse(
            message=response_text,
            sources=context,
            suggestions=suggestions,
            location=request.location,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

@app.get("/api/knowledge")
async def get_knowledge():
    categories = {}
    for item in backend.knowledge_base:
        category = item.get('category', 'general')
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    return {
        "categories": categories,
        "total_items": len(backend.knowledge_base),
        "available_categories": list(categories.keys())
    }

@app.get("/api/destinations")
async def get_destinations():
    return {
        "destinations": backend.destinations,
        "total": len(backend.destinations)
    }

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    feedback_data = {
        "rating": request.rating,
        "feedback": request.feedback_text,
        "user_query": request.user_query,
        "ai_response": request.ai_response,
        "timestamp": datetime.now().isoformat()
    }
    
    # Log feedback (in production, save to database)
    logger.info(f"Feedback received: {feedback_data}")
    
    return {"message": "Thank you for your feedback!", "success": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1) 