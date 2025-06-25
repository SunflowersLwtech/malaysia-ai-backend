#!/usr/bin/env python3
"""
Malaysia AI Travel Guide - Ultra-Minimal Backend API
Zero-dependency approach for maximum compatibility with Render.com free tier
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# Google AI
import google.generativeai as genai

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalaysiaAIGuideMinimal:
    """
    Ultra-minimal Malaysia AI Travel Guide - Zero external dependencies
    Optimized for Render.com free tier deployment
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.is_ready = False
        self.conversation_memory = {}
        self.knowledge_base = []
        
        logger.info("🚀 Initializing Malaysia AI Travel Guide (Minimal)...")
        self._initialize_components()
        self._setup_routes()
    
    def _initialize_components(self):
        """Initialize AI components with zero dependencies"""
        try:
            # 1. Setup Gemini AI
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini AI configured successfully!")
            else:
                logger.warning("⚠️ No Gemini API key found - using fallback responses")
                self.model = None
            
            # 2. Load basic knowledge (hardcoded for reliability)
            self._load_basic_knowledge()
            
            self.is_ready = True
            logger.info("🎉 Malaysia AI Guide Minimal initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {str(e)}")
            # Continue even if there are errors - fallback mode
            self.is_ready = True
    
    def _load_basic_knowledge(self):
        """Load basic hardcoded knowledge about Malaysia"""
        self.knowledge_base = [
            {
                "content": "Kuala Lumpur: Capital city with Petronas Twin Towers, vibrant street food scene, and modern shopping malls",
                "category": "destination",
                "region": "central"
            },
            {
                "content": "Penang: UNESCO World Heritage site known for street art, hawker food, and George Town historical architecture",
                "category": "destination", 
                "region": "north"
            },
            {
                "content": "Langkawi: Tropical island paradise with duty-free shopping, cable car rides, and beautiful beaches",
                "category": "destination",
                "region": "north"
            },
            {
                "content": "Malacca: Historical city with Portuguese, Dutch, and British colonial influences and Nyonya cuisine",
                "category": "destination",
                "region": "south"
            },
            {
                "content": "Nasi Lemak: Malaysia's national dish with coconut rice, sambal, anchovies, peanuts, and boiled egg",
                "category": "food",
                "type": "main_dish"
            },
            {
                "content": "Char Kway Teow: Stir-fried rice noodles with prawns, Chinese lap cheong, eggs, and bean sprouts",
                "category": "food",
                "type": "noodles"
            },
            {
                "content": "Rendang: Slow-cooked spicy meat curry, originally from Indonesia but popular in Malaysia",
                "category": "food",
                "type": "curry"
            },
            {
                "content": "Roti Canai: Flaky flatbread served with curry dipping sauce, popular breakfast item",
                "category": "food",
                "type": "bread"
            },
            {
                "content": "Cendol: Traditional dessert with pandan-flavored rice flour jelly, coconut milk, and palm sugar",
                "category": "food",
                "type": "dessert"
            },
            {
                "content": "Cameron Highlands: Cool hill station known for tea plantations, strawberry farms, and escape from tropical heat",
                "category": "destination",
                "region": "central"
            }
        ]
        logger.info(f"📚 Loaded {len(self.knowledge_base)} knowledge items")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            return jsonify({
                "service": "Malaysia AI Travel Guide",
                "status": "running",
                "version": "minimal-1.0",
                "description": "Discover the best of Malaysia - food, destinations, and experiences!",
                "endpoints": [
                    "/health", 
                    "/api/status", 
                    "/api/chat",
                    "/api/knowledge",
                    "/api/feedback"
                ]
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "memory_usage": "minimal",
                "uptime": "running"
            })
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                "is_ready": self.is_ready,
                "has_ai": self.model is not None,
                "knowledge_items": len(self.knowledge_base),
                "version": "minimal",
                "features": ["chat", "knowledge_search", "feedback"],
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/knowledge')
        def get_knowledge():
            """Get all available knowledge categories"""
            categories = {}
            for item in self.knowledge_base:
                category = item.get('category', 'general')
                if category not in categories:
                    categories[category] = []
                categories[category].append({
                    "content": item['content'],
                    "region": item.get('region'),
                    "type": item.get('type')
                })
            
            return jsonify({
                "categories": categories,
                "total_items": len(self.knowledge_base),
                "available_categories": list(categories.keys())
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                query = data.get('message', '').strip()
                user_id = data.get('user_id', 'anonymous')
                location = data.get('location')
                
                if not query:
                    return jsonify({"error": "Message is required"}), 400
                
                # Process query
                response = self._process_query(query, user_id, location)
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"❌ Chat error: {str(e)}")
                return jsonify({
                    "error": "Internal server error",
                    "message": "I apologize, but I'm experiencing technical difficulties. Let me help you with basic Malaysia travel information!",
                    "suggested_topics": ["food recommendations", "places to visit", "Kuala Lumpur attractions"],
                    "success": False
                }), 500
        
        @self.app.route('/api/feedback', methods=['POST'])
        def feedback():
            try:
                data = request.get_json()
                logger.info(f"📝 Feedback received: {data}")
                
                return jsonify({
                    "status": "success",
                    "message": "Thank you for your feedback! We appreciate your input.",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"❌ Feedback error: {str(e)}")
                return jsonify({
                    "error": "Failed to save feedback",
                    "message": "We're sorry, but we couldn't save your feedback right now. Please try again later."
                }), 500
    
    def _process_query(self, query: str, user_id: str, location: Optional[str] = None) -> Dict[str, Any]:
        """Process user query with fallback responses"""
        try:
            # Search knowledge base
            relevant_info = self._search_knowledge_base(query)
            
            # Generate AI response or fallback
            ai_response = self._generate_response(query, relevant_info, location)
            
            # Update conversation memory (with limits)
            self._update_memory(user_id, query, ai_response)
            
            return {
                "message": ai_response,
                "sources": relevant_info[:2],  # Limit sources
                "location": location,
                "suggestions": self._get_suggestions(query),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {str(e)}")
            return {
                "message": self._get_fallback_response(query),
                "sources": [],
                "suggestions": ["Tell me about Malaysian food", "What are the best places to visit?", "Recommend things to do in KL"],
                "error": "Using fallback response",
                "success": True  # Still successful, just degraded
            }
    
    def _search_knowledge_base(self, query: str, max_results: int = 3):
        """Simple keyword-based search of knowledge base"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_base:
            content_lower = item['content'].lower()
            
            # Simple keyword matching
            score = 0
            query_words = query_lower.split()
            
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    if word in content_lower:
                        score += 1
            
            if score > 0:
                results.append({
                    "content": item['content'],
                    "category": item.get('category', 'general'),
                    "relevance": score,
                    "region": item.get('region'),
                    "type": item.get('type')
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:max_results]
    
    def _generate_response(self, query: str, context: list, location: Optional[str] = None) -> str:
        """Generate AI response using Gemini or fallback"""
        try:
            if not self.model:
                return self._get_fallback_response(query, context)
            
            # Build context from search results
            context_text = ""
            if context:
                context_text = "\n\nRelevant information:\n"
                for item in context[:2]:
                    context_text += f"- {item['content']}\n"
            
            location_text = f"\nUser location: {location}" if location else ""
            
            # Create prompt for Gemini
            prompt = f"""You are a friendly and knowledgeable Malaysia travel guide. Help visitors discover the best food, places, and experiences in Malaysia.

User question: {query}{location_text}{context_text}

Provide a helpful, accurate response about Malaysia in 100-150 words. Be specific with recommendations and include practical details when possible."""
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return self._get_fallback_response(query, context)
            
        except Exception as e:
            logger.error(f"❌ AI response error: {str(e)}")
            return self._get_fallback_response(query, context)
    
    def _get_fallback_response(self, query: str, context: list = None) -> str:
        """Generate fallback response when AI is unavailable"""
        query_lower = query.lower()
        
        # Food-related queries
        if any(word in query_lower for word in ['food', 'eat', 'restaurant', 'dish', 'cuisine']):
            return """Malaysia offers incredible culinary diversity! Must-try dishes include Nasi Lemak (coconut rice with sambal), Char Kway Teow (stir-fried noodles), and Rendang (spicy curry). Visit hawker centers in Kuala Lumpur or Penang for authentic street food experiences. Don't miss trying Roti Canai for breakfast and Cendol for dessert!"""
        
        # Destination queries
        elif any(word in query_lower for word in ['visit', 'place', 'destination', 'attraction', 'city']):
            return """Top Malaysian destinations include Kuala Lumpur (Petronas Towers, vibrant city life), Penang (UNESCO heritage George Town, street art), Langkawi (beautiful beaches, duty-free shopping), and Malacca (historical sites, Nyonya culture). For cooler weather, visit Cameron Highlands with its tea plantations and strawberry farms."""
        
        # General queries
        else:
            if context:
                relevant = context[0]['content'] if context else ""
                return f"Based on your question about Malaysia: {relevant}. Malaysia is a diverse country with amazing food, beautiful destinations, and rich cultural heritage. What specific aspect would you like to know more about?"
            else:
                return """Welcome to Malaysia! I'm here to help you discover the best this beautiful country has to offer. You can ask me about delicious Malaysian food, amazing places to visit, cultural experiences, or travel tips. What interests you most?"""
    
    def _get_suggestions(self, query: str) -> list:
        """Get relevant suggestions based on query"""
        query_lower = query.lower()
        
        if 'food' in query_lower:
            return ["What's the best street food in Penang?", "Where to find authentic Nasi Lemak?", "Malaysian desserts to try"]
        elif any(word in query_lower for word in ['visit', 'place', 'destination']):
            return ["Best things to do in Kuala Lumpur", "Island destinations in Malaysia", "Cultural sites to visit"]
        else:
            return ["Tell me about Malaysian cuisine", "Best places to visit in Malaysia", "What to do in Penang"]
    
    def _update_memory(self, user_id: str, query: str, response: str):
        """Update conversation memory with strict limits"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        # Add to memory with length limits
        self.conversation_memory[user_id].append({
            "query": query[:100],  # Limit query length
            "response": response[:200],  # Limit response length
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 2 exchanges to save memory
        self.conversation_memory[user_id] = self.conversation_memory[user_id][-2:]
        
        # Limit total users in memory
        if len(self.conversation_memory) > 50:
            # Remove oldest user conversations
            oldest_user = min(self.conversation_memory.keys())
            del self.conversation_memory[oldest_user]

# Create application instance
app_instance = MalaysiaAIGuideMinimal()
app = app_instance.app

if __name__ == '__main__':
    # Get port from environment (required for Render)
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 