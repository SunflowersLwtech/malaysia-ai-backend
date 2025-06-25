#!/usr/bin/env python3
"""
Malaysia AI Travel Guide - Lightweight Backend API
Optimized for Render.com free tier (under 512MB RAM)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

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

class MalaysiaAIGuideLite:
    """
    Lightweight Malaysia AI Travel Guide - Optimized for free tier deployment
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.is_ready = False
        self.conversation_memory = {}
        self.knowledge_base = []
        
        logger.info("🚀 Initializing Malaysia AI Travel Guide (Lite)...")
        self._initialize_components()
        self._setup_routes()
    
    def _initialize_components(self):
        """Initialize AI components with minimal memory usage"""
        try:
            # 1. Setup Gemini AI
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini AI configured successfully!")
            else:
                logger.warning("⚠️ No Gemini API key found")
                self.model = None
            
            # 2. Load lightweight knowledge base
            self._load_lightweight_data()
            
            self.is_ready = True
            logger.info("🎉 Malaysia AI Guide Lite initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {str(e)}")
    
    def _load_lightweight_data(self):
        """Load data efficiently without heavy vector processing"""
        try:
            logger.info("📂 Loading knowledge base...")
            
            # Load CSV data efficiently
            csv_files = ["RestaurantOriginalCSV.csv", "vertex_ai_training_data.jsonl"]
            
            for file_path in csv_files:
                if os.path.exists(file_path):
                    logger.info(f"📄 Loading {file_path}...")
                    
                    if file_path.endswith('.csv'):
                        # Load only essential columns to save memory
                        df = pd.read_csv(file_path, nrows=1000)  # Limit rows for memory
                        for _, row in df.head(500).iterrows():  # Further limit for free tier
                            item = {
                                "content": self._extract_content(row),
                                "source": file_path,
                                "type": "restaurant" if "restaurant" in file_path.lower() else "general"
                            }
                            self.knowledge_base.append(item)
                    
                    elif file_path.endswith('.jsonl'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            count = 0
                            for line in f:
                                if count >= 100:  # Limit for memory
                                    break
                                try:
                                    data = json.loads(line.strip())
                                    item = {
                                        "content": str(data),
                                        "source": file_path,
                                        "type": "training_data"
                                    }
                                    self.knowledge_base.append(item)
                                    count += 1
                                except:
                                    continue
            
            logger.info(f"📊 Loaded {len(self.knowledge_base)} knowledge items")
            
        except Exception as e:
            logger.error(f"❌ Data loading error: {str(e)}")
    
    def _extract_content(self, row):
        """Extract meaningful content from data row"""
        content_parts = []
        
        # Common column names to check
        text_columns = ['name', 'description', 'location', 'address', 'cuisine', 'price', 'rating']
        
        for col in text_columns:
            if col in row.index and pd.notna(row[col]):
                content_parts.append(f"{col}: {row[col]}")
        
        # If no specific columns, use all string columns
        if not content_parts:
            for col, val in row.items():
                if isinstance(val, str) and len(val) > 5:
                    content_parts.append(f"{col}: {val}")
        
        return " | ".join(content_parts[:5])  # Limit to save memory
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            return jsonify({
                "service": "Malaysia AI Travel Guide",
                "status": "running",
                "version": "lite-1.0",
                "endpoints": ["/health", "/api/chat", "/api/status"]
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "memory_usage": "optimized"
            })
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                "is_ready": self.is_ready,
                "has_ai": self.model is not None,
                "knowledge_items": len(self.knowledge_base),
                "version": "lite",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            try:
                data = request.get_json()
                query = data.get('message', '')
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
                    "message": "I apologize, but I'm experiencing technical difficulties. Please try again."
                }), 500
        
        @self.app.route('/api/feedback', methods=['POST'])
        def feedback():
            try:
                data = request.get_json()
                logger.info(f"📝 Feedback received: {data}")
                
                return jsonify({
                    "status": "success",
                    "message": "Thank you for your feedback!"
                })
                
            except Exception as e:
                logger.error(f"❌ Feedback error: {str(e)}")
                return jsonify({"error": "Failed to save feedback"}), 500
    
    def _process_query(self, query: str, user_id: str, location: Optional[str] = None) -> Dict[str, Any]:
        """Process user query with lightweight approach"""
        try:
            # Simple keyword search in knowledge base
            relevant_info = self._search_knowledge_base(query)
            
            # Generate AI response
            ai_response = self._generate_response(query, relevant_info, location)
            
            # Update conversation memory (limited)
            self._update_memory(user_id, query, ai_response)
            
            return {
                "message": ai_response,
                "sources": relevant_info[:2],  # Limit sources
                "location": location,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {str(e)}")
            return {
                "message": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "error": str(e),
                "success": False
            }
    
    def _search_knowledge_base(self, query: str, max_results: int = 3):
        """Simple keyword-based search"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_base:
            content_lower = item['content'].lower()
            
            # Simple keyword matching
            score = 0
            for word in query_lower.split():
                if word in content_lower:
                    score += 1
            
            if score > 0:
                results.append({
                    "content": item['content'][:200] + "..." if len(item['content']) > 200 else item['content'],
                    "source": item['source'],
                    "relevance": score
                })
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:max_results]
    
    def _generate_response(self, query: str, context: list, location: Optional[str] = None) -> str:
        """Generate AI response using Gemini"""
        try:
            if not self.model:
                return "I'm a Malaysia travel guide! I can help you discover amazing places to visit and delicious food to try in Malaysia. What would you like to know?"
            
            # Build context
            context_text = ""
            if context:
                context_text = "\n\nRelevant information:\n"
                for item in context[:2]:  # Limit context for memory
                    context_text += f"- {item['content'][:150]}...\n"
            
            location_text = f"\nUser location: {location}" if location else ""
            
            # Create prompt
            prompt = f"""You are a helpful Malaysia travel and food guide. 
Answer the user's question about Malaysia tourism and food with accurate, friendly advice.

User question: {query}{location_text}{context_text}

Provide a helpful response in 100-150 words with specific recommendations."""
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "I'd be happy to help you explore Malaysia! Could you please ask me about specific places, food, or activities you're interested in?"
            
        except Exception as e:
            logger.error(f"❌ AI response error: {str(e)}")
            return "I'm here to help you discover the best of Malaysia! Please feel free to ask about places to visit, food to try, or activities to do."
    
    def _update_memory(self, user_id: str, query: str, response: str):
        """Update conversation memory with limits"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "query": query[:100],  # Limit length
            "response": response[:200],  # Limit length
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 3 exchanges to save memory
        self.conversation_memory[user_id] = self.conversation_memory[user_id][-3:]

# Create application instance
app_instance = MalaysiaAIGuideLite()
app = app_instance.app

if __name__ == '__main__':
    # Get port from environment (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 