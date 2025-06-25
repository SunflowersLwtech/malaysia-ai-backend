#!/usr/bin/env python3
"""
Malaysia AI Travel Guide - Backend API Server
Professional Flask API with Vertex AI integration and RAG capabilities
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Flask and CORS
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Google Cloud and AI
from google.cloud import aiplatform
from google.oauth2 import service_account
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Vector Database and ML
import chromadb
from sentence_transformers import SentenceTransformer

# Image processing
from PIL import Image

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalaysiaAIGuide:
    """
    Professional RAG system for Malaysia tourism and food recommendations
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.project_id = "bright-coyote-463315-q8"
        self.location = "us-west1"
        self.endpoint_id = "4352232060597829632"
        self.sheet_id = "1tE80wYY5yqEW0uR553RcnM7IkKqhQF1INtmgf54aRp4"
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.vertex_ai_endpoint = None
        self.google_sheet = None
        self.is_ready = False
        self.conversation_memory = {}
        
        logger.info("🚀 Initializing Malaysia AI Travel Guide...")
        self._initialize_components()
        self._setup_routes()
    
    def _initialize_components(self):
        """Initialize all AI and database components"""
        try:
            # 1. Initialize embedding model
            logger.info("📥 Loading embedding model...")
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            logger.info("✅ Embedding model loaded successfully!")
            
            # 2. Initialize ChromaDB
            logger.info("🗄️ Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(path="./vector_database")
            self.collection = self.chroma_client.get_or_create_collection(
                name="malaysia_travel_guide",
                metadata={"description": "Malaysia tourism and food data"}
            )
            logger.info("✅ ChromaDB initialized successfully!")
            
            # 3. Initialize Vertex AI
            self._setup_vertex_ai()
            
            # 4. Initialize Google Sheets
            self._setup_google_sheets()
            
            # 5. Load data if available
            self._load_data()
            
            self.is_ready = True
            logger.info("🎉 Malaysia AI Guide initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _setup_vertex_ai(self):
        """Setup Vertex AI connection"""
        try:
            logger.info("🔐 Setting up Vertex AI connection...")
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.location
            )
            
            # Get the endpoint
            endpoint_name = f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
            self.vertex_ai_endpoint = aiplatform.Endpoint(endpoint_name)
            
            logger.info("✅ Vertex AI connection established!")
            
        except Exception as e:
            logger.error(f"❌ Vertex AI setup failed: {str(e)}")
            self.vertex_ai_endpoint = None
    
    def _setup_google_sheets(self):
        """Setup Google Sheets connection for feedback"""
        try:
            logger.info("📊 Setting up Google Sheets connection...")
            
            # Define the scope
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Load credentials
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                'credentials.json', scope
            )
            
            # Authorize and open sheet
            client = gspread.authorize(creds)
            self.google_sheet = client.open_by_key(self.sheet_id).sheet1
            
            logger.info("✅ Google Sheets connection established!")
            
        except Exception as e:
            logger.error(f"❌ Google Sheets setup failed: {str(e)}")
            self.google_sheet = None
    
    def _load_data(self):
        """Load and process data files into vector database"""
        try:
            logger.info("📂 Loading data files...")
            
            data_files = [
                "RestaurantOriginalCSV.csv",
                "vertex_ai_training_data.jsonl",
                "Curation Queue (3).xlsx"
            ]
            
            all_data = []
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    logger.info(f"📄 Processing {file_path}...")
                    
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        df['source_file'] = file_path
                        all_data.append(df)
                    
                    elif file_path.endswith('.jsonl'):
                        data = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        data.append(json.loads(line))
                                    except:
                                        continue
                        df = pd.DataFrame(data)
                        df['source_file'] = file_path
                        all_data.append(df)
                    
                    elif file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                        df['source_file'] = file_path
                        all_data.append(df)
            
            if all_data:
                # Combine all data
                combined_df = pd.concat(all_data, ignore_index=True, sort=False)
                logger.info(f"📊 Combined {len(combined_df)} records from {len(all_data)} files")
                
                # Process and store in vector database
                self._process_and_store_data(combined_df)
            
        except Exception as e:
            logger.error(f"❌ Data loading error: {str(e)}")
    
    def _process_and_store_data(self, df):
        """Process data and store in vector database"""
        try:
            logger.info("🔧 Processing data for vector storage...")
            
            # Create combined text for each record
            df['combined_text'] = df.apply(self._create_combined_text, axis=1)
            
            # Process in batches
            batch_size = 100
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                # Generate embeddings
                texts = batch_df['combined_text'].tolist()
                embeddings = self.embedding_model.encode(texts)
                
                # Prepare metadata
                metadatas = []
                for _, row in batch_df.iterrows():
                    metadata = {}
                    for col in batch_df.columns:
                        if col != 'combined_text':
                            value = row[col]
                            if pd.notna(value):
                                metadata[col] = str(value)[:1000]  # Limit length
                    metadatas.append(metadata)
                
                # Generate IDs
                ids = [f"doc_{start_idx + j}" for j in range(len(batch_df))]
                
                # Store in ChromaDB
                self.collection.upsert(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"✅ Processed batch {i+1}/{total_batches}")
            
            logger.info(f"🎉 Successfully stored {len(df)} records in vector database!")
            
        except Exception as e:
            logger.error(f"❌ Data processing error: {str(e)}")
    
    def _create_combined_text(self, row):
        """Create searchable text from row data"""
        text_parts = []
        
        # Define important fields to prioritize
        priority_fields = ['name', 'title', 'description', 'category', 'location', 'address']
        
        for field in priority_fields:
            for col in row.index:
                if field.lower() in col.lower() and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
        
        # Add other non-null fields
        for col in row.index:
            if col not in ['combined_text', 'source_file'] and pd.notna(row[col]):
                value = str(row[col])
                if value not in text_parts and len(value) > 2:
                    text_parts.append(value)
        
        return ' | '.join(text_parts[:10])  # Limit to first 10 parts
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            return jsonify({
                "service": "Malaysia AI Travel Guide API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "GET /": "Service information",
                    "GET /health": "Health check",
                    "POST /ask": "Ask questions about Malaysia",
                    "POST /feedback": "Submit feedback",
                    "GET /stats": "Get database statistics"
                }
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({
                "status": "healthy" if self.is_ready else "initializing",
                "vertex_ai": "connected" if self.vertex_ai_endpoint else "disconnected",
                "database": "ready" if self.collection else "not_ready",
                "google_sheets": "connected" if self.google_sheet else "disconnected",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            try:
                data = request.get_json()
                if not data or 'query' not in data:
                    return jsonify({"error": "Missing 'query' in request"}), 400
                
                query = data['query']
                user_id = data.get('user_id', 'anonymous')
                location = data.get('location')
                image_data = data.get('image')
                max_results = data.get('max_results', 5)
                
                # Process the query
                result = self._process_query(
                    query=query,
                    user_id=user_id,
                    location=location,
                    image_data=image_data,
                    max_results=max_results
                )
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"❌ Query processing error: {str(e)}")
                return jsonify({
                    "error": "Internal server error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/feedback', methods=['POST'])
        def submit_feedback():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No data provided"}), 400
                
                # Save feedback to Google Sheets
                self._save_feedback(data)
                
                return jsonify({"success": True, "message": "Feedback saved successfully"})
                
            except Exception as e:
                logger.error(f"❌ Feedback error: {str(e)}")
                return jsonify({
                    "error": "Failed to save feedback",
                    "message": str(e)
                }), 500
        
        @self.app.route('/stats')
        def get_stats():
            try:
                stats = {
                    "total_records": self.collection.count() if self.collection else 0,
                    "is_ready": self.is_ready,
                    "services": {
                        "vertex_ai": bool(self.vertex_ai_endpoint),
                        "vector_database": bool(self.collection),
                        "google_sheets": bool(self.google_sheet)
                    }
                }
                return jsonify(stats)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _process_query(self, query: str, user_id: str, location: Optional[str] = None, 
                      image_data: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """Process user query and return comprehensive response"""
        try:
            logger.info(f"🔍 Processing query: {query[:100]}...")
            
            # 1. Search vector database
            search_results = self._search_vector_database(query, max_results)
            
            # 2. Generate AI response using Vertex AI
            ai_response = self._generate_ai_response(
                query=query, 
                context=search_results,
                location=location,
                user_id=user_id
            )
            
            # 3. Process image if provided
            image_analysis = None
            if image_data:
                image_analysis = self._analyze_image(image_data, query)
            
            # 4. Update conversation memory
            self._update_conversation_memory(user_id, query, ai_response)
            
            return {
                "query": query,
                "ai_response": ai_response,
                "sources": search_results,
                "image_analysis": image_analysis,
                "location_context": location,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {str(e)}")
            return {
                "query": query,
                "ai_response": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "success": False
            }
    
    def _search_vector_database(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search the vector database for relevant information"""
        try:
            if not self.collection or not self.embedding_model:
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
            for i in range(len(results['ids'][0])):
                result = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "source_file": results['metadatas'][0][i].get('source_file', 'unknown')
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {str(e)}")
            return []
    
    def _generate_ai_response(self, query: str, context: List[Dict], 
                             location: Optional[str] = None, user_id: str = "anonymous") -> str:
        """Generate AI response using Vertex AI"""
        try:
            if not self.vertex_ai_endpoint:
                return "I apologize, but the AI service is currently unavailable. Please try again later."
            
            # Prepare context
            context_text = ""
            if context:
                context_text = "\n\nRelevant information:\n"
                for i, item in enumerate(context[:3], 1):
                    context_text += f"{i}. {item['content'][:300]}...\n"
            
            # Get conversation history
            history = self.conversation_memory.get(user_id, [])
            history_text = ""
            if history:
                recent_history = history[-3:]  # Last 3 exchanges
                history_text = "\n\nRecent conversation:\n"
                for h in recent_history:
                    history_text += f"User: {h['query']}\nAssistant: {h['response'][:200]}...\n"
            
            # Create comprehensive prompt
            location_context = f"\nUser's location: {location}" if location else ""
            
            prompt = f"""You are a knowledgeable and friendly Malaysia travel and food guide AI assistant. 
Help users discover the best places to visit, eat, and experience in Malaysia.

User Query: {query}{location_context}{history_text}{context_text}

Please provide a helpful, accurate, and engaging response about Malaysia's tourism and food scene. 
Focus on specific recommendations with practical details like location, pricing, and what makes each place special.
Keep your response conversational and informative, around 150-200 words."""

            # Make prediction request
            instances = [{"content": prompt}]
            
            response = self.vertex_ai_endpoint.predict(instances=instances)
            
            if response and response.predictions:
                return response.predictions[0].get('content', 
                    "I'd be happy to help you explore Malaysia! Could you please rephrase your question?")
            else:
                return "I'd be happy to help you explore Malaysia! Could you please provide more details about what you're looking for?"
                
        except Exception as e:
            logger.error(f"❌ AI response generation error: {str(e)}")
            return "I'm currently experiencing technical difficulties. Please try asking your question again."
    
    def _analyze_image(self, image_data: str, query: str) -> Optional[Dict[str, Any]]:
        """Analyze uploaded image (basic implementation)"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_bytes))
            
            # Basic image analysis (can be enhanced with Vision API)
            return {
                "size": image.size,
                "format": image.format,
                "mode": image.mode,
                "analysis": "Image uploaded successfully. For detailed food/location analysis, please describe what you see in the image."
            }
            
        except Exception as e:
            logger.error(f"❌ Image analysis error: {str(e)}")
            return None
    
    def _update_conversation_memory(self, user_id: str, query: str, response: str):
        """Update conversation memory for context"""
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = []
        
        self.conversation_memory[user_id].append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges per user
        self.conversation_memory[user_id] = self.conversation_memory[user_id][-10:]
    
    def _save_feedback(self, feedback_data: Dict[str, Any]):
        """Save user feedback to Google Sheets"""
        try:
            if not self.google_sheet:
                logger.warning("Google Sheets not available for feedback")
                return
            
            # Prepare row data
            row_data = [
                datetime.now().isoformat(),  # timestamp
                feedback_data.get('user_query', ''),  # user_query
                feedback_data.get('ai_response', ''),  # ai_response
                feedback_data.get('rating', ''),  # rating
                feedback_data.get('feedback_text', ''),  # feedback_text
                feedback_data.get('location', '')  # location
            ]
            
            # Append to sheet
            self.google_sheet.append_row(row_data)
            logger.info("✅ Feedback saved to Google Sheets")
            
        except Exception as e:
            logger.error(f"❌ Feedback save error: {str(e)}")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"🚀 Starting Malaysia AI Guide API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Create application instance
malaysia_ai = MalaysiaAIGuide()

if __name__ == '__main__':
    # For local development
    malaysia_ai.run(debug=True)
else:
    # For production (Gunicorn)
    app = malaysia_ai.app