# Complete Malaysia Tourism RAG API Server
# Version 2.0 - Optimized for Google Cloud Run
# Full RAG implementation with fine-tuned Vertex AI model

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
import traceback

# Core dependencies
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# AI/ML dependencies
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Google Cloud dependencies
import google.auth
from google.cloud import aiplatform
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Configuration
# ========================================

class Config:
    """Central configuration management"""
    
    # Google Cloud Settings
    PROJECT_ID = "bright-coyote-463315-q8"
    LOCATION = "us-west1"
    ENDPOINT_ID = "4352232060597829632"  # TourismMalaysiaAI model
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    RETRIEVAL_TOP_K = 5
    MAX_TOKENS = 1024
    TEMPERATURE = 0.7
    
    # API settings
    CORS_ORIGINS = ["*"]  # Adjust for production
    MAX_QUERY_LENGTH = 1000
    
    # Database settings
    CHROMA_DB_PATH = "./vector_database"
    COLLECTION_NAME = "malaysia_travel_guide"

# ========================================
# Pydantic Models
# ========================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    top_k: Optional[int] = Field(default=Config.RETRIEVAL_TOP_K, ge=1, le=20)
    include_metadata: Optional[bool] = False

class QueryResponse(BaseModel):
    status: str
    query: str
    response: str
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    services: Dict[str, str]

class RelevanceCheckRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)

class RelevanceCheckResponse(BaseModel):
    query: str
    is_malaysia_travel_related: bool
    confidence: str
    keywords_found: List[str]

# ========================================
# RAG System Components
# ========================================

class RAGSystem:
    """Complete RAG system for Malaysia Tourism"""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.vertex_endpoint = None
        self.gemini_model = None
        self.malaysia_keywords = {
            # Places in Malaysia
            'places': [
                'malaysia', 'kuala lumpur', 'penang', 'langkawi', 'malacca', 'melaka',
                'sabah', 'sarawak', 'johor', 'kedah', 'kelantan', 'negeri sembilan',
                'pahang', 'perak', 'perlis', 'selangor', 'terengganu', 'putrajaya',
                'kl', 'georgetown', 'kota kinabalu', 'kuching', 'ipoh', 'shah alam'
            ],
            # Activities and attractions
            'activities': [
                'petronas towers', 'batu caves', 'genting highlands', 'cameron highlands',
                'mount kinabalu', 'sepilok', 'kinabatangan', 'redang', 'tioman',
                'perhentian', 'diving', 'snorkeling', 'jungle trekking', 'rainforest',
                'orangutan', 'proboscis monkey', 'durian', 'nasi lemak', 'rendang',
                'satay', 'laksa', 'char kway teow', 'roti canai', 'cendol'
            ],
            # Culture and general
            'culture': [
                'malay', 'chinese', 'indian', 'iban', 'kadazan', 'muslim', 'buddhist',
                'hindu', 'ramadan', 'chinese new year', 'deepavali', 'hari raya',
                'batik', 'songket', 'wayang kulit', 'gamelan', 'ringgit', 'rm'
            ]
        }
    
    async def initialize(self):
        """Initialize all RAG components"""
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize embedding model
            await self._init_embedding_model()
            
            # Initialize vector database
            await self._init_vector_database()
            
            # Initialize AI models
            await self._init_ai_models()
            
            logger.info("RAG system initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def _init_vector_database(self):
        """Initialize ChromaDB connection"""
        try:
            logger.info("Connecting to vector database...")
            
            # Check if database exists
            if not os.path.exists(Config.CHROMA_DB_PATH):
                raise FileNotFoundError(
                    f"Vector database not found at {Config.CHROMA_DB_PATH}. "
                    "Please run 1_build_database.py first to create the database."
                )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get the collection
            self.collection = self.chroma_client.get_collection(
                name=Config.COLLECTION_NAME
            )
            
            # Verify collection
            count = self.collection.count()
            logger.info(f"Connected to vector database with {count:,} documents")
            
            if count == 0:
                raise ValueError("Vector database is empty. Please rebuild the database.")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def _init_ai_models(self):
        """Initialize Vertex AI and Gemini models"""
        try:
            # Initialize Vertex AI
            logger.info("Initializing Vertex AI...")
            aiplatform.init(
                project=Config.PROJECT_ID,
                location=Config.LOCATION
            )
            
            # Initialize custom endpoint
            try:
                self.vertex_endpoint = aiplatform.Endpoint(
                    endpoint_name=f"projects/{Config.PROJECT_ID}/locations/{Config.LOCATION}/endpoints/{Config.ENDPOINT_ID}"
                )
                logger.info("TourismMalaysiaAI endpoint connected successfully")
            except Exception as e:
                logger.warning(f"Failed to connect to custom endpoint: {e}")
                self.vertex_endpoint = None
            
            # Initialize Gemini as fallback
            try:
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-pro')
                    logger.info("Gemini fallback model initialized")
                else:
                    logger.warning("GEMINI_API_KEY not found - Gemini fallback unavailable")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    def is_malaysia_travel_related(self, query: str) -> tuple[bool, List[str]]:
        """Check if query is related to Malaysia travel"""
        query_lower = query.lower()
        found_keywords = []
        
        # Check for Malaysia keywords
        for category, keywords in self.malaysia_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_keywords.append(keyword)
        
        # Travel-related keywords
        travel_keywords = [
            'travel', 'visit', 'trip', 'tour', 'vacation', 'holiday',
            'attraction', 'destination', 'hotel', 'restaurant', 'food',
            'activity', 'sightseeing', 'culture', 'history', 'beach',
            'mountain', 'city', 'budget', 'cost', 'price', 'itinerary'
        ]
        
        for keyword in travel_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        # Must have Malaysia-related keywords OR be clearly travel-related with context
        has_malaysia_keywords = any(
            keyword in found_keywords 
            for category in ['places', 'activities', 'culture'] 
            for keyword in self.malaysia_keywords[category]
        )
        
        has_travel_keywords = any(keyword in found_keywords for keyword in travel_keywords)
        
        is_relevant = has_malaysia_keywords or (has_travel_keywords and 'malaysia' in query_lower)
        
        return is_relevant, found_keywords
    
    async def retrieve_documents(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search vector database
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0] * len(results['documents'][0])
                )):
                    documents.append({
                        'content': doc,
                        'metadata': metadata,
                        'relevance_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    async def generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate response using AI model with retrieved context"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"[Document {i}]\n{doc['content']}\n")
            
            context = "\n".join(context_parts)
            
            # Create enhanced prompt
            system_prompt = """You are TourismMalaysiaAI, an expert AI assistant specializing in Malaysian tourism and travel.

Your Role:
- Provide comprehensive, accurate, and helpful information about traveling to Malaysia
- Focus on attractions, activities, food, culture, accommodations, and practical travel advice
- Use the provided context documents to give specific, detailed responses
- Always prioritize Malaysian destinations, experiences, and cultural insights

Guidelines:
- Answer ONLY questions related to Malaysia travel and tourism
- Provide practical details like prices, locations, timing, and tips when available
- If asked about non-Malaysia topics, politely redirect to Malaysia travel information
- Be enthusiastic and knowledgeable about Malaysia's diverse offerings
- Include specific recommendations based on the context provided

Context Documents:
{context}

User Question: {query}

Provide a comprehensive response based on the context above:"""

            prompt = system_prompt.format(context=context, query=query)
            
            # Try custom Vertex AI model first
            if self.vertex_endpoint:
                try:
                    instances = [{"content": prompt}]
                    response = self.vertex_endpoint.predict(instances=instances)
                    
                    if response and hasattr(response, 'predictions') and response.predictions:
                        generated_text = response.predictions[0]
                        if isinstance(generated_text, dict):
                            return generated_text.get('content', str(generated_text))
                        return str(generated_text)
                        
                except Exception as e:
                    logger.warning(f"Custom model failed, trying fallback: {e}")
            
            # Fallback to Gemini
            if self.gemini_model:
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=Config.MAX_TOKENS,
                            temperature=Config.TEMPERATURE,
                        )
                    )
                    return response.text
                except Exception as e:
                    logger.warning(f"Gemini fallback failed: {e}")
            
            # Final fallback - context-based response
            return self._create_fallback_response(query, documents)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error processing your question about {query}. Please try asking about specific Malaysia travel destinations, attractions, or activities."
    
    def _create_fallback_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a fallback response when AI models are unavailable"""
        if not documents:
            return "I apologize, but I couldn't find specific information about your query in my Malaysia tourism database. Please try asking about popular Malaysian destinations like Kuala Lumpur, Penang, Langkawi, or specific activities you're interested in."
        
        response_parts = [
            f"Based on my Malaysia tourism database, here's what I found regarding '{query}':\n"
        ]
        
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 for fallback
            content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
            response_parts.append(f"{i}. {content_preview}\n")
        
        response_parts.append("\nFor more detailed information, please ask specific questions about Malaysian destinations, activities, food, or travel tips.")
        
        return "\n".join(response_parts)

# ========================================
# FastAPI Application
# ========================================

# Initialize FastAPI app
app = FastAPI(
    title="Malaysia Tourism RAG API",
    description="Complete RAG system for Malaysia tourism with fine-tuned AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

# ========================================
# Startup and Shutdown Events
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        logger.info("Starting Malaysia Tourism RAG API...")
        await rag_system.initialize()
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Malaysia Tourism RAG API...")

# ========================================
# API Endpoints
# ========================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Malaysia Tourism RAG API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = "healthy" if rag_system.collection and rag_system.collection.count() > 0 else "unhealthy"
        
        # Check AI models
        vertex_status = "connected" if rag_system.vertex_endpoint else "unavailable"
        gemini_status = "connected" if rag_system.gemini_model else "unavailable"
        
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            timestamp=datetime.now().isoformat(),
            services={
                "vector_database": db_status,
                "vertex_ai": vertex_status,
                "gemini_fallback": gemini_status,
                "embedding_model": "loaded" if rag_system.embedding_model else "failed"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/api/query", response_model=QueryResponse)
async def query_tourism_info(request: QueryRequest):
    """Main query endpoint for Malaysia tourism information"""
    try:
        query = request.query.strip()
        
        # Check if query is Malaysia travel related
        is_relevant, keywords_found = rag_system.is_malaysia_travel_related(query)
        
        if not is_relevant:
            return QueryResponse(
                status="rejected",
                query=query,
                response="I specialize in Malaysia tourism and travel information. Please ask me about Malaysian destinations, attractions, food, culture, or travel tips. For example: 'What are the best places to visit in Kuala Lumpur?' or 'What Malaysian food should I try?'",
                metadata={
                    "rejection_reason": "query_not_malaysia_travel_related",
                    "timestamp": datetime.now().isoformat(),
                    "keywords_found": keywords_found
                }
            )
        
        # Retrieve relevant documents
        documents = await rag_system.retrieve_documents(query, request.top_k)
        
        if not documents:
            return QueryResponse(
                status="no_results",
                query=query,
                response="I couldn't find specific information about your query in my Malaysia tourism database. Please try asking about popular destinations like Kuala Lumpur, Penang, Langkawi, or specific activities you're interested in.",
                metadata={
                    "retrieved_count": 0,
                    "timestamp": datetime.now().isoformat(),
                    "keywords_found": keywords_found
                }
            )
        
        # Generate AI response
        response_text = await rag_system.generate_response(query, documents)
        
        # Prepare response
        response_data = QueryResponse(
            status="success",
            query=query,
            response=response_text,
            metadata={
                "retrieved_count": len(documents),
                "top_relevance_score": documents[0]["relevance_score"] if documents else 0,
                "processing_timestamp": datetime.now().isoformat(),
                "keywords_found": keywords_found,
                "model_used": "vertex_ai" if rag_system.vertex_endpoint else "gemini" if rag_system.gemini_model else "fallback"
            }
        )
        
        # Include documents if requested
        if request.include_metadata:
            response_data.retrieved_documents = documents
        
        return response_data
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error processing query: {str(e)}"
        )

@app.post("/api/check-relevance", response_model=RelevanceCheckResponse)
async def check_query_relevance(request: RelevanceCheckRequest):
    """Check if a query is related to Malaysia travel"""
    try:
        is_relevant, keywords_found = rag_system.is_malaysia_travel_related(request.query)
        
        # Determine confidence level
        confidence = "high" if len(keywords_found) >= 3 else "medium" if len(keywords_found) >= 1 else "low"
        
        return RelevanceCheckResponse(
            query=request.query,
            is_malaysia_travel_related=is_relevant,
            confidence=confidence,
            keywords_found=keywords_found
        )
        
    except Exception as e:
        logger.error(f"Relevance check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to check query relevance")

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        # Database stats
        doc_count = rag_system.collection.count() if rag_system.collection else 0
        
        # Model status
        models_available = {
            "vertex_ai": rag_system.vertex_endpoint is not None,
            "gemini": rag_system.gemini_model is not None,
            "embeddings": rag_system.embedding_model is not None
        }
        
        return {
            "database": {
                "total_documents": doc_count,
                "collection_name": Config.COLLECTION_NAME,
                "embedding_model": Config.EMBEDDING_MODEL
            },
            "models": models_available,
            "configuration": {
                "max_retrieval": Config.RETRIEVAL_TOP_K,
                "max_tokens": Config.MAX_TOKENS,
                "temperature": Config.TEMPERATURE
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system stats")

# ========================================
# Error Handlers
# ========================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ========================================
# Main Entry Point
# ========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable in production
        access_log=True
    ) 