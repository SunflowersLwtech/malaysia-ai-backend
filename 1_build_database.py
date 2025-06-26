#!/usr/bin/env python3
"""
Malaysia Tourism RAG Database Builder
Build vector database with tourism data for semantic search
Compatible with Cloud Run deployment
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# Import required libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tourism_data() -> List[Dict[str, Any]]:
    """Load tourism data from JSONL file"""
    data_files = [
        "vertex_ai_training_data.jsonl",
        "../vertex_ai_training_data.jsonl"
    ]
    
    documents = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            logger.info(f"Loading data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():
                            data = json.loads(line.strip())
                            
                            # Handle Vertex AI training data format
                            if 'contents' in data:
                                # Extract user question and model response
                                user_text = ""
                                model_text = ""
                                
                                for content in data['contents']:
                                    if content.get('role') == 'user':
                                        for part in content.get('parts', []):
                                            if 'text' in part:
                                                user_text = part['text']
                                    elif content.get('role') == 'model':
                                        for part in content.get('parts', []):
                                            if 'text' in part:
                                                model_text = part['text']
                                
                                if user_text and model_text:
                                    content_text = f"Q: {user_text}\nA: {model_text}"
                                    documents.append({
                                        'content': content_text,
                                        'metadata': {
                                            'source': 'vertex_ai_training',
                                            'type': 'qa_pair',
                                            'line': line_num
                                        }
                                    })
                            
                            # Handle other formats
                            elif 'input_text' in data and 'output_text' in data:
                                content = f"Q: {data['input_text']}\nA: {data['output_text']}"
                                documents.append({
                                    'content': content,
                                    'metadata': {
                                        'source': 'training_data',
                                        'type': 'qa_pair',
                                        'line': line_num
                                    }
                                })
                            elif 'content' in data:
                                documents.append({
                                    'content': data['content'],
                                    'metadata': data.get('metadata', {'source': 'training_data', 'line': line_num})
                                })
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            break
    
    return documents

def build_vector_database():
    """Build ChromaDB vector database with tourism data"""
    
    # Initialize embedding model
    logger.info("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
    
    # Load tourism data
    logger.info("Loading tourism data...")
    documents = load_tourism_data()
    
    if not documents:
        logger.error("No documents loaded! Please check your data files.")
        return False
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create database directory
    db_path = "./vector_database"
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB with compatible settings
    logger.info("Initializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("malaysia_travel_guide")
        logger.info("Deleted existing collection")
    except:
        pass
    
    # Create new collection with simple metadata
    collection = client.create_collection(
        name="malaysia_travel_guide",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process documents in batches
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(documents)} documents in {total_batches} batches...")
    
    for batch_num in tqdm(range(total_batches), desc="Building database"):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(documents))
        batch_docs = documents[start_idx:end_idx]
        
        # Prepare batch data
        batch_texts = [doc['content'] for doc in batch_docs]
        batch_ids = [f"doc_{start_idx + i}" for i in range(len(batch_docs))]
        batch_metadata = [doc['metadata'] for doc in batch_docs]
        
        # Generate embeddings
        embeddings = model.encode(batch_texts).tolist()
        
        # Add to collection
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadata,
            ids=batch_ids
        )
        
        if batch_num % 10 == 0:
            logger.info(f"Processed batch {batch_num + 1}/{total_batches}")
    
    # Verify database
    total_count = collection.count()
    logger.info(f"Database built successfully!")
    logger.info(f"Total documents: {total_count:,}")
    
    # Test query
    logger.info("Testing database...")
    test_results = collection.query(
        query_texts=["What are popular attractions in Malaysia?"],
        n_results=3
    )
    logger.info(f"Test query returned {len(test_results['documents'][0])} results")
    
    return True

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Malaysia Tourism RAG Database Builder")
    logger.info("=" * 50)
    
    success = build_vector_database()
    
    if success:
        logger.info("✅ Database build completed successfully!")
        logger.info("🚀 Ready for deployment!")
    else:
        logger.error("❌ Database build failed!")
        exit(1) 