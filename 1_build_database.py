import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import os
from datetime import datetime

def load_jsonl_data(file_path):
    """Load JSONL data with progress tracking"""
    print(f"Loading data from: {file_path}")
    
    # First, count total lines for progress bar
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Found {total_lines:,} total entries to process")
    
    # Load data with progress bar
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Loading JSONL entries"):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Successfully loaded {len(data):,} valid entries")
    return data

def initialize_embedding_model():
    """Initialize the sentence transformer model"""
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model ready")
    return model

def initialize_chromadb():
    """Initialize ChromaDB with persistent storage"""
    print("Initializing ChromaDB database...")
    
    # Create database directory
    db_path = "./vector_database"
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection
    collection_name = "malaysia_travel_guide"
    try:
        # Try to delete existing collection to start fresh
        client.delete_collection(collection_name)
        print(f"Removed existing collection: {collection_name}")
    except:
        pass  # Collection doesn't exist, which is fine
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Complete Malaysia Tourism Dataset"}
    )
    
    print(f"Created fresh collection: {collection_name}")
    return client, collection

def extract_text_content(item):
    """Extract text content from JSONL item based on common field patterns"""
    # Try different common field names for text content
    text_fields = ['content', 'text', 'description', 'message', 'body', 'input', 'output']
    
    for field in text_fields:
        if field in item and item[field]:
            return str(item[field])
    
    # If no standard field found, try to combine all string values
    text_parts = []
    for key, value in item.items():
        if isinstance(value, str) and value.strip():
            text_parts.append(value)
        elif isinstance(value, (dict, list)):
            # Handle nested structures
            text_parts.append(str(value))
    
    return " ".join(text_parts) if text_parts else ""

def process_and_index_data(data, embedding_model, collection):
    """Process all data and create vector embeddings"""
    print(f"\nStarting to process and index {len(data):,} entries...")
    print("This will take a few minutes - every single entry is being processed!")
    
    batch_size = 100  # Process in batches for memory efficiency
    total_processed = 0
    skipped_empty = 0
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]
        
        # Prepare batch data
        batch_texts = []
        batch_ids = []
        batch_metadata = []
        
        for j, item in enumerate(batch):
            # Create unique ID
            doc_id = f"doc_{i + j + 1:06d}"
            
            # Extract text content
            text_content = extract_text_content(item)
            
            # Skip empty content
            if not text_content.strip():
                skipped_empty += 1
                continue
                
            batch_texts.append(text_content)
            batch_ids.append(doc_id)
            
            # Prepare metadata (store original data)
            metadata = {
                "source": "malaysia_tourism_dataset",
                "batch_index": i + j,
                "processed_at": datetime.now().isoformat(),
                "doc_length": len(text_content)
            }
            
            # Add key fields from original data as metadata
            for key, value in item.items():
                if key in ['role', 'category', 'type', 'id', 'location', 'price', 'rating']:
                    metadata[f"original_{key}"] = str(value)[:200]  # Limit length
            
            batch_metadata.append(metadata)
        
        # Generate embeddings for this batch
        if batch_texts:
            try:
                embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                
                # Add to ChromaDB
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                total_processed += len(batch_texts)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
    
    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} empty entries")
    
    return total_processed

def verify_database(collection):
    """Verify the database was built correctly"""
    print("\nVerifying database...")
    
    try:
        # Get total count
        db_count = collection.count()
        
        # Test a sample query
        if db_count > 0:
            sample_results = collection.query(
                query_texts=["Malaysia tourism attractions"],
                n_results=3
            )
            
            print(f"Database verification successful!")
            print(f"Sample query returned {len(sample_results['documents'][0])} results")
        
        return db_count
        
    except Exception as e:
        print(f"Database verification failed: {e}")
        return 0

def main():
    """Main function to build the complete vector database"""
    print("Malaysia Tourism RAG Database Builder")
    print("=" * 60)
    
    # Configuration
    jsonl_file_path = "vertex_ai_training_data.jsonl"  # Adjust this path as needed
    
    # Check if file exists
    if not os.path.exists(jsonl_file_path):
        print(f"Error: File not found: {jsonl_file_path}")
        print("Please make sure your JSONL file is in the current directory")
        print("Available files:")
        for f in os.listdir("."):
            if f.endswith(('.jsonl', '.json')):
                print(f"  - {f}")
        return
    
    try:
        # Step 1: Load all data
        data = load_jsonl_data(jsonl_file_path)
        
        if not data:
            print("Error: No valid data found in the file")
            return
        
        # Step 2: Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Step 3: Initialize ChromaDB
        client, collection = initialize_chromadb()
        
        # Step 4: Process and index ALL data
        total_indexed = process_and_index_data(data, embedding_model, collection)
        
        # Step 5: Verify the database
        db_count = verify_database(collection)
        
        # Final results
        print("\n" + "=" * 60)
        print("DATABASE BUILD COMPLETE!")
        print("=" * 60)
        print(f"SUCCESS! Indexed a total of {total_indexed:,} documents")
        print(f"Database contains {db_count:,} searchable entries")
        print(f"Database location: ./vector_database")
        print(f"Collection name: malaysia_travel_guide")
        print(f"Original dataset size: {len(data):,} entries")
        print(f"Processing efficiency: {(total_indexed/len(data)*100):.1f}%")
        print("\nYour ENTIRE 40MB+ dataset is now efficiently searchable!")
        print("Ready to proceed to Step 2: API Server")
        
    except Exception as e:
        print(f"Error during database building: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 