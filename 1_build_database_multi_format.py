import pandas as pd
import json
import csv
import xlsx2csv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import os
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

def detect_file_format(file_path):
    """Automatically detect file format based on extension"""
    extension = Path(file_path).suffix.lower()
    format_map = {
        '.jsonl': 'jsonl',
        '.json': 'json',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xml': 'xml',
        '.txt': 'text',
        '.tsv': 'tsv'
    }
    return format_map.get(extension, 'unknown')

def load_jsonl_data(file_path):
    """Load JSONL data with progress tracking"""
    print(f"Loading JSONL data from: {file_path}")
    
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

def load_json_data(file_path):
    """Load regular JSON data"""
    print(f"Loading JSON data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            print(f"Successfully loaded {len(data):,} entries from JSON array")
            return data
        elif isinstance(data, dict):
            # Try to find arrays in the JSON
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"Found {len(value):,} entries in JSON key: {key}")
                    return value
            # If no arrays found, treat the dict as a single entry
            print("Treating JSON object as single entry")
            return [data]
        else:
            print("Treating JSON content as single entry")
            return [data]
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return []
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

def load_csv_data(file_path, delimiter=','):
    """Load CSV data"""
    print(f"Loading CSV data from: {file_path}")
    
    try:
        # Try to read with pandas first
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
        print(f"Successfully loaded {len(df):,} rows from CSV")
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        return data
        
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'cp1252', 'gbk']:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                print(f"Successfully loaded {len(df):,} rows from CSV (encoding: {encoding})")
                return df.to_dict('records')
            except:
                continue
        
        print("Error: Could not read CSV with any encoding")
        return []
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def load_excel_data(file_path):
    """Load Excel data"""
    print(f"Loading Excel data from: {file_path}")
    
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        all_data = []
        
        for sheet_name in excel_file.sheet_names:
            print(f"Reading sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_data = df.to_dict('records')
            
            # Add sheet name to metadata
            for item in sheet_data:
                item['_source_sheet'] = sheet_name
            
            all_data.extend(sheet_data)
        
        print(f"Successfully loaded {len(all_data):,} rows from Excel ({len(excel_file.sheet_names)} sheets)")
        return all_data
        
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []

def load_xml_data(file_path):
    """Load XML data"""
    print(f"Loading XML data from: {file_path}")
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = []
        # Try to extract data from XML
        for elem in root:
            item = {}
            # Add element text
            if elem.text and elem.text.strip():
                item['content'] = elem.text.strip()
            
            # Add attributes
            item.update(elem.attrib)
            
            # Add child elements
            for child in elem:
                if child.text and child.text.strip():
                    item[child.tag] = child.text.strip()
            
            if item:  # Only add non-empty items
                data.append(item)
        
        print(f"Successfully loaded {len(data):,} entries from XML")
        return data
        
    except Exception as e:
        print(f"Error loading XML: {e}")
        return []

def load_text_data(file_path):
    """Load plain text data (each line becomes an entry)"""
    print(f"Loading text data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # Skip empty lines
                data.append({
                    'content': line,
                    'line_number': i + 1
                })
        
        print(f"Successfully loaded {len(data):,} lines from text file")
        return data
        
    except Exception as e:
        print(f"Error loading text file: {e}")
        return []

def load_data_auto(file_path):
    """Automatically detect and load data from various formats"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    
    file_format = detect_file_format(file_path)
    print(f"Detected file format: {file_format}")
    
    if file_format == 'jsonl':
        return load_jsonl_data(file_path)
    elif file_format == 'json':
        return load_json_data(file_path)
    elif file_format == 'csv':
        return load_csv_data(file_path)
    elif file_format == 'tsv':
        return load_csv_data(file_path, delimiter='\t')
    elif file_format == 'excel':
        return load_excel_data(file_path)
    elif file_format == 'xml':
        return load_xml_data(file_path)
    elif file_format == 'text':
        return load_text_data(file_path)
    else:
        print(f"Unsupported file format: {file_format}")
        print("Supported formats: JSONL, JSON, CSV, TSV, Excel (.xlsx/.xls), XML, TXT")
        return []

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
        metadata={"description": "Complete Malaysia Tourism Dataset - Multi Format"}
    )
    
    print(f"Created fresh collection: {collection_name}")
    return client, collection

def extract_text_content(item):
    """Extract text content from various data structures"""
    # Handle different data types
    if isinstance(item, str):
        return item
    
    if not isinstance(item, dict):
        return str(item)
    
    # Try different common field names for text content
    text_fields = [
        'content', 'text', 'description', 'message', 'body', 
        'input', 'output', 'title', 'name', 'comment', 
        'review', 'feedback', 'summary', 'details'
    ]
    
    for field in text_fields:
        if field in item and item[field]:
            return str(item[field])
    
    # If no standard field found, try to combine all string values
    text_parts = []
    for key, value in item.items():
        if key.startswith('_'):  # Skip metadata fields
            continue
            
        if isinstance(value, str) and value.strip():
            text_parts.append(value)
        elif isinstance(value, (int, float)):
            text_parts.append(str(value))
        elif isinstance(value, (dict, list)):
            # Handle nested structures
            text_parts.append(str(value))
    
    return " ".join(text_parts) if text_parts else ""

def process_and_index_data(data, embedding_model, collection, source_file):
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
                "source_file": os.path.basename(source_file),
                "source_format": detect_file_format(source_file),
                "batch_index": i + j,
                "processed_at": datetime.now().isoformat(),
                "doc_length": len(text_content)
            }
            
            # Add key fields from original data as metadata
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in ['role', 'category', 'type', 'id', 'location', 'price', 'rating', 'title', 'name']:
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

def scan_available_files():
    """Scan current directory for supported data files"""
    supported_extensions = ['.jsonl', '.json', '.csv', '.xlsx', '.xls', '.xml', '.txt', '.tsv']
    available_files = []
    
    for file in os.listdir("."):
        file_path = Path(file)
        if file_path.suffix.lower() in supported_extensions:
            file_size = os.path.getsize(file) / (1024 * 1024)  # MB
            available_files.append((file, file_size, detect_file_format(file)))
    
    return available_files

def main():
    """Main function to build the complete vector database"""
    print("Malaysia Tourism RAG Database Builder - Multi Format Support")
    print("=" * 70)
    print("Supported formats: JSONL, JSON, CSV, TSV, Excel, XML, TXT")
    print("=" * 70)
    
    # Scan for available files
    available_files = scan_available_files()
    
    if not available_files:
        print("No supported data files found in current directory!")
        print("Please ensure you have one of these file types:")
        print("  - JSONL (.jsonl)")
        print("  - JSON (.json)")
        print("  - CSV (.csv)")
        print("  - TSV (.tsv)")
        print("  - Excel (.xlsx, .xls)")
        print("  - XML (.xml)")
        print("  - Text (.txt)")
        return
    
    # Display available files
    print(f"\nFound {len(available_files)} supported data files:")
    for i, (filename, size, format_type) in enumerate(available_files, 1):
        print(f"  {i}. {filename} ({size:.1f}MB, {format_type.upper()})")
    
    # Configuration - you can modify this
    default_file = "vertex_ai_training_data.jsonl"
    
    # Use default file if it exists, otherwise use the first available file
    if os.path.exists(default_file):
        data_file_path = default_file
        print(f"\nUsing default file: {data_file_path}")
    else:
        data_file_path = available_files[0][0]
        print(f"\nDefault file not found, using: {data_file_path}")
    
    # You can manually specify a different file here:
    # data_file_path = "your_custom_file.csv"  # Uncomment and modify as needed
    
    try:
        # Step 1: Load all data (auto-detect format)
        data = load_data_auto(data_file_path)
        
        if not data:
            print("Error: No valid data found in the file")
            return
        
        # Step 2: Initialize embedding model
        embedding_model = initialize_embedding_model()
        
        # Step 3: Initialize ChromaDB
        client, collection = initialize_chromadb()
        
        # Step 4: Process and index ALL data
        total_indexed = process_and_index_data(data, embedding_model, collection, data_file_path)
        
        # Step 5: Verify the database
        db_count = verify_database(collection)
        
        # Final results
        print("\n" + "=" * 70)
        print("DATABASE BUILD COMPLETE!")
        print("=" * 70)
        print(f"SUCCESS! Indexed a total of {total_indexed:,} documents")
        print(f"Database contains {db_count:,} searchable entries")
        print(f"Source file: {data_file_path}")
        print(f"Source format: {detect_file_format(data_file_path).upper()}")
        print(f"Database location: ./vector_database")
        print(f"Collection name: malaysia_travel_guide")
        print(f"Original dataset size: {len(data):,} entries")
        print(f"Processing efficiency: {(total_indexed/len(data)*100):.1f}%")
        print(f"\nYour ENTIRE dataset is now efficiently searchable!")
        print("Ready to proceed to Step 2: API Server")
        
    except Exception as e:
        print(f"Error during database building: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 