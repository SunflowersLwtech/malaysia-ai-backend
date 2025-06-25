#!/usr/bin/env python3
"""
Auto File Monitor System - Malaysia Tourism RAG Database
Monitors specified folders, automatically detects new files and adds them to vector database
Supports multiple file formats: JSONL, JSON, CSV, Excel, XML, TXT
Real-time database updates for live AI system
"""

import os
import sys
import time
import json
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import logging
import threading
import queue

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Please install watchdog: pip install watchdog")
    sys.exit(1)

# AI/ML dependencies
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration"""
    CHROMA_DB_PATH = "./vector_database"
    COLLECTION_NAME = "malaysia_travel_guide"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    BATCH_SIZE = 100
    
    # Supported file formats
    SUPPORTED_FORMATS = {
        '.jsonl': 'jsonl',
        '.json': 'json', 
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xml': 'xml',
        '.txt': 'text',
        '.tsv': 'tsv'
    }
    
    # Monitored folder paths
    WATCH_FOLDERS = [
        ".",  # Current folder
        "../",  # Parent folder
        "./data/",  # Data folder (if exists)
        "./uploads/",  # Upload folder (if exists)
        "./new_data/"  # New data folder (if exists)
    ]

class FileProcessor:
    """File processor - supports multiple formats"""
    
    @staticmethod
    def detect_format(file_path: str) -> str:
        """Detect file format"""
        extension = Path(file_path).suffix.lower()
        return DatabaseConfig.SUPPORTED_FORMATS.get(extension, 'unknown')
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")
        return data
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try to find arrays in the JSON
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        return value
                return [data]
            else:
                return [data]
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return []
    
    @staticmethod
    def load_csv(file_path: str, delimiter: str = ',') -> List[Dict]:
        """Load CSV file"""
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            return df.to_dict('records')
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'gbk']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                    logger.info(f"Successfully read CSV using encoding {encoding}")
                    return df.to_dict('records')
                except:
                    continue
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {e}")
        return []
    
    @staticmethod
    def load_excel(file_path: str) -> List[Dict]:
        """Load Excel file"""
        try:
            excel_file = pd.ExcelFile(file_path)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_data = df.to_dict('records')
                
                # Add sheet information
                for item in sheet_data:
                    item['_source_sheet'] = sheet_name
                
                all_data.extend(sheet_data)
            
            return all_data
        except Exception as e:
            logger.error(f"Failed to load Excel file {file_path}: {e}")
            return []
    
    @staticmethod
    def load_text(file_path: str) -> List[Dict]:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    data.append({
                        'content': line,
                        'line_number': i + 1
                    })
            return data
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            return []
    
    @classmethod
    def load_file(cls, file_path: str) -> List[Dict]:
        """Auto-detect and load file"""
        file_format = cls.detect_format(file_path)
        
        if file_format == 'jsonl':
            return cls.load_jsonl(file_path)
        elif file_format == 'json':
            return cls.load_json(file_path)
        elif file_format == 'csv':
            return cls.load_csv(file_path)
        elif file_format == 'tsv':
            return cls.load_csv(file_path, delimiter='\t')
        elif file_format == 'excel':
            return cls.load_excel(file_path)
        elif file_format == 'text':
            return cls.load_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_format}")
            return []

class VectorDatabase:
    """Vector database manager"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.processed_files = set()  # Track processed files
        self.file_hashes = {}  # Track file hashes
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
    def initialize(self):
        """Initialize database and models"""
        try:
            logger.info("Initializing vector database...")
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {DatabaseConfig.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(DatabaseConfig.EMBEDDING_MODEL)
            
            # Initialize ChromaDB
            os.makedirs(DatabaseConfig.CHROMA_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=DatabaseConfig.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(DatabaseConfig.COLLECTION_NAME)
                logger.info(f"Connected to existing database collection: {DatabaseConfig.COLLECTION_NAME}")
                
                # Load processed file information
                self._load_processed_files()
                
            except ValueError:
                # Collection doesn't exist, create new one
                self.collection = self.client.create_collection(
                    name=DatabaseConfig.COLLECTION_NAME,
                    metadata={"description": "Auto-Updated Malaysia Tourism Dataset"}
                )
                logger.info(f"Created new database collection: {DatabaseConfig.COLLECTION_NAME}")
            
            logger.info("Vector database initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def _load_processed_files(self):
        """Load records of processed files"""
        try:
            # Get processed files from database metadata
            results = self.collection.get()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'source_file' in metadata:
                        self.processed_files.add(metadata['source_file'])
                        if 'file_hash' in metadata:
                            self.file_hashes[metadata['source_file']] = metadata['file_hash']
            
            logger.info(f"Loaded {len(self.processed_files)} processed file records")
            
        except Exception as e:
            logger.warning(f"Failed to load processed file records: {e}")
    
    def extract_text_content(self, item: Dict) -> str:
        """Extract text content from data item"""
        if isinstance(item, str):
            return item
        
        if not isinstance(item, dict):
            return str(item)
        
        # Try common text fields
        text_fields = [
            'content', 'text', 'description', 'message', 'body',
            'input', 'output', 'title', 'name', 'comment',
            'review', 'feedback', 'summary', 'details'
        ]
        
        for field in text_fields:
            if field in item and item[field]:
                return str(item[field])
        
        # Combine all string values
        text_parts = []
        for key, value in item.items():
            if key.startswith('_'):  # Skip metadata fields
                continue
            
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
            elif isinstance(value, (int, float)):
                text_parts.append(str(value))
            elif isinstance(value, (dict, list)):
                text_parts.append(str(value))
        
        return " ".join(text_parts) if text_parts else ""
    
    def add_documents(self, data: List[Dict], source_file: str) -> int:
        """Add documents to database"""
        try:
            logger.info(f"Starting to process {len(data)} documents from file: {source_file}")
            
            # Get file hash
            file_hash = FileProcessor.get_file_hash(source_file)
            
            # Check if file was already processed
            if source_file in self.processed_files:
                if self.file_hashes.get(source_file) == file_hash:
                    logger.info(f"File unchanged, skipping: {source_file}")
                    return 0
                else:
                    logger.info(f"File modified, reprocessing: {source_file}")
                    # Remove old data
                    self._remove_file_documents(source_file)
            
            # Process documents
            total_added = 0
            batch_size = DatabaseConfig.BATCH_SIZE
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                batch_texts = []
                batch_ids = []
                batch_metadata = []
                
                for j, item in enumerate(batch):
                    # Create unique ID
                    doc_id = f"{Path(source_file).stem}_{i + j + 1:06d}_{int(time.time())}"
                    
                    # Extract text content
                    text_content = self.extract_text_content(item)
                    
                    if not text_content.strip():
                        continue
                    
                    batch_texts.append(text_content)
                    batch_ids.append(doc_id)
                    
                    # Prepare metadata
                    metadata = {
                        "source_file": os.path.basename(source_file),
                        "source_format": FileProcessor.detect_format(source_file),
                        "file_hash": file_hash,
                        "batch_index": i + j,
                        "added_at": datetime.now().isoformat(),
                        "doc_length": len(text_content)
                    }
                    
                    # Add key fields from original data
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key in ['role', 'category', 'type', 'id', 'location', 'price', 'rating', 'title', 'name']:
                                metadata[f"original_{key}"] = str(value)[:200]
                    
                    batch_metadata.append(metadata)
                
                # Generate embeddings and add to database
                if batch_texts:
                    embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                    
                    self.collection.add(
                        embeddings=embeddings.tolist(),
                        documents=batch_texts,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    
                    total_added += len(batch_texts)
            
            # Update processed file records
            self.processed_files.add(source_file)
            self.file_hashes[source_file] = file_hash
            
            logger.info(f"Successfully added {total_added} documents from file: {source_file}")
            return total_added
            
        except Exception as e:
            logger.error(f"Failed to add documents {source_file}: {e}")
            return 0
    
    def _remove_file_documents(self, source_file: str):
        """Remove all documents from specified file"""
        try:
            # Find all documents from this file
            results = self.collection.get()
            ids_to_delete = []
            
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    if metadata.get('source_file') == os.path.basename(source_file):
                        ids_to_delete.append(results['ids'][i])
            
            # Delete documents
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old documents from file: {source_file}")
                
        except Exception as e:
            logger.warning(f"Failed to delete old documents {source_file}: {e}")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "processed_files": len(self.processed_files),
                "collection_name": DatabaseConfig.COLLECTION_NAME,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

class FileMonitorHandler(FileSystemEventHandler):
    """File monitor handler"""
    
    def __init__(self, database: VectorDatabase):
        self.database = database
        self.processing_queue = set()  # Prevent duplicate processing
        
    def on_created(self, event):
        """Triggered when file is created"""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def on_modified(self, event):
        """Triggered when file is modified"""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def _process_file(self, file_path: str):
        """Process file"""
        try:
            # Avoid duplicate processing
            if file_path in self.processing_queue:
                return
            
            # Check file format
            file_format = FileProcessor.detect_format(file_path)
            if file_format == 'unknown':
                return
            
            # Wait for file write completion
            time.sleep(2)
            
            # Check if file exists and is not empty
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return
            
            self.processing_queue.add(file_path)
            
            logger.info(f"Detected new file: {file_path} (format: {file_format.upper()})")
            
            # Load and process file
            data = FileProcessor.load_file(file_path)
            if data:
                added_count = self.database.add_documents(data, file_path)
                logger.info(f"Successfully processed file {file_path}: added {added_count} documents")
                
                # Update API server if it's running
                self._notify_api_server_update()
            else:
                logger.warning(f"File {file_path} has no valid data")
            
            # Remove from processing queue
            self.processing_queue.discard(file_path)
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            self.processing_queue.discard(file_path)
    
    def _notify_api_server_update(self):
        """Notify API server about database update"""
        try:
            # This can be extended to send notifications to your API server
            # For example, through HTTP requests or message queues
            logger.info("Database updated - new data available for queries")
        except Exception as e:
            logger.warning(f"Failed to notify API server: {e}")

class AutoDatabaseMonitor:
    """Auto database monitor"""
    
    def __init__(self):
        self.database = VectorDatabase()
        self.observers = []
        self.running = False
        
    def initialize(self):
        """Initialize monitor"""
        if not self.database.initialize():
            return False
        
        logger.info("Auto file monitor initialized successfully")
        return True
    
    def process_existing_files(self):
        """Process existing files"""
        logger.info("Scanning existing files...")
        
        for folder in DatabaseConfig.WATCH_FOLDERS:
            if not os.path.exists(folder):
                continue
                
            for file_path in Path(folder).rglob("*"):
                if file_path.is_file():
                    file_format = FileProcessor.detect_format(str(file_path))
                    if file_format != 'unknown':
                        # Check if needs processing
                        file_name = os.path.basename(str(file_path))
                        if file_name not in self.database.processed_files:
                            logger.info(f"Processing existing file: {file_path}")
                            data = FileProcessor.load_file(str(file_path))
                            if data:
                                self.database.add_documents(data, str(file_path))
    
    def start_monitoring(self):
        """Start monitoring"""
        handler = FileMonitorHandler(self.database)
        
        for folder in DatabaseConfig.WATCH_FOLDERS:
            if os.path.exists(folder):
                observer = Observer()
                observer.schedule(handler, folder, recursive=True)
                observer.start()
                self.observers.append(observer)
                logger.info(f"Started monitoring folder: {os.path.abspath(folder)}")
        
        if not self.observers:
            logger.warning("No folders found to monitor")
            return False
        
        self.running = True
        return True
    
    def stop_monitoring(self):
        """Stop monitoring"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        self.observers.clear()
        self.running = False
        logger.info("File monitoring stopped")
    
    def get_status(self):
        """Get monitoring status"""
        stats = self.database.get_stats()
        stats['monitoring_folders'] = [
            os.path.abspath(folder) for folder in DatabaseConfig.WATCH_FOLDERS 
            if os.path.exists(folder)
        ]
        stats['is_monitoring'] = self.running
        return stats

def create_upload_folder():
    """Create upload folder for new data"""
    upload_folder = "./uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        logger.info(f"Created upload folder: {upload_folder}")
        
        # Create README file
        readme_content = """# Upload Folder for Malaysia Tourism RAG System

## Supported File Formats:
- JSONL (.jsonl) - Recommended for large datasets
- JSON (.json) - Single objects or arrays
- CSV (.csv) - Comma-separated values
- TSV (.tsv) - Tab-separated values
- Excel (.xlsx, .xls) - All sheets will be processed
- XML (.xml) - Basic XML structure
- TXT (.txt) - Plain text files

## How to Use:
1. Drop your data files into this folder
2. The system will automatically detect and process them
3. Your AI will immediately have access to the new data

## File Processing:
- Files are processed automatically when added
- Duplicate files are detected and skipped
- Modified files are reprocessed automatically
- All processing is logged in auto_database.log

## Data Fields:
The system looks for these fields in your data:
- content, text, description, message, body
- input, output, title, name, comment
- review, feedback, summary, details

Upload your files here and watch the magic happen!
"""
        
        with open(os.path.join(upload_folder, "README.md"), "w") as f:
            f.write(readme_content)

def main():
    """Main function"""
    print("Malaysia Tourism RAG - Auto File Monitor System")
    print("=" * 70)
    print("Supported formats: JSONL, JSON, CSV, TSV, Excel, XML, TXT")
    print("Monitoring mode: Auto-detect new files and add to vector database")
    print("Real-time updates: Your AI system gets new data immediately")
    print("=" * 70)
    
    # Create upload folder
    create_upload_folder()
    
    # Initialize monitor
    monitor = AutoDatabaseMonitor()
    
    if not monitor.initialize():
        logger.error("System initialization failed")
        sys.exit(1)
    
    try:
        # Process existing files
        monitor.process_existing_files()
        
        # Start monitoring
        if monitor.start_monitoring():
            logger.info("File monitoring system started")
            logger.info("You can now upload files to monitored folders for automatic processing")
            
            # Show status
            status = monitor.get_status()
            print(f"\nCurrent Status:")
            print(f"  - Database documents: {status.get('total_documents', 0):,}")
            print(f"  - Processed files: {status.get('processed_files', 0)}")
            print(f"  - Monitoring folders: {len(status.get('monitoring_folders', []))}")
            for folder in status.get('monitoring_folders', []):
                print(f"    * {folder}")
            
            print(f"\nPress Ctrl+C to stop monitoring...")
            print(f"Upload new files to any monitored folder for real-time updates!")
            
            # Keep running
            while True:
                time.sleep(30)  # Status update every 30 seconds
                status = monitor.get_status()
                logger.info(f"Status: {status['total_documents']:,} documents, {status['processed_files']} files processed")
                
        else:
            logger.error("Failed to start file monitoring")
            
    except KeyboardInterrupt:
        logger.info("Received stop signal...")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        monitor.stop_monitoring()
        logger.info("Auto monitoring system stopped")

if __name__ == "__main__":
    main() 