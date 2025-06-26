#!/usr/bin/env python3
"""
Malaysia Tourism RAG Startup Script
Automatically builds database if needed and starts the API server
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_exists():
    """Check if vector database exists and has data"""
    db_path = Path("./vector_database")
    sqlite_path = db_path / "chroma.sqlite3"
    
    if not sqlite_path.exists():
        return False
    
    # Check if database has actual data
    try:
        import sqlite3
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM collections WHERE name='malaysia_travel_guide'")
        result = cursor.fetchone()
        conn.close()
        
        return result and result[0] > 0
    except Exception as e:
        logger.warning(f"Error checking database: {e}")
        return False

def build_database():
    """Build the vector database"""
    logger.info("Building vector database...")
    try:
        result = subprocess.run([sys.executable, "1_build_database.py"], 
                              capture_output=True, text=True, check=True)
        logger.info("Database build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Database build failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    logger.info("Starting API server...")
    try:
        # Use uvicorn to start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--workers", "1"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

def main():
    """Main startup sequence"""
    logger.info("=" * 50)
    logger.info("Malaysia Tourism RAG - Startup")
    logger.info("=" * 50)
    
    # Check if database exists
    if not check_database_exists():
        logger.info("Vector database not found or empty. Building database...")
        
        # Build database
        if not build_database():
            logger.error("Failed to build database. Cannot start service.")
            sys.exit(1)
    else:
        logger.info("Vector database found and ready!")
    
    # Start API server
    logger.info("Database ready. Starting API server...")
    start_api_server()

if __name__ == "__main__":
    main() 