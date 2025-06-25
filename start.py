#!/usr/bin/env python3
"""
Robust startup script for Malaysia AI Travel Guide
Handles initialization errors gracefully and provides fallback functionality
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_dependencies():
    """Wait for dependencies to be ready"""
    max_retries = 30
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Test basic imports
            import fastapi
            import uvicorn
            logger.info("✅ Basic dependencies loaded")
            break
        except ImportError as e:
            if attempt < max_retries - 1:
                logger.warning(f"⏳ Dependencies not ready, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ Failed to load dependencies after {max_retries} attempts: {e}")
                sys.exit(1)

def start_server():
    """Start the FastAPI server with error handling"""
    try:
        logger.info("🚀 Starting Malaysia AI Travel Guide...")
        
        # Wait for dependencies
        wait_for_dependencies()
        
        # Import the main application
        from main import app
        
        # Get port from environment
        port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"🌐 Starting server on port {port}")
        
        # Start the server
        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,
            loop="asyncio",
            access_log=False  # Reduce noise in logs
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        
        # Start a minimal fallback server
        logger.info("🔄 Starting fallback server...")
        start_fallback_server(port)

def start_fallback_server(port):
    """Start a minimal fallback server if main app fails"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI(title="Malaysia AI - Fallback Mode")
    
    @app.get("/")
    async def fallback_root():
        return {"status": "fallback_mode", "message": "Malaysia AI is starting up..."}
    
    @app.get("/health")
    async def fallback_health():
        return {"status": "fallback", "ready": False}
    
    @app.get("/api/status")
    async def fallback_status():
        return {"is_ready": False, "mode": "fallback", "message": "Main application is initializing"}
    
    logger.info("🆘 Fallback server started")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    start_server() 