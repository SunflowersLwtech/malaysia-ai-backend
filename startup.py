#!/usr/bin/env python3
"""
Robust startup script for Malaysia AI Travel Guide
Ensures proper port binding for Cloud Run
"""

import os
import sys
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    logger.info("🚀 Starting Malaysia AI Travel Guide...")
    
    # Get port from environment (Cloud Run sets this)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    logger.info(f"🌐 Binding to {host}:{port}")
    
    # Verify environment
    logger.info("📋 Environment check:")
    logger.info(f"   PORT: {port}")
    logger.info(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    logger.info(f"   GEMINI_API_KEY: {'✅ set' if os.environ.get('GEMINI_API_KEY') else '❌ not set'}")
    
    try:
        # Import and run the FastAPI app
        logger.info("📦 Importing application...")
        import uvicorn
        from main import app
        
        logger.info("✅ Application imported successfully!")
        logger.info(f"🎯 Starting server on {host}:{port}")
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,
            timeout_keep_alive=300,
            access_log=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 