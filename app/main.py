"""FastAPI application main entry point."""

import logging
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Add app directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.api.routes import router
from app.core.sql_qa import EnhancedSQLQA

# Configure logging
def setup_logging():
    """Set up application logging."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up file and console handlers
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Global variables
sql_qa_system = None
app_start_time = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting SQL QA System...")
    
    # Startup
    try:
        # Test Ollama connection
        from langchain_community.llms import Ollama
        test_llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        test_response = test_llm.predict("Hello")
        logger.info("✅ Ollama connection successful")
        
        # Initialize SQL QA system if database is configured
        if all([settings.DB_NAME, settings.DB_USER, settings.DB_PASSWORD]):
            global sql_qa_system
            sql_qa_system = EnhancedSQLQA()
            logger.info("✅ SQL QA system pre-initialized")
        else:
            logger.info("⚠️  Database not configured - manual setup required")
            
    except Exception as e:
        logger.warning(f"⚠️  Startup warning: {e}")
    
    logger.info("🎉 Application startup complete")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SQL QA System...")

# Create FastAPI application
app = FastAPI(
    title="SQL Question-Answering System",
    description="Production-ready natural language interface for PostgreSQL databases using Ollama",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes
app.include_router(router, prefix="/api")

# Root endpoint - serve web interface
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the main web interface."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback simple interface
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SQL QA System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { background: #667eea; color: white; padding: 20px; border-radius: 8px; }
                .status { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 SQL Question-Answering System</h1>
                    <p>Natural language interface for PostgreSQL databases</p>
                </div>
                
                <div class="status">
                    <h2>📋 Getting Started</h2>
                    <ol>
                        <li>Configure your database connection via the API</li>
                        <li>Visit <a href="/docs">/docs</a> for interactive API documentation</li>
                        <li>Use the <code>/api/ask</code> endpoint to ask questions</li>
                    </ol>
                </div>
                
                <div class="status">
                    <h2>🔗 Quick Links</h2>
                    <ul>
                        <li><a href="/docs">📖 API Documentation (Swagger)</a></li>
                        <li><a href="/redoc">📚 API Documentation (ReDoc)</a></li>
                        <li><a href="/api/health">🏥 Health Check</a></li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="An internal server error occurred. Please check the logs."
    )

# Health check endpoint (also available at root level)
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    global sql_qa_system
    
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
        "version": "2.0.0"
    }
    
    # Check if SQL QA system is initialized
    if sql_qa_system:
        try:
            health_result = await sql_qa_system.health_check()
            status.update(health_result)
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = str(e)
    else:
        status["sql_qa_system"] = "not_configured"
    
    return status

# Development server runner
if __name__ == "__main__":
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )