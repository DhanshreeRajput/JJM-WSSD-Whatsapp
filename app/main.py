"""FastAPI application main entry point with LangGraph support."""

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
from app.dependencies import set_langgraph_system

# Configure logging
def setup_logging():
    """Set up application logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Global variables
langgraph_system = None
app_start_time = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for LangGraph system."""
    logger.info("🚀 Starting LangGraph Multi-Agent SQL QA System...")
    
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
        
        # Initialize LangGraph system if database is configured
        if all([settings.DB_NAME, settings.DB_USER, settings.DB_PASSWORD]):
            global langgraph_system
            from app.core.langgraph_system import LangGraphSQLQA
            langgraph_system = LangGraphSQLQA()
            set_langgraph_system(langgraph_system)
            logger.info("✅ LangGraph multi-agent system pre-initialized")
        else:
            logger.info("⚠️ Database not configured - manual setup required")
            
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")
    
    logger.info("🎉 LangGraph application startup complete")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down LangGraph SQL QA System...")

# Create FastAPI application
app = FastAPI(
    title="LangGraph Multi-Agent SQL QA System",
    description="Production-ready natural language interface using LangGraph multi-agent architecture for WSSD",
    version="3.0.0",
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
    allowed_hosts=["*"]
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
        # Updated fallback interface
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LangGraph Multi-Agent SQL QA System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .header { background: #667eea; color: white; padding: 20px; border-radius: 8px; }
                .status { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
                .agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
                .agent-card { background: white; border: 2px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 LangGraph Multi-Agent SQL QA System</h1>
                    <p>Intelligent natural language interface with specialized agents for WSSD</p>
                </div>
                
                <div class="status">
                    <h2>🎯 Multi-Agent Architecture</h2>
                    <p>Our system uses specialized agents for different types of queries:</p>
                    <div class="agent-grid">
                        <div class="agent-card">
                            <h3>📍 Location Agent</h3>
                            <p>Districts, circles, blocks, villages</p>
                        </div>
                        <div class="agent-card">
                            <h3>👥 User Agent</h3>
                            <p>Citizens, registrations, accounts</p>
                        </div>
                        <div class="agent-card">
                            <h3>📝 Grievance Agent</h3>
                            <p>Complaints, issues, resolutions</p>
                        </div>
                        <div class="agent-card">
                            <h3>🏛️ Schemes Agent</h3>
                            <p>Government programs, initiatives</p>
                        </div>
                        <div class="agent-card">
                            <h3>📊 Tracker Agent</h3>
                            <p>Status updates, progress logs</p>
                        </div>
                        <div class="agent-card">
                            <h3>🎯 Router Agent</h3>
                            <p>Intelligent question routing</p>
                        </div>
                    </div>
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint for LangGraph system."""
    global langgraph_system
    
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
        "version": "3.0.0",
        "system_type": "LangGraph Multi-Agent"
    }
    
    # Check if LangGraph system is initialized
    if langgraph_system:
        try:
            health_result = await langgraph_system.health_check()
            status.update(health_result)
            status["agents_initialized"] = ["router", "location", "user", "grievance", "schemes", "tracker"]
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = str(e)
    else:
        status["langgraph_system"] = "not_configured"
    
    return status

# Development server runner
if __name__ == "__main__":
    logger.info(f"Starting LangGraph server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )