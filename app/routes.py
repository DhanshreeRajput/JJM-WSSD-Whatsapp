"""Updated API routes for LangGraph multi-agent system."""

import logging
import asyncio
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
import aiofiles

from app.models import (
    DatabaseConfig, QuestionRequest, QueryResponse, TableInfo, 
    HealthCheck, BatchQuestionRequest, BatchQueryResponse
)
from app.core.langgraph_system import LangGraphSQLQA
from app.dependencies import get_current_langgraph_system, set_langgraph_system
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/configure_database")
async def get_database_config_info():
    """Get information about database configuration status."""
    langgraph_system = get_current_langgraph_system()
    
    if langgraph_system:
        try:
            tables = langgraph_system.get_table_info()
            return {
                "status": "configured",
                "message": "LangGraph multi-agent system is configured and working",
                "database_info": {
                    "host": settings.POSTGRES_HOST,
                    "port": settings.POSTGRES_PORT,
                    "database": settings.POSTGRES_DB,
                    "schema": settings.DB_SCHEMA,
                    "tables_count": len(tables),
                    "uptime_seconds": langgraph_system.get_uptime(),
                    "system_type": "LangGraph Multi-Agent"
                },
                "agents": {
                    "router": "Routes questions to appropriate specialist agents",
                    "location": "Handles location and administrative queries",
                    "user": "Handles user and citizen queries", 
                    "grievance": "Handles grievance and complaint queries",
                    "schemes": "Handles government schemes queries",
                    "tracker": "Handles tracking and status queries"
                },
                "instruction": "Use POST method to reconfigure database if needed"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"LangGraph system configured but error accessing: {str(e)}",
                "instruction": "Use POST method to reconfigure database"
            }
    else:
        return {
            "status": "not_configured", 
            "message": "LangGraph multi-agent system not configured",
            "instruction": "Use POST method with database credentials to configure",
            "example_payload": {
                "host": "localhost",
                "port": 5432,
                "database": "wssd", 
                "username": "postgres",
                "password": "root@123",
                "schema": "public"
            },
            "note": "Send POST request to this endpoint with the above JSON structure"
        }

@router.post("/configure_database")
async def configure_database(config: DatabaseConfig, background_tasks: BackgroundTasks):
    """Configure database connection and initialize LangGraph system."""
    try:
        # Create database URI with proper URL encoding
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(config.password)
        database_uri = f"postgresql://{config.username}:{encoded_password}@{config.host}:{config.port}/{config.database}"
        
        logger.info(f"Configuring LangGraph system for {config.host}:{config.port}/{config.database}")
        
        # Initialize LangGraph SQL QA system
        langgraph_system = LangGraphSQLQA(database_uri)
        set_langgraph_system(langgraph_system)
        
        # Get table information to verify connection
        tables = langgraph_system.get_table_info()
        
        # Schedule background health check
        background_tasks.add_task(log_system_status, langgraph_system)
        
        logger.info(f"LangGraph system configured successfully: {len(tables)} tables found")
        
        return {
            "message": "LangGraph multi-agent system configured successfully",
            "database": config.database,
            "host": config.host,
            "port": config.port,
            "schema": config.schema,
            "tables_found": len(tables),
            "table_names": [table.table_name for table in tables[:10]],
            "total_tables": len(tables),
            "system_type": "LangGraph Multi-Agent",
            "agents_initialized": ["router", "location", "user", "grievance", "schemes", "tracker"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LangGraph system configuration failed: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"LangGraph system configuration failed: {str(e)}"
        )

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QuestionRequest):
    """Ask a natural language question using LangGraph multi-agent system."""
    langgraph_system = get_current_langgraph_system()
    
    if langgraph_system is None:
        raise HTTPException(
            status_code=503, 
            detail="LangGraph system not configured. Please configure database connection first."
        )
    
    try:
        logger.info(f"Processing question through LangGraph: {request.question[:100]}...")
        
        result = await langgraph_system.answer_question(
            question=request.question,
            use_safety=request.use_safety,
            limit_results=request.limit_results,
            response_style=getattr(request, 'response_style', 'brief')
        )
        
        # Log query for audit purposes
        logger.info(f"LangGraph query completed - Agent: {result.get('current_agent')}, "
                   f"Success: {result['is_safe']}, "
                   f"Execution time: {result['execution_time']:.2f}s, "
                   f"Rows: {result.get('row_count', 0)}")
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing question through LangGraph: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

@router.post("/ask_batch", response_model=BatchQueryResponse)
async def ask_batch_questions(request: BatchQuestionRequest):
    """Process multiple questions in batch using LangGraph."""
    langgraph_system = get_current_langgraph_system()
    
    if langgraph_system is None:
        raise HTTPException(
            status_code=503, 
            detail="LangGraph system not configured. Please configure database connection first."
        )
    
    try:
        start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0
        
        logger.info(f"Processing batch of {len(request.questions)} questions through LangGraph")
        
        # Process questions concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)
        
        async def process_question(question: str):
            async with semaphore:
                try:
                    result = await langgraph_system.answer_question(
                        question=question,
                        use_safety=request.use_safety,
                        response_style=getattr(request, 'response_style', 'brief')
                    )
                    return QueryResponse(**result)
                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    return QueryResponse(
                        question=question,
                        sql_query="",
                        result=None,
                        interpretation=f"Error: {str(e)}",
                        execution_time=0.0,
                        is_safe=False,
                        validation_message=str(e),
                        row_count=0,
                        response_style=getattr(request, 'response_style', 'brief')
                    )
        
        # Execute all questions concurrently
        tasks = [process_question(q) for q in request.questions]
        results = await asyncio.gather(*tasks)
        
        # Count successes and failures
        for result in results:
            if result.is_safe and "Error:" not in result.interpretation:
                successful_count += 1
            else:
                failed_count += 1
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"LangGraph batch processing completed - Success: {successful_count}, "
                   f"Failed: {failed_count}, Total time: {total_execution_time:.2f}s")
        
        return BatchQueryResponse(
            results=results,
            total_execution_time=total_execution_time,
            successful_count=successful_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        logger.error(f"Error processing batch questions through LangGraph: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing batch questions: {str(e)}"
        )

@router.get("/tables", response_model=List[TableInfo])
async def get_tables():
    """Get information about all tables in the database."""
    langgraph_system = get_current_langgraph_system()
    
    if langgraph_system is None:
        raise HTTPException(
            status_code=503, 
            detail="LangGraph system not configured. Please configure database connection first."
        )
    
    try:
        tables = langgraph_system.get_table_info()
        logger.info(f"Retrieved information for {len(tables)} tables")
        return tables
        
    except Exception as e:
        logger.error(f"Error getting table info: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting table info: {str(e)}"
        )

@router.get("/health", response_model=HealthCheck)
async def comprehensive_health_check():
    """Perform comprehensive health check of LangGraph system."""
    langgraph_system = get_current_langgraph_system()
    
    try:
        base_health = HealthCheck(
            status="healthy",
            database_status="not_configured",
            ollama_status="unknown"
        )
        
        if langgraph_system:
            health_result = await langgraph_system.health_check()
            
            base_health.database_status = health_result.get("database", "unknown")
            base_health.ollama_status = health_result.get("agents", "unknown")
            base_health.status = health_result.get("overall", "unknown")
            base_health.uptime = langgraph_system.get_uptime()
        
        return base_health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="error",
            database_status="error",
            ollama_status="error"
        )

@router.get("/system_info")
async def get_system_info():
    """Get comprehensive LangGraph system information."""
    langgraph_system = get_current_langgraph_system()
    
    info = {
        "application": {
            "name": "LangGraph Multi-Agent SQL QA System",
            "version": "3.0.0",
            "timestamp": datetime.now().isoformat(),
            "system_type": "LangGraph Multi-Agent"
        },
        "configuration": {
            "ollama_model": settings.OLLAMA_MODEL,
            "ollama_url": settings.OLLAMA_BASE_URL,
            "max_query_results": settings.MAX_QUERY_RESULTS,
            "safety_enabled": settings.ENABLE_SAFETY_BY_DEFAULT,
            "allowed_tables": settings.ALLOWED_TABLES if settings.ALLOWED_TABLES else "all"
        },
        "agents": {
            "router": "Intelligent question routing and response generation",
            "location": "Districts, circles, blocks, villages, administrative boundaries",
            "user": "Citizens, registrations, accounts, user management",
            "grievance": "Complaints, grievances, issues, resolutions",
            "schemes": "Government schemes, programs, initiatives",
            "tracker": "Status tracking, progress monitoring, logs"
        },
        "database": {
            "configured": langgraph_system is not None,
            "host": settings.DB_HOST if langgraph_system else "not configured",
            "port": settings.DB_PORT if langgraph_system else "not configured",
            "schema": settings.DB_SCHEMA if langgraph_system else "not configured"
        }
    }
    
    if langgraph_system:
        try:
            tables = langgraph_system.get_table_info()
            info["database"]["tables_count"] = len(tables)
            info["database"]["uptime_seconds"] = langgraph_system.get_uptime()
        except Exception as e:
            info["database"]["error"] = str(e)
    
    return info

# Background task for logging system status
async def log_system_status(langgraph_system: LangGraphSQLQA):
    """Log LangGraph system status in background."""
    try:
        health = await langgraph_system.health_check()
        logger.info(f"LangGraph system status - Database: {health['database']}, Agents: {health['agents']}")
    except Exception as e:
        logger.error(f"Background health check failed: {e}")