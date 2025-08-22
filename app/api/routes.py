"""API routes for the SQL QA System."""

import logging
import asyncio
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
import aiofiles

from app.models import (
    DatabaseConfig, QuestionRequest, QueryResponse, TableInfo, 
    HealthCheck, BatchQuestionRequest, BatchQueryResponse,
    SQLToNLPRequest, SQLToNLPResponse, BatchSQLToNLPRequest, BatchSQLToNLPResponse
)
from app.core.sql_qa import EnhancedSQLQA
from app.dependencies import get_sql_qa_system
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variable to store the SQL QA system
_sql_qa_system = None

def set_sql_qa_system(system: EnhancedSQLQA):
    """Set the global SQL QA system instance."""
    global _sql_qa_system
    _sql_qa_system = system

def get_current_sql_qa_system():
    """Get the current SQL QA system instance."""
    return _sql_qa_system

@router.get("/configure_database")
async def get_database_config_info():
    """Get information about database configuration status."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system:
        try:
            tables = sql_qa_system.get_table_info()
            return {
                "status": "configured",
                "message": "Database is already configured and working",
                "database_info": {
                    "host": settings.POSTGRES_HOST,
                    "port": settings.POSTGRES_PORT,
                    "database": settings.POSTGRES_DB,
                    "schema": settings.DB_SCHEMA,
                    "tables_count": len(tables),
                    "uptime_seconds": sql_qa_system.get_uptime()
                },
                "instruction": "Use POST method to reconfigure database if needed"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Database configured but error accessing: {str(e)}",
                "instruction": "Use POST method to reconfigure database"
            }
    else:
        return {
            "status": "not_configured", 
            "message": "Database not configured",
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
    """Configure database connection and initialize SQL QA system."""
    try:
        # Create database URI with proper URL encoding
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(config.password)
        database_uri = f"postgresql://{config.username}:{encoded_password}@{config.host}:{config.port}/{config.database}"
        
        logger.info(f"Configuring database connection to {config.host}:{config.port}/{config.database}")
        
        # Initialize SQL QA system
        sql_qa_system = EnhancedSQLQA(database_uri)
        set_sql_qa_system(sql_qa_system)
        
        # Get table information to verify connection
        tables = sql_qa_system.get_table_info()
        
        # Schedule background health check
        background_tasks.add_task(log_system_status, sql_qa_system)
        
        logger.info(f"Database configured successfully: {len(tables)} tables found")
        
        return {
            "message": "Database configured successfully",
            "database": config.database,
            "host": config.host,
            "port": config.port,
            "schema": config.schema,
            "tables_found": len(tables),
            "table_names": [table.table_name for table in tables[:10]],  # Limit to first 10
            "total_tables": len(tables),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database configuration failed: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Database configuration failed: {str(e)}"
        )

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QuestionRequest):
    """Ask a natural language question about the database."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        result = await sql_qa_system.answer_question(
            question=request.question,
            use_safety=request.use_safety,
            limit_results=request.limit_results,
            response_style=request.response_style
        )
        
        # Log query for audit purposes
        logger.info(f"Query completed - Success: {result['is_safe']}, "
                   f"Execution time: {result['execution_time']:.2f}s, "
                   f"Rows: {result.get('row_count', 0)}, "
                   f"Style: {result.get('response_style', 'brief')}")
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

@router.post("/ask_batch", response_model=BatchQueryResponse)
async def ask_batch_questions(request: BatchQuestionRequest):
    """Process multiple questions in batch."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0
        
        logger.info(f"Processing batch of {len(request.questions)} questions")
        
        # Process questions concurrently with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent queries
        
        async def process_question(question: str):
            async with semaphore:
                try:
                    result = await sql_qa_system.answer_question(
                        question=question,
                        use_safety=request.use_safety,
                        response_style=request.response_style
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
                        response_style=request.response_style
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
        
        logger.info(f"Batch processing completed - Success: {successful_count}, "
                   f"Failed: {failed_count}, Total time: {total_execution_time:.2f}s")
        
        return BatchQueryResponse(
            results=results,
            total_execution_time=total_execution_time,
            successful_count=successful_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        logger.error(f"Error processing batch questions: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing batch questions: {str(e)}"
        )

@router.get("/tables", response_model=List[TableInfo])
async def get_tables():
    """Get information about all tables in the database."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        tables = sql_qa_system.get_table_info()
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
    """Perform comprehensive health check of all system components."""
    sql_qa_system = get_current_sql_qa_system()
    
    try:
        base_health = HealthCheck(
            status="healthy",
            database_status="not_configured",
            ollama_status="unknown"
        )
        
        if sql_qa_system:
            health_result = await sql_qa_system.health_check()
            
            base_health.database_status = health_result.get("database", "unknown")
            base_health.ollama_status = health_result.get("llm", "unknown")
            base_health.status = health_result.get("overall", "unknown")
            base_health.uptime = sql_qa_system.get_uptime()
        
        # Test Ollama connection independently if system not configured
        else:
            try:
                from langchain_community.llms import Ollama
                test_llm = Ollama(
                    model=settings.OLLAMA_MODEL,
                    base_url=settings.OLLAMA_BASE_URL
                )
                test_response = test_llm.predict("Hello")
                base_health.ollama_status = "healthy" if test_response else "error"
            except Exception as e:
                base_health.ollama_status = f"error: {str(e)}"
        
        return base_health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="error",
            database_status="error",
            ollama_status="error"
        )

@router.post("/upload_sql_file")
async def upload_sql_file(file: UploadFile = File(...)):
    """Upload a .pgsql file for database setup reference."""
    if not file.filename.endswith(('.sql', '.pgsql')):
        raise HTTPException(
            status_code=400, 
            detail="Only .sql and .pgsql files are allowed"
        )
    
    try:
        # Create uploads directory if it doesn't exist
        import os
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = f"{uploads_dir}/{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Read file content for preview
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content_preview = await f.read()
            preview = content_preview[:500] + "..." if len(content_preview) > 500 else content_preview
        
        logger.info(f"SQL file uploaded: {file.filename} ({len(content)} bytes)")
        
        return {
            "message": f"SQL file '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "size_bytes": len(content),
            "file_path": file_path,
            "preview": preview,
            "note": "File saved for reference. Execute manually in pgAdmin or psql to set up your database."
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error uploading file: {str(e)}"
        )

@router.get("/system_info")
async def get_system_info():
    """Get comprehensive system information."""
    sql_qa_system = get_current_sql_qa_system()
    
    info = {
        "application": {
            "name": "SQL Question-Answering System",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        },
        "configuration": {
            "ollama_model": settings.OLLAMA_MODEL,
            "ollama_url": settings.OLLAMA_BASE_URL,
            "max_query_results": settings.MAX_QUERY_RESULTS,
            "safety_enabled": settings.ENABLE_SAFETY_BY_DEFAULT,
            "allowed_tables": settings.ALLOWED_TABLES if settings.ALLOWED_TABLES else "all"
        },
        "database": {
            "configured": sql_qa_system is not None,
            "host": settings.DB_HOST if sql_qa_system else "not configured",
            "port": settings.DB_PORT if sql_qa_system else "not configured",
            "schema": settings.DB_SCHEMA if sql_qa_system else "not configured"
        },
        "response_styles": {
            "available": ["brief", "normal", "detailed"],
            "default": "brief",
            "description": {
                "brief": "1-2 sentences, key info only",
                "normal": "2-3 sentences with context", 
                "detailed": "Full explanation with suggestions"
            }
        }
    }
    
    if sql_qa_system:
        try:
            tables = sql_qa_system.get_table_info()
            info["database"]["tables_count"] = len(tables)
            info["database"]["uptime_seconds"] = sql_qa_system.get_uptime()
        except Exception as e:
            info["database"]["error"] = str(e)
    
    return info

@router.post("/explain_sql", response_model=SQLToNLPResponse)
async def explain_sql_query(request: SQLToNLPRequest):
    """Convert SQL query to natural language description."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        logger.info(f"Converting SQL to NLP: {request.sql_query[:100]}...")
        
        # Get SQL to NLP agent from LangGraph system
        if hasattr(sql_qa_system, 'sql_to_nlp_agent'):
            sql_to_nlp_agent = sql_qa_system.sql_to_nlp_agent
        else:
            # Initialize SQL to NLP agent if not available
            from app.agents.sql_to_nlp_agent import SQLToNLPAgent
            from app.core.database import DatabaseManager
            
            db_manager = DatabaseManager(sql_qa_system.database_uri)
            sql_to_nlp_agent = SQLToNLPAgent(db_manager)
        
        result = await sql_to_nlp_agent.convert_sql_to_nlp(
            sql_query=request.sql_query,
            context=request.context or "",
            include_analysis=request.include_analysis
        )
        
        # Log conversion for audit purposes
        logger.info(f"SQL to NLP conversion completed - Safe: {result['is_safe']}, "
                   f"Complexity: {result.get('complexity', 'unknown')}")
        
        return SQLToNLPResponse(
            sql_query=result["sql_query"],
            description=result["description"],
            is_safe=result["is_safe"],
            validation_message=result["validation_message"],
            analysis=result.get("analysis"),
            complexity=result.get("complexity", "unknown"),
            agent=result["agent"],
            language=request.language
        )
        
    except Exception as e:
        logger.error(f"Error converting SQL to NLP: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error converting SQL to NLP: {str(e)}"
        )

@router.post("/explain_sql_batch", response_model=BatchSQLToNLPResponse)
async def explain_sql_queries_batch(request: BatchSQLToNLPRequest):
    """Convert multiple SQL queries to natural language descriptions."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        start_time = datetime.now()
        logger.info(f"Processing batch SQL to NLP conversion for {len(request.sql_queries)} queries")
        
        # Get or initialize SQL to NLP agent
        if hasattr(sql_qa_system, 'sql_to_nlp_agent'):
            sql_to_nlp_agent = sql_qa_system.sql_to_nlp_agent
        else:
            from app.agents.sql_to_nlp_agent import SQLToNLPAgent
            from app.core.database import DatabaseManager
            
            db_manager = DatabaseManager(sql_qa_system.database_uri)
            sql_to_nlp_agent = SQLToNLPAgent(db_manager)
        
        # Process batch conversion
        batch_results = await sql_to_nlp_agent.batch_convert(
            sql_queries=request.sql_queries,
            context=request.context or ""
        )
        
        # Convert to response format
        results = []
        successful_count = 0
        failed_count = 0
        
        for result in batch_results:
            if "error" not in result:
                successful_count += 1
                results.append(SQLToNLPResponse(
                    sql_query=result["sql_query"],
                    description=result["description"],
                    is_safe=result["is_safe"],
                    validation_message=result["validation_message"],
                    analysis=result.get("analysis"),
                    complexity=result.get("complexity", "unknown"),
                    agent=result["agent"],
                    language=request.language
                ))
            else:
                failed_count += 1
                results.append(SQLToNLPResponse(
                    sql_query=result.get("sql_query", ""),
                    description=f"Error: {result['error']}",
                    is_safe=False,
                    validation_message=result["error"],
                    analysis=None,
                    complexity="unknown",
                    agent="sql_to_nlp",
                    language=request.language
                ))
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch SQL to NLP conversion completed - Success: {successful_count}, "
                   f"Failed: {failed_count}, Total time: {total_execution_time:.2f}s")
        
        return BatchSQLToNLPResponse(
            results=results,
            total_queries=len(request.sql_queries),
            successful_count=successful_count,
            failed_count=failed_count,
            total_execution_time=total_execution_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch SQL to NLP: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing batch SQL to NLP: {str(e)}"
        )

@router.post("/analyze_query")
async def analyze_query_components(request: SQLToNLPRequest):
    """Analyze SQL query components and provide detailed explanation."""
    sql_qa_system = get_current_sql_qa_system()
    
    if sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    
    try:
        logger.info(f"Analyzing SQL query components: {request.sql_query[:100]}...")
        
        # Get or initialize SQL to NLP agent
        if hasattr(sql_qa_system, 'sql_to_nlp_agent'):
            sql_to_nlp_agent = sql_qa_system.sql_to_nlp_agent
        else:
            from app.agents.sql_to_nlp_agent import SQLToNLPAgent
            from app.core.database import DatabaseManager
            
            db_manager = DatabaseManager(sql_qa_system.database_uri)
            sql_to_nlp_agent = SQLToNLPAgent(db_manager)
        
        result = await sql_to_nlp_agent.explain_query_components(request.sql_query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Query analysis completed - Complexity: {result.get('complexity', 'unknown')}")
        
        return {
            "sql_query": result["sql_query"],
            "explanation": result["explanation"],
            "components": result["components"],
            "complexity": result["complexity"],
            "agent": result["agent"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing query: {str(e)}"
        )

# Background task for logging system status
async def log_system_status(sql_qa_system: EnhancedSQLQA):
    """Log system status in background."""
    try:
        health = await sql_qa_system.health_check()
        logger.info(f"System status - Database: {health['database']}, LLM: {health['llm']}")
    except Exception as e:
        logger.error(f"Background health check failed: {e}")