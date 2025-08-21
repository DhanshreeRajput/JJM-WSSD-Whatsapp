"""Dependency injection for FastAPI."""

from fastapi import HTTPException, Depends
from app.core.sql_qa import EnhancedSQLQA

# Global SQL QA system instance
_sql_qa_system: EnhancedSQLQA = None

def set_sql_qa_system(system: EnhancedSQLQA):
    """Set the global SQL QA system instance."""
    global _sql_qa_system
    _sql_qa_system = system

async def get_sql_qa_system() -> EnhancedSQLQA:
    """Dependency to get the SQL QA system instance."""
    if _sql_qa_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Database not configured. Please configure database connection first."
        )
    return _sql_qa_system

def get_current_sql_qa_system() -> EnhancedSQLQA:
    """Get current SQL QA system without dependency injection."""
    return _sql_qa_system