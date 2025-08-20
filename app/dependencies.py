"""Dependency injection for FastAPI with LangGraph support."""

from fastapi import HTTPException

# Global LangGraph SQL QA system instance
_langgraph_system = None

def set_langgraph_system(system):
    """Set the global LangGraph system instance."""
    global _langgraph_system
    _langgraph_system = system

async def get_langgraph_system():
    """Dependency to get the LangGraph system instance."""
    if _langgraph_system is None:
        raise HTTPException(
            status_code=503, 
            detail="LangGraph system not configured. Please configure database connection first."
        )
    return _langgraph_system

def get_current_langgraph_system():
    """Get current LangGraph system without dependency injection."""
    return _langgraph_system

# Keep legacy functions for backward compatibility
def set_sql_qa_system(system):
    """Legacy function - redirects to LangGraph system."""
    set_langgraph_system(system)

def get_current_sql_qa_system():
    """Legacy function - redirects to LangGraph system."""
    return get_current_langgraph_system()

async def get_sql_qa_system():
    """Legacy function - redirects to LangGraph system."""
    return await get_langgraph_system()