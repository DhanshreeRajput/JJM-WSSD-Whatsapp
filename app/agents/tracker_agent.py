"""Tracker agent for handling tracking and status queries."""

import logging
from typing import Dict, List, Any
import asyncio

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.config import settings
from app.core.database import DatabaseManager
from app.core.security import SecurityValidator

logger = logging.getLogger(__name__)


class TrackerAgent:
    """Agent specialized in tracking and status monitoring."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize tracker agent."""
        self.db_manager = db_manager
        self.agent_name = "tracker"
        self.security_validator = SecurityValidator()
        
        # Initialize LLM
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE
        )
        
        # Setup prompts
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info"],
            template="""You are a SQL expert for WSSD's tracking and monitoring data.

Available tracking tables:
{table_info}

User question: {question}

Generate a PostgreSQL query for tracking-related questions. Focus on:
- Grievance resolution tracking
- Status updates and progress monitoring
- Assignment and acceptance tracking
- Timeline and history analysis

Rules:
1. Return ONLY the SQL query, no explanations
2. Use only SELECT statements
3. Include LIMIT 100 for safety
4. Use JOINs to get grievance details
5. Focus on status and progress information
6. Use date/time fields for tracking progress

Tracker table relationships:
- grievance_resolve_tracks (main tracking table)
- grievance_resolve_track_logs (detailed tracking logs)
- grievance_assigned_accept_reject_users (assignment tracking)

Common tracking queries:
- Track status: SELECT status, COUNT(*) FROM grievance_resolve_tracks GROUP BY status LIMIT 100
- Recent tracks: SELECT * FROM grievance_resolve_tracks ORDER BY updated_at DESC LIMIT 100
- Track logs: SELECT * FROM grievance_resolve_track_logs ORDER BY created_at DESC LIMIT 100
- Assignment status: SELECT status, COUNT(*) FROM grievance_assigned_accept_reject_users GROUP BY status LIMIT 100

SQL query only:"""
        )
        
        # Create LLM chain
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        
        logger.info("Tracker agent initialized")
    
    def _get_relevant_tables(self) -> List[str]:
        """Get tracker-related tables."""
        return [
            "grievance_resolve_tracks",
            "grievance_resolve_track_logs", 
            "grievance_assigned_accept_reject_users"
        ]
    
    async def process(self, question: str) -> Dict[str, Any]:
        """Process a tracking-related question."""
        try:
            # Get table information for this agent
            table_info = self._get_table_context()
            
            # Generate SQL query
            sql_query = await self._generate_sql(question, table_info)
            
            # Validate query
            is_safe, validation_message = self.security_validator.validate_sql(sql_query)
            
            if not is_safe:
                return {
                    "agent": self.agent_name,
                    "sql_query": sql_query,
                    "result": None,
                    "is_safe": False,
                    "validation_message": validation_message,
                    "row_count": 0
                }
            
            # Execute query
            result = await self.db_manager.execute_query(sql_query)
            row_count = len(result) if isinstance(result, list) else 1 if result else 0
            
            return {
                "agent": self.agent_name,
                "sql_query": sql_query,
                "result": result,
                "is_safe": True,
                "validation_message": "Query executed successfully",
                "row_count": row_count
            }
            
        except Exception as e:
            logger.error(f"Tracker agent error: {e}")
            return {
                "agent": self.agent_name,
                "sql_query": "",
                "result": None,
                "is_safe": False,
                "validation_message": str(e),
                "row_count": 0
            }
    
    async def _generate_sql(self, question: str, table_info: str) -> str:
        """Generate SQL query for the question."""
        try:
            sql_response = await asyncio.to_thread(
                self.sql_chain.run,
                question=question,
                table_info=table_info
            )
            return self._clean_sql_query(sql_response)
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return ""
    
    def _get_table_context(self) -> str:
        """Get table context for this agent."""
        relevant_tables = self._get_relevant_tables()
        all_tables = self.db_manager.get_table_info()
        
        # Filter to only relevant tables
        agent_tables = [table for table in all_tables if table.table_name in relevant_tables]
        
        # Format table information
        context = []
        for table in agent_tables:
            table_desc = f"Table {table.table_name}:"
            if table.row_count is not None:
                table_desc += f" ({table.row_count:,} rows)"
            
            columns = [f"  - {col.name}: {col.type}" for col in table.columns[:8]]
            context.append(table_desc + "\n" + "\n".join(columns))
        
        return "\n\n".join(context)
    
    def _clean_sql_query(self, sql_response: str) -> str:
        """Clean and extract SQL query from response."""
        if not sql_response:
            return ""
        
        sql_query = sql_response.replace("```sql", "").replace("```", "").strip()
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('--') or not line:
                continue
            sql_lines.append(line)
        
        return ' '.join(sql_lines)
    
    async def health_check(self) -> bool:
        """Check if agent is healthy."""
        try:
            test_response = await asyncio.to_thread(self.llm.predict, "Hello")
            return bool(test_response)
        except Exception as e:
            logger.error(f"Tracker agent health check failed: {e}")
            return False