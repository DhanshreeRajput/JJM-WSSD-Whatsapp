"""Location agent for handling location-related queries."""

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


class LocationAgent:
    """Agent specialized in location and administrative boundary queries."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize location agent."""
        self.db_manager = db_manager
        self.agent_name = "location"
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
            template="""You are a SQL expert for WSSD's location and administrative boundary data.

Available location tables:
{table_info}

User question: {question}

Generate a PostgreSQL query for location-related questions. Focus on:
- Administrative hierarchies: State → District → Division → Circle → Block → Village  
- Geographic boundaries and administrative areas
- Location relationships and mappings
- Area-wise data organization

Rules:
1. Return ONLY the SQL query, no explanations
2. Use only SELECT statements
3. Include LIMIT 100 for safety
4. Use proper JOINs for hierarchical data
5. Use ILIKE for case-insensitive text searches
6. Consider the administrative hierarchy in relationships

Location table relationships:
- districts (state_id, district_name, district_name_mar, id)
- divisions (state_id, circle_id, division_name, is_active)
- sub_divisions (state_id, division_id, division_name, is_active)
- circles, blocks, villages, grampanchayats, habitations

Common location queries:
- List districts: SELECT district_name FROM districts ORDER BY district_name LIMIT 100
- Find circles in a district: SELECT c.* FROM circles c JOIN districts d ON c.district_id = d.id WHERE d.district_name ILIKE '%name%' LIMIT 100
- Get hierarchy: SELECT d.district_name, c.circle_name, b.block_name FROM districts d JOIN circles c ON d.id = c.district_id JOIN blocks b ON c.id = b.circle_id LIMIT 100

SQL query only:"""
        )
        
        # Create LLM chain
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        
        logger.info("Location agent initialized")
    
    def _get_relevant_tables(self) -> List[str]:
        """Get location-related tables."""
        return [
            "districts", "divisions", "sub_divisions", "regions",
            "region_circles", "circles", "blocks", "villages",
            "grampanchayats", "habitations", "states"
        ]
    
    async def process(self, question: str) -> Dict[str, Any]:
        """Process a location-related question."""
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
            logger.error(f"Location agent error: {e}")
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
            
            columns = [f"  - {col.name}: {col.type}" for col in table.columns[:8]]  # Limit columns
            context.append(table_desc + "\n" + "\n".join(columns))
        
        return "\n\n".join(context)
    
    def _clean_sql_query(self, sql_response: str) -> str:
        """Clean and extract SQL query from response."""
        if not sql_response:
            return ""
        
        # Remove markdown formatting
        sql_query = sql_response.replace("```sql", "").replace("```", "").strip()
        
        # Extract first SELECT statement
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
            logger.error(f"Location agent health check failed: {e}")
            return False