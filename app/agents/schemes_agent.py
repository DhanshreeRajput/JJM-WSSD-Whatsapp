"""Schemes agent for handling scheme-related queries."""

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


class SchemesAgent:
    """Agent specialized in government schemes and programs."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize schemes agent."""
        self.db_manager = db_manager
        self.agent_name = "schemes"
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
            template="""You are a SQL expert for WSSD's government schemes and programs data.

Available schemes tables:
{table_info}

User question: {question}

Generate a PostgreSQL query for schemes-related questions. Focus on:
- Government schemes and programs
- Scheme categories and types
- Program implementation and coverage
- Scheme beneficiaries and statistics

Rules:
1. Return ONLY the SQL query, no explanations
2. Use only SELECT statements
3. Include LIMIT 100 for safety
4. Use JOINs for scheme category and type relationships
5. Focus on active/available schemes
6. Consider scheme status and eligibility

Scheme table relationships:
- scheme_categories (categories of government schemes)
- scheme_types (types of schemes)
- schemes (main schemes table with category_id and type_id references)

Common scheme queries:
- List schemes: SELECT scheme_name, description FROM schemes ORDER BY scheme_name LIMIT 100
- By category: SELECT sc.category_name, COUNT(*) FROM schemes s JOIN scheme_categories sc ON s.category_id = sc.id GROUP BY sc.category_name LIMIT 100
- By type: SELECT st.type_name, s.scheme_name FROM schemes s JOIN scheme_types st ON s.type_id = st.id LIMIT 100
- Active schemes: SELECT * FROM schemes WHERE status = 'active' ORDER BY created_at DESC LIMIT 100

SQL query only:"""
        )
        
        # Create LLM chain
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        
        logger.info("Schemes agent initialized")
    
    def _get_relevant_tables(self) -> List[str]:
        """Get schemes-related tables."""
        return ["scheme_categories", "scheme_types", "schemes"]
    
    async def process(self, question: str) -> Dict[str, Any]:
        """Process a schemes-related question."""
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
            logger.error(f"Schemes agent error: {e}")
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
            logger.error(f"Schemes agent health check failed: {e}")
            return False