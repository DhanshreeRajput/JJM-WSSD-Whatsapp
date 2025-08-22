"""SQL to NLP agent for converting SQL queries to natural language descriptions."""

import logging
from typing import Dict, Any, Optional, List
import asyncio
import re

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.config import settings
from app.core.database import DatabaseManager
from app.core.security import SecurityValidator

logger = logging.getLogger(__name__)


class SQLToNLPAgent:
    """Agent specialized in converting SQL queries to natural language descriptions."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize SQL to NLP agent."""
        self.db_manager = db_manager
        self.agent_name = "sql_to_nlp"
        self.security_validator = SecurityValidator()
        
        # Initialize LLM
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE
        )
        
        # Setup prompts
        self.sql_to_nlp_prompt = PromptTemplate(
            input_variables=["sql_query", "table_info", "context"],
            template="""You are an expert at explaining SQL queries in simple, natural language for the Water Supply and Sanitation Department.

Available database tables and their purpose:
{table_info}

SQL Query to explain:
{sql_query}

Additional context: {context}

Convert this SQL query into a clear, natural language description that a non-technical person can understand. Follow these guidelines:

1. **Start with action**: Begin with "This query..." or "This finds..." or "This shows..."
2. **Use domain language**: Use government and WSSD terms (schemes, grievances, citizens, districts, etc.)
3. **Explain joins simply**: Instead of "JOIN", say "combines data from" or "links information between"
4. **Describe conditions clearly**: Explain WHERE clauses as "filtering" or "looking for"
5. **Mention limits**: If there's a LIMIT, mention "showing the first X results"
6. **Avoid technical terms**: No SQL keywords, table names in caps, or database jargon

Examples:
- SELECT * FROM districts → "This shows all district information in the system"
- SELECT COUNT(*) FROM grievances WHERE status = 'pending' → "This counts how many grievances are still pending resolution"
- SELECT u.name, g.complaint_text FROM users u JOIN grievances g ON u.id = g.user_id → "This finds citizen names along with their complaint details by linking user and grievance information"

Your clear, natural language explanation:"""
        )
        
        # Enhanced context analysis prompt
        self.context_analysis_prompt = PromptTemplate(
            input_variables=["sql_query"],
            template="""Analyze this SQL query and identify its key components:

SQL Query: {sql_query}

Identify:
1. Main action (SELECT, COUNT, etc.)
2. Primary tables involved
3. Key conditions/filters
4. Relationships (JOINs)
5. Sorting/grouping
6. Limits/restrictions

Provide a structured analysis in this format:
Action: [what the query does]
Tables: [main tables involved]
Filters: [key conditions]
Relationships: [any joins or connections]
Output: [what kind of results this produces]

Analysis:"""
        )
        
        # Create LLM chains
        self.sql_to_nlp_chain = LLMChain(llm=self.llm, prompt=self.sql_to_nlp_prompt)
        self.context_chain = LLMChain(llm=self.llm, prompt=self.context_analysis_prompt)
        
        logger.info("SQL to NLP agent initialized")
    
    async def convert_sql_to_nlp(
        self, 
        sql_query: str, 
        context: str = "",
        include_analysis: bool = False
    ) -> Dict[str, Any]:
        """Convert SQL query to natural language description."""
        try:
            # Clean and validate the SQL query
            cleaned_sql = self._clean_sql_query(sql_query)
            
            if not cleaned_sql:
                return {
                    "agent": self.agent_name,
                    "sql_query": sql_query,
                    "description": "I couldn't understand this SQL query. Please check if it's valid.",
                    "is_safe": False,
                    "analysis": None,
                    "error": "Invalid or empty SQL query"
                }
            
            # Security validation
            is_safe, validation_message = self.security_validator.validate_sql(cleaned_sql)
            
            if not is_safe:
                return {
                    "agent": self.agent_name,
                    "sql_query": cleaned_sql,
                    "description": "This query appears to contain unsafe operations and cannot be explained.",
                    "is_safe": False,
                    "analysis": None,
                    "error": validation_message
                }
            
            # Get table context
            table_info = self._get_table_context()
            
            # Generate analysis if requested
            analysis = None
            if include_analysis:
                try:
                    analysis_response = await asyncio.to_thread(
                        self.context_chain.run,
                        sql_query=cleaned_sql
                    )
                    analysis = analysis_response.strip()
                except Exception as e:
                    logger.warning(f"Analysis generation failed: {e}")
                    analysis = "Analysis unavailable"
            
            # Generate natural language description
            description = await self._generate_description(cleaned_sql, table_info, context)
            
            return {
                "agent": self.agent_name,
                "sql_query": cleaned_sql,
                "original_sql": sql_query,
                "description": description,
                "is_safe": True,
                "validation_message": "Query is safe",
                "analysis": analysis,
                "table_info_used": len(table_info) > 0,
                "context_provided": bool(context.strip())
            }
            
        except Exception as e:
            logger.error(f"SQL to NLP conversion error: {e}")
            return {
                "agent": self.agent_name,
                "sql_query": sql_query,
                "description": "I encountered an error while analyzing this SQL query.",
                "is_safe": False,
                "analysis": None,
                "error": str(e)
            }
    
    async def _generate_description(self, sql_query: str, table_info: str, context: str) -> str:
        """Generate natural language description of the SQL query."""
        try:
            description_response = await asyncio.to_thread(
                self.sql_to_nlp_chain.run,
                sql_query=sql_query,
                table_info=table_info,
                context=context or "No additional context provided"
            )
            
            # Clean and format the response
            description = description_response.strip()
            
            # Ensure it starts with an action word
            if not description.lower().startswith(('this ', 'the query', 'this query')):
                description = f"This query {description.lower()}"
            
            return description
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return f"This appears to be a database query that {self._basic_sql_analysis(sql_query)}"
    
    def _basic_sql_analysis(self, sql_query: str) -> str:
        """Provide basic analysis as fallback."""
        sql_upper = sql_query.upper()
        
        if 'COUNT(' in sql_upper:
            return "counts records in the database"
        elif 'SELECT *' in sql_upper:
            return "retrieves all information from database tables"
        elif 'SELECT' in sql_upper and 'JOIN' in sql_upper:
            return "combines information from multiple database tables"
        elif 'SELECT' in sql_upper:
            return "retrieves specific information from the database"
        elif 'INSERT' in sql_upper:
            return "adds new records to the database"
        elif 'UPDATE' in sql_upper:
            return "modifies existing records in the database"
        elif 'DELETE' in sql_upper:
            return "removes records from the database"
        else:
            return "performs operations on the database"
    
    def _get_table_context(self) -> str:
        """Get context about available tables."""
        try:
            all_tables = self.db_manager.get_table_info()
            
            if not all_tables:
                return "Database schema information not available"
            
            # Create a summary of tables and their purposes
            table_descriptions = {
                "districts": "Administrative districts in Maharashtra",
                "circles": "Administrative circles within districts",
                "blocks": "Administrative blocks within circles",
                "villages": "Villages within blocks",
                "users": "Registered citizens and users",
                "grievances": "Citizen complaints and issues",
                "schemes": "Government schemes and programs",
                "grievance_categories": "Types of complaints",
                "grievance_resolve_tracks": "Grievance resolution tracking",
                "divisions": "Administrative divisions",
                "sub_divisions": "Administrative sub-divisions"
            }
            
            context_lines = []
            for table in all_tables[:15]:  # Limit for performance
                table_name = table.table_name
                description = table_descriptions.get(table_name, f"Data table: {table_name}")
                
                # Add key columns
                key_columns = [col.name for col in table.columns[:5]]
                context_lines.append(f"- {table_name}: {description} (columns: {', '.join(key_columns)})")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting table context: {e}")
            return "Database table information currently unavailable"
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate SQL query."""
        if not sql_query:
            return ""
        
        # Remove common markdown formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Remove leading/trailing whitespace
        sql_query = sql_query.strip()
        
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        
        # Normalize whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query)
        
        return sql_query.strip()
    
    async def explain_query_components(self, sql_query: str) -> Dict[str, Any]:
        """Provide detailed explanation of query components."""
        try:
            cleaned_sql = self._clean_sql_query(sql_query)
            
            if not cleaned_sql:
                return {"error": "Invalid SQL query provided"}
            
            # Analyze query structure
            components = self._analyze_query_structure(cleaned_sql)
            
            # Get detailed explanation
            explanation = await self._generate_description(cleaned_sql, self._get_table_context(), "")
            
            return {
                "agent": self.agent_name,
                "sql_query": cleaned_sql,
                "components": components,
                "explanation": explanation,
                "complexity": self._assess_query_complexity(cleaned_sql)
            }
            
        except Exception as e:
            logger.error(f"Query component explanation error: {e}")
            return {"error": str(e)}
    
    def _analyze_query_structure(self, sql_query: str) -> Dict[str, Any]:
        """Analyze the structure of a SQL query."""
        sql_upper = sql_query.upper()
        
        # Extract main components
        components = {
            "type": "unknown",
            "tables": [],
            "columns": [],
            "joins": [],
            "conditions": [],
            "has_aggregation": False,
            "has_grouping": False,
            "has_ordering": False,
            "has_limit": False
        }
        
        # Determine query type
        if sql_upper.startswith('SELECT'):
            components["type"] = "SELECT"
        elif sql_upper.startswith('INSERT'):
            components["type"] = "INSERT"
        elif sql_upper.startswith('UPDATE'):
            components["type"] = "UPDATE"
        elif sql_upper.startswith('DELETE'):
            components["type"] = "DELETE"
        
        # Extract table names (basic extraction)
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            components["tables"].append(from_match.group(1).lower())
        
        # Extract JOIN tables
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        for join_table in join_matches:
            components["tables"].append(join_table.lower())
            components["joins"].append(f"Joins with {join_table.lower()}")
        
        # Check for common clauses
        components["has_aggregation"] = any(func in sql_upper for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
        components["has_grouping"] = 'GROUP BY' in sql_upper
        components["has_ordering"] = 'ORDER BY' in sql_upper
        components["has_limit"] = 'LIMIT' in sql_upper
        
        # Extract WHERE conditions (simplified)
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER|\s+GROUP|\s+LIMIT|$)', sql_upper)
        if where_match:
            conditions_text = where_match.group(1)
            components["conditions"].append(f"Filters: {conditions_text}")
        
        return components
    
    def _assess_query_complexity(self, sql_query: str) -> str:
        """Assess the complexity level of a SQL query."""
        sql_upper = sql_query.upper()
        
        complexity_score = 0
        
        # Count complexity factors
        if 'JOIN' in sql_upper:
            complexity_score += sql_upper.count('JOIN') * 2
        if 'SUBQUERY' in sql_upper or '(' in sql_query:
            complexity_score += 3
        if any(func in sql_upper for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
            complexity_score += 1
        if 'GROUP BY' in sql_upper:
            complexity_score += 2
        if 'HAVING' in sql_upper:
            complexity_score += 2
        if 'UNION' in sql_upper:
            complexity_score += 3
        if 'CASE WHEN' in sql_upper:
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 3:
            return "moderate"
        elif complexity_score <= 6:
            return "complex"
        else:
            return "very_complex"
    
    async def batch_convert(self, sql_queries: List[str], context: str = "") -> List[Dict[str, Any]]:
        """Convert multiple SQL queries to NLP descriptions."""
        try:
            # Limit concurrent conversions
            semaphore = asyncio.Semaphore(3)
            
            async def convert_single_query(sql_query: str):
                async with semaphore:
                    return await self.convert_sql_to_nlp(sql_query, context)
            
            # Process all queries concurrently
            tasks = [convert_single_query(query) for query in sql_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error converting query {i}: {result}")
                    processed_results.append({
                        "agent": self.agent_name,
                        "sql_query": sql_queries[i],
                        "description": "Error processing this query",
                        "is_safe": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch conversion error: {e}")
            return [{"error": str(e)} for _ in sql_queries]
    
    async def health_check(self) -> bool:
        """Check if agent is healthy."""
        try:
            test_response = await asyncio.to_thread(
                self.llm.predict, 
                "Explain this query: SELECT 1"
            )
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as e:
            logger.error(f"SQL to NLP agent health check failed: {e}")
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent's capabilities."""
        return {
            "name": self.agent_name,
            "description": "Converts SQL queries to natural language descriptions",
            "capabilities": [
                "SQL query explanation in plain language",
                "Query complexity assessment",
                "Component analysis of SQL queries",
                "Batch processing of multiple queries",
                "Context-aware descriptions"
            ],
            "supported_sql_types": ["SELECT", "INSERT", "UPDATE", "DELETE"],
            "safety_features": ["SQL injection detection", "Query validation"],
            "languages_supported": ["English", "Hindi", "Marathi"]
        }