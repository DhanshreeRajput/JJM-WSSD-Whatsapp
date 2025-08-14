"""Enhanced SQL Question-Answering engine with custom prompts and safety validation."""

import asyncio
import logging
import re
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from sqlalchemy import create_engine, text, inspect
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from app.config import settings
from app.core.security import SecurityValidator
from app.models import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class EnhancedSQLQA:
    """Enhanced SQL Question-Answering system with Ollama integration."""
    
    def __init__(self, database_uri: str = None):
        """Initialize the SQL QA system."""
        self.database_uri = database_uri or settings.database_url
        self.db = None
        self.llm = None
        self.security_validator = SecurityValidator()
        self.startup_time = datetime.now()
        
        # Initialize components
        self._initialize_database()
        self._initialize_llm()
        self._setup_prompts()
        
        logger.info("Enhanced SQL QA system initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection."""
        try:
            self.db = SQLDatabase.from_uri(self.database_uri)
            # Test connection
            self.db.run("SELECT 1")
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def _initialize_llm(self):
        """Initialize Ollama LLM."""
        try:
            self.llm = Ollama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.OLLAMA_TEMPERATURE,
                top_k=settings.OLLAMA_TOP_K,
                top_p=settings.OLLAMA_TOP_P,
                num_predict=settings.OLLAMA_NUM_PREDICT
            )
            
            # Test LLM connection
            test_response = self.llm.predict("Hello")
            if not test_response:
                raise ValueError("Empty response from LLM")
            
            logger.info("✅ Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise ConnectionError(f"Failed to initialize Ollama LLM: {e}")
    
    def _setup_prompts(self):
        """Set up custom prompts for SQL generation and interpretation."""
        
        # Clean SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "database_type", "schema_name"],
            template="""Generate a PostgreSQL query for the Water Supply and Sanitation Department database.

Available tables and columns:
{table_info}

User question: {question}

Rules:
1. Return ONLY the SQL query, no explanations
2. Use only SELECT statements
3. Use actual table/column names from the schema
4. Include LIMIT 100
5. If exploring structure, use information_schema

Examples:
Question: "What tables do we have?"
Answer: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name LIMIT 100

Question: "What are the district names?"
Answer: SELECT DISTINCT district_name FROM districts LIMIT 100

Question: "Tell me about blocks"
Answer: SELECT * FROM blocks LIMIT 10

SQL query only:"""
        )
        
        # Chatbot-style interpretation prompt  
        self.interpret_prompt = PromptTemplate(
            input_variables=["question", "sql_result", "sql_query", "row_count"],
            template="""You are a friendly AI assistant for the Water Supply and Sanitation Department. Respond naturally like you're having a conversation.

User asked: {question}
Data found: {sql_result}
Number of records: {row_count}

Important: Don't mention SQL, databases, or technical details. Just talk about the information like you personally looked it up.

Response style:
- Be conversational and helpful
- Use "I found...", "Looking at your data...", "Here's what I see..."
- Focus only on the actual information
- Suggest related questions they might ask
- Be encouraging and friendly

Examples:
- If found district names: "I found several districts in your system: Mumbai, Pune, Nashik, and 12 others. These represent the main administrative areas you're managing. Would you like to know more about any specific district?"
- If found table info: "I can help you with information about citizens, complaints, water connections, and administrative areas. What would you like to explore?"
- If no data: "I didn't find any records for that. The information might be stored differently. What specific details are you looking for?"

Your friendly response:"""
        )
        
        # Create LLM chains
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.interpret_chain = LLMChain(llm=self.llm, prompt=self.interpret_prompt)
        
        logger.info("Custom prompts configured successfully")
    
    async def answer_question(
        self, 
        question: str, 
        use_safety: bool = True,
        limit_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a natural language question and return chatbot-style results."""
        start_time = time.time()
        
        try:
            # Validate input
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            # Get database schema (simplified for AI)
            table_info = self._get_simplified_table_info()
            
            # Generate SQL query
            logger.info(f"Processing question: {question[:100]}...")
            
            sql_response = await asyncio.to_thread(
                self.sql_chain.run,
                question=question,
                table_info=table_info,
                database_type="PostgreSQL",
                schema_name=settings.DB_SCHEMA
            )
            
            sql_query = self._clean_sql_query(sql_response)
            logger.info(f"Generated SQL: {sql_query}")
            
            # Apply result limit if specified
            if limit_results:
                sql_query = self._apply_result_limit(sql_query, limit_results)
            
            # Validate query safety
            validation_result = {"is_safe": True, "message": "Query is safe"}
            if use_safety:
                is_safe, validation_message = self.security_validator.validate_sql(sql_query)
                validation_result = {"is_safe": is_safe, "message": validation_message}
                
                if not is_safe:
                    return self._create_chatbot_response(
                        question, sql_query, "I'm sorry, I can't process that request for security reasons. Could you try asking in a different way?", 
                        start_time, validation_result, 0
                    )
            
            # Execute SQL query
            try:
                logger.info("Fetching data...")
                result = self.db.run(sql_query)
                row_count = len(result) if isinstance(result, list) else 1 if result else 0
                logger.info(f"Data retrieved successfully, {row_count} rows found")
            except Exception as e:
                return self._create_chatbot_response(
                    question, sql_query, "I'm having trouble finding that information right now. Could you try asking about something else?", 
                    start_time, validation_result, 0
                )
            
            # Generate chatbot-style response
            logger.info("Creating response...")
            interpretation = await asyncio.to_thread(
                self.interpret_chain.run,
                question=question,
                sql_result=str(result)[:2000],  # Limit result size for LLM
                sql_query=sql_query,
                row_count=row_count
            )
            
            execution_time = time.time() - start_time
            
            return self._create_chatbot_response(
                question, sql_query, interpretation.strip(), 
                start_time, validation_result, row_count, result
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return self._create_chatbot_response(
                question, "", "I'm sorry, I encountered an issue while looking up that information. Could you try asking something else?", 
                start_time, {"is_safe": False, "message": str(e)}, 0
            )
    
    def _create_chatbot_response(
        self, 
        question: str, 
        sql_query: str, 
        interpretation: str, 
        start_time: float, 
        validation_result: Dict[str, Any],
        row_count: int,
        result: Any = None
    ) -> Dict[str, Any]:
        """Create chatbot-style response that hides technical details."""
        return {
            "question": question,
            "sql_query": sql_query,  # Keep for debugging, but hide in UI
            "result": result,
            "interpretation": interpretation,
            "execution_time": time.time() - start_time,
            "is_safe": validation_result["is_safe"],
            "validation_message": validation_result["message"],
            "row_count": row_count,
            "timestamp": datetime.now()
        }
    
    def _get_simplified_table_info(self) -> str:
        """Get simplified table information for the AI (no overwhelming details)."""
        try:
            tables = self.get_table_info()
            if not tables:
                return "Database tables available for queries."
            
            # Create a simple summary for the AI
            table_summary = []
            for table in tables:
                cols = [col.name for col in table.columns[:5]]  # Only first 5 columns
                table_summary.append(f"Table {table.table_name}: {', '.join(cols)}")
            
            return "\n".join(table_summary[:10])  # Only first 10 tables
        except Exception as e:
            logger.error(f"Error getting simplified table info: {e}")
            return "Database available for queries."
    
    def _clean_sql_query(self, sql_response: str) -> str:
        """Clean and extract SQL query from the response."""
        if not sql_response:
            return ""
        
        lines = sql_response.split('\n')
        sql_lines = []
        found_sql = False
        sql_query = ""  # Initialize with empty string
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and explanatory text
            if not line or line.startswith('Since') or line.startswith('To ') or line.startswith('Let'):
                continue
            # Look for SQL keywords to identify actual SQL
            if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
                found_sql = True
            if found_sql:
                sql_lines.append(line)
        
        if sql_lines:
            sql_query = ' '.join(sql_lines)
        
        # Remove markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        
        return sql_query.strip()  # Return the cleaned query or empty string
    
    def _apply_result_limit(self, sql_query: str, limit: int) -> str:
        """Apply or modify LIMIT clause in SQL query."""
        # Remove existing LIMIT clause
        sql_query = re.sub(r'\s+LIMIT\s+\d+', '', sql_query, flags=re.IGNORECASE)
        
        # Add new LIMIT
        return f"{sql_query} LIMIT {min(limit, settings.MAX_QUERY_RESULTS)}"
    
    def _create_error_response(
        self, 
        question: str, 
        sql_query: str, 
        error_message: str, 
        start_time: float, 
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "question": question,
            "sql_query": sql_query,
            "result": None,
            "interpretation": error_message,
            "execution_time": time.time() - start_time,
            "is_safe": validation_result["is_safe"],
            "validation_message": validation_result["message"],
            "row_count": 0,
            "timestamp": datetime.now()
        }
    
    def _get_formatted_table_info(self) -> str:
        """Get formatted table information for the LLM prompt with better structure."""
        try:
            tables = self.get_table_info()
            if not tables:
                return "No table information available. Use information_schema queries to explore."
            
            formatted_info = []
            
            # Add summary first
            formatted_info.append(f"Database contains {len(tables)} tables:")
            
            for table in tables:
                table_section = f"\nTable: {table.table_name}"
                if table.row_count and table.row_count > 0:
                    table_section += f" (approx. {table.row_count:,} rows)"
                elif table.row_count == 0:
                    table_section += " (empty table)"
                
                columns = []
                primary_keys = []
                foreign_keys = []
                
                for col in table.columns:
                    col_info = f"  - {col.name}: {col.type}"
                    if col.primary_key:
                        primary_keys.append(col.name)
                        col_info += " (PK)"
                    if not col.nullable:
                        col_info += " NOT NULL"
                    if col.foreign_key:
                        foreign_keys.append(f"{col.name} -> {col.foreign_key}")
                        col_info += f" -> {col.foreign_key}"
                    columns.append(col_info)
                
                table_section += "\n" + "\n".join(columns)
                
                # Add relationship info
                if foreign_keys:
                    table_section += f"\n  Foreign Keys: {', '.join(foreign_keys)}"
                
                formatted_info.append(table_section)
            
            return "\n".join(formatted_info)
        except Exception as e:
            logger.error(f"Error formatting table info: {e}")
            # Fallback to basic schema exploration
            return """Use these queries to explore the database:
- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
- SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'table_name';"""
    
    def get_table_info(self) -> List[TableInfo]:
        """Get detailed information about all database tables."""
        try:
            engine = self.db._engine
            inspector = inspect(engine)
            tables_info = []
            
            schema_name = settings.DB_SCHEMA
            table_names = inspector.get_table_names(schema=schema_name)
            
            # Filter tables if ALLOWED_TABLES is configured
            if settings.ALLOWED_TABLES:
                table_names = [t for t in table_names if t in settings.ALLOWED_TABLES]
            
            for table_name in table_names:
                try:
                    # Get column information
                    columns = []
                    for column in inspector.get_columns(table_name, schema=schema_name):
                        # Get foreign key info
                        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
                        fk_info = None
                        for fk in foreign_keys:
                            if column["name"] in fk["constrained_columns"]:
                                fk_table = fk["referred_table"]
                                fk_column = fk["referred_columns"][0]
                                fk_info = f"{fk_table}.{fk_column}"
                                break
                        
                        columns.append(ColumnInfo(
                            name=column["name"],
                            type=str(column["type"]),
                            nullable=column.get("nullable", True),
                            primary_key=column.get("primary_key", False),
                            foreign_key=fk_info,
                            default_value=str(column.get("default")) if column.get("default") else None
                        ))
                    
                    # Get approximate row count
                    try:
                        row_count_result = self.db.run(
                            f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{table_name}'"
                        )
                        row_count = int(row_count_result[0][0]) if row_count_result and row_count_result[0] else None
                    except:
                        row_count = None
                    
                    tables_info.append(TableInfo(
                        table_name=table_name,
                        schema_name=schema_name,
                        columns=columns,
                        row_count=row_count
                    ))
                
                except Exception as e:
                    logger.warning(f"Error getting info for table {table_name}: {e}")
                    continue
            
            logger.info(f"Retrieved information for {len(tables_info)} tables")
            return tables_info
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return []
    
    async def health_check(self) -> Dict[str, str]:
        """Perform comprehensive health check."""
        health_status = {
            "database": "unknown",
            "llm": "unknown",
            "overall": "unknown"
        }
        
        # Check database connection
        try:
            self.db.run("SELECT 1")
            health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
            logger.error(f"Database health check failed: {e}")
        
        # Check LLM connection
        try:
            response = await asyncio.to_thread(self.llm.predict, "Hello")
            health_status["llm"] = "healthy" if response else "error: empty response"
        except Exception as e:
            health_status["llm"] = f"error: {str(e)}"
            logger.error(f"LLM health check failed: {e}")
        
        # Determine overall status
        if all("healthy" in status for status in [health_status["database"], health_status["llm"]]):
            health_status["overall"] = "healthy"
        else:
            health_status["overall"] = "degraded"
        
        return health_status
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now() - self.startup_time).total_seconds()