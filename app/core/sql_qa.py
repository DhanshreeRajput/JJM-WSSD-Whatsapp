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
        self.table_cache = None
        self.last_cache_update = None
        
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
            test_result = self.db.run("SELECT 1")
            logger.info(f"✅ Database connection established - Test result: {test_result}")
            
            # Get initial table count
            tables_count = self.db.run("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            logger.info(f"📊 Found {tables_count[0][0] if tables_count else 0} tables in database")
            
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
            
            logger.info(f"✅ Ollama LLM initialized successfully - Model: {settings.OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise ConnectionError(f"Failed to initialize Ollama LLM: {e}")
    
    def _setup_prompts(self):
        """Set up custom prompts for SQL generation and interpretation."""
        
        # Enhanced SQL generation prompt with better guidance
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "database_type", "schema_name"],
            template="""You are a PostgreSQL expert for the Water Supply and Sanitation Department database.

Database Schema:
{table_info}

User Question: {question}

Generate a PostgreSQL query following these rules:
1. ONLY return the SQL query, no explanations
2. Use only SELECT statements (no INSERT, UPDATE, DELETE)
3. Always include LIMIT clause (max 100 rows)
4. Use exact table and column names from the schema above
5. For exploring database structure, use information_schema
6. Handle case-insensitive searches with ILIKE
7. Use proper JOINs when needed

Common patterns:
- List tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'
- Show columns: SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'table_name'
- Search by ID: SELECT * FROM table_name WHERE id = value
- Search text: SELECT * FROM table_name WHERE column_name ILIKE '%search_term%'
- Count records: SELECT COUNT(*) FROM table_name

SQL Query:"""
        )
        
        # Enhanced interpretation prompt with better context handling
        self.interpret_prompt = PromptTemplate(
            input_variables=["question", "sql_result", "sql_query", "row_count", "response_style"],
            template="""You are a friendly assistant for the Water Supply and Sanitation Department.

User asked: {question}
Database returned: {sql_result}
Number of records: {row_count}
Response style: {response_style}

Guidelines for {response_style} responses:
- brief: 1-2 sentences, direct and to the point
- normal: 2-3 sentences with some context
- detailed: Complete explanation with suggestions

Important rules:
1. Never mention SQL, databases, or technical terms
2. Speak naturally as if you personally looked up the information
3. If no data found, suggest alternative searches
4. For ID lookups, show key details clearly
5. For lists, show first few items then mention total count
6. Be helpful and encouraging

Examples:
- ID found: "Found Track 123: Route from Mumbai to Pune, Status: Active, Last updated Jan 2024"
- List found: "I found 15 districts: Mumbai, Pune, Nashik, Kolhapur... Would you like details about any specific one?"
- No data: "I couldn't find that specific ID. Try searching by name or check if the ID is correct."
- Empty database: "It looks like there's no data in that category yet. What else can I help you find?"

Your natural response:"""
        )
        
        # Create LLM chains
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.interpret_chain = LLMChain(llm=self.llm, prompt=self.interpret_prompt)
        
        logger.info("Enhanced prompts configured successfully")
    
    def _get_database_summary(self) -> str:
        """Get a comprehensive database summary for better AI understanding."""
        try:
            # Get table information with row counts
            tables_info = self.db.run("""
                SELECT 
                    t.table_name,
                    COALESCE(
                        (SELECT reltuples::bigint 
                         FROM pg_class c 
                         WHERE c.relname = t.table_name), 0
                    ) as estimated_rows,
                    (SELECT COUNT(*) 
                     FROM information_schema.columns 
                     WHERE table_name = t.table_name 
                     AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE t.table_schema = 'public'
                ORDER BY estimated_rows DESC
                LIMIT 20
            """)
            
            if not tables_info:
                return "Database appears to be empty or inaccessible."
            
            summary_parts = [f"Database contains {len(tables_info)} tables:"]
            
            for table_name, row_count, col_count in tables_info:
                # Get column details for this table
                try:
                    columns_info = self.db.run(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND table_schema = 'public'
                        ORDER BY ordinal_position
                        LIMIT 8
                    """)
                    
                    column_names = [col[0] for col in columns_info[:5]]  # First 5 columns
                    
                    table_desc = f"\nTable: {table_name}"
                    if row_count > 0:
                        table_desc += f" ({row_count:,} rows)"
                    elif row_count == 0:
                        table_desc += " (empty)"
                    
                    table_desc += f"\n  Columns: {', '.join(column_names)}"
                    if len(columns_info) > 5:
                        table_desc += f" ...+{len(columns_info)-5} more"
                    
                    summary_parts.append(table_desc)
                    
                except Exception as e:
                    logger.warning(f"Error getting columns for {table_name}: {e}")
                    summary_parts.append(f"\nTable: {table_name} (structure unavailable)")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
            return "Database connection available but structure unknown."
    
    async def answer_question(
        self, 
        question: str, 
        use_safety: bool = True,
        limit_results: Optional[int] = None,
        response_style: str = "brief"
    ) -> Dict[str, Any]:
        """Process a natural language question and return results."""
        start_time = time.time()
        
        try:
            # Validate input
            question = question.strip()
            if not question:
                raise ValueError("Question cannot be empty")
            
            logger.info(f"🤔 Processing question: {question}")
            
            # Get fresh database schema information
            table_info = self._get_database_summary()
            logger.info(f"📋 Database summary: {len(table_info)} characters")
            
            # Generate SQL query
            try:
                sql_response = await asyncio.to_thread(
                    self.sql_chain.run,
                    question=question,
                    table_info=table_info,
                    database_type="PostgreSQL",
                    schema_name=settings.DB_SCHEMA
                )
                
                sql_query = self._clean_sql_query(sql_response)
                logger.info(f"🔍 Generated SQL: {sql_query}")
                
                if not sql_query:
                    return self._create_error_response(
                        question, "", "I'm not sure how to search for that. Could you try rephrasing your question?", 
                        start_time, {"is_safe": True, "message": "No SQL generated"}, response_style
                    )
                
            except Exception as e:
                logger.error(f"SQL generation failed: {e}")
                return self._create_error_response(
                    question, "", "I'm having trouble understanding your question. Could you try asking differently?", 
                    start_time, {"is_safe": False, "message": str(e)}, response_style
                )
            
            # Apply result limit
            if limit_results:
                sql_query = self._apply_result_limit(sql_query, limit_results)
            
            # Validate query safety
            validation_result = {"is_safe": True, "message": "Query is safe"}
            if use_safety:
                try:
                    is_safe, validation_message = self.security_validator.validate_sql(sql_query)
                    validation_result = {"is_safe": is_safe, "message": validation_message}
                    
                    if not is_safe:
                        return self._create_error_response(
                            question, sql_query, "I can't process that request for security reasons.", 
                            start_time, validation_result, response_style
                        )
                except Exception as e:
                    logger.warning(f"Security validation failed: {e}")
            
            # Execute SQL query
            try:
                logger.info("🔄 Executing query...")
                result = self.db.run(sql_query)
                row_count = len(result) if isinstance(result, list) else 1 if result else 0
                logger.info(f"✅ Query executed successfully - {row_count} rows returned")
                
                # Handle empty results
                if row_count == 0:
                    return self._create_chatbot_response(
                        question, sql_query, "I couldn't find any records matching your request. Try a different search term or check if the information exists.",
                        start_time, validation_result, 0, response_style, result
                    )
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                return self._create_error_response(
                    question, sql_query, f"I encountered an issue while searching: {str(e)}", 
                    start_time, validation_result, response_style
                )
            
            # Generate natural language response
            try:
                logger.info("💬 Generating response...")
                
                # Limit result size for LLM processing
                limited_result = str(result)[:3000] if len(str(result)) > 3000 else str(result)
                
                interpretation = await asyncio.to_thread(
                    self.interpret_chain.run,
                    question=question,
                    sql_result=limited_result,
                    sql_query=sql_query,
                    row_count=row_count,
                    response_style=response_style
                )
                
                execution_time = time.time() - start_time
                logger.info(f"🎉 Response generated successfully in {execution_time:.2f}s")
                
                return self._create_chatbot_response(
                    question, sql_query, interpretation.strip(), 
                    start_time, validation_result, row_count, response_style, result
                )
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                # Fallback response with actual data
                fallback_response = f"I found {row_count} record(s). Here's what I discovered: {str(result)[:200]}..."
                return self._create_chatbot_response(
                    question, sql_query, fallback_response,
                    start_time, validation_result, row_count, response_style, result
                )
            
        except Exception as e:
            logger.error(f"❌ Critical error processing question: {e}")
            return self._create_error_response(
                question, "", "I'm experiencing technical difficulties. Please try again in a moment.", 
                start_time, {"is_safe": False, "message": str(e)}, response_style
            )
    
    def _create_chatbot_response(
        self, 
        question: str, 
        sql_query: str, 
        interpretation: str, 
        start_time: float, 
        validation_result: Dict[str, Any],
        row_count: int,
        response_style: str = "brief",
        result: Any = None
    ) -> Dict[str, Any]:
        """Create structured chatbot response."""
        return {
            "question": question,
            "sql_query": sql_query,
            "result": result,
            "interpretation": interpretation,
            "execution_time": time.time() - start_time,
            "is_safe": validation_result["is_safe"],
            "validation_message": validation_result["message"],
            "row_count": row_count,
            "response_style": response_style,
            "timestamp": datetime.now()
        }
    
    def _create_error_response(
        self, 
        question: str, 
        sql_query: str, 
        error_message: str, 
        start_time: float, 
        validation_result: Dict[str, Any],
        response_style: str = "brief"
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
            "response_style": response_style,
            "timestamp": datetime.now()
        }
    
    def _clean_sql_query(self, sql_response: str) -> str:
        """Clean and extract SQL query from LLM response."""
        if not sql_response:
            return ""
        
        # Remove markdown formatting
        sql_response = re.sub(r'```sql\s*', '', sql_response)
        sql_response = re.sub(r'```\s*', '', sql_response)
        
        # Split into lines and process
        lines = sql_response.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip explanatory text
            if any(skip_phrase in line.lower() for skip_phrase in [
                'here is', 'this query', 'the query', 'explanation:', 'note:'
            ]):
                continue
            
            # Look for SQL keywords
            if any(keyword in line.upper() for keyword in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                sql_lines.append(line)
            elif sql_lines:  # Continue if we've started collecting SQL
                sql_lines.append(line)
        
        if not sql_lines:
            # Fallback: try to extract any SELECT statement
            sql_match = re.search(r'(SELECT.*?)(?:\n|$)', sql_response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            return ""
        
        sql_query = ' '.join(sql_lines).strip()
        
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        
        # Ensure proper LIMIT
        if 'LIMIT' not in sql_query.upper():
            sql_query += ' LIMIT 100'
        
        return sql_query.strip()
    
    def _apply_result_limit(self, sql_query: str, limit: int) -> str:
        """Apply or modify LIMIT clause in SQL query."""
        # Remove existing LIMIT clause
        sql_query = re.sub(r'\s+LIMIT\s+\d+', '', sql_query, flags=re.IGNORECASE)
        
        # Add new LIMIT
        return f"{sql_query} LIMIT {min(limit, settings.MAX_QUERY_RESULTS)}"
    
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
            result = self.db.run("SELECT 1 as health_check")
            if result and result[0][0] == 1:
                health_status["database"] = "healthy"
            else:
                health_status["database"] = "error: unexpected result"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
            logger.error(f"Database health check failed: {e}")
        
        # Check LLM connection
        try:
            response = await asyncio.to_thread(self.llm.predict, "Test")
            health_status["llm"] = "healthy" if response and len(response.strip()) > 0 else "error: empty response"
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
    
    def debug_database_connection(self) -> Dict[str, Any]:
        """Debug database connection and return detailed info."""
        try:
            # Test basic connection
            test_result = self.db.run("SELECT 1 as test")
            
            # Get database info
            db_info = self.db.run("""
                SELECT 
                    current_database() as database_name,
                    current_user as current_user,
                    version() as postgres_version
            """)
            
            # Get table count and info
            tables_info = self.db.run("""
                SELECT 
                    COUNT(*) as total_tables,
                    SUM(CASE WHEN reltuples > 0 THEN 1 ELSE 0 END) as non_empty_tables
                FROM information_schema.tables t
                LEFT JOIN pg_class c ON c.relname = t.table_name
                WHERE t.table_schema = 'public'
            """)
            
            # Get sample table data
            sample_tables = self.db.run("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                LIMIT 5
            """)
            
            return {
                "connection_status": "successful",
                "test_query_result": test_result,
                "database_info": db_info[0] if db_info else None,
                "tables_summary": tables_info[0] if tables_info else None,
                "sample_tables": [t[0] for t in sample_tables] if sample_tables else [],
                "database_uri": self.database_uri.replace(settings.POSTGRES_PASSWORD, "***")
            }
            
        except Exception as e:
            return {
                "connection_status": "failed",
                "error": str(e),
                "database_uri": self.database_uri.replace(settings.POSTGRES_PASSWORD, "***")
            }