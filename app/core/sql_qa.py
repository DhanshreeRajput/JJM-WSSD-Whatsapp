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
            logger.error(f"❌ Database initialization failed: {e}")
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
        
        # Enhanced SQL generation prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "table_info", "database_type", "schema_name"],
            template="""You are an expert PostgreSQL query generator. Create a precise SQL query to answer the user's question.

DATABASE INFORMATION:
- Database Type: {database_type}
- Schema: {schema_name}
- Available Tables and Columns:
{table_info}

USER QUESTION: {question}

CRITICAL REQUIREMENTS:
1. Generate ONLY a SELECT query - no INSERT, UPDATE, DELETE, DROP, or other modifying operations
2. Use proper PostgreSQL syntax and functions
3. Only reference tables and columns that exist in the provided schema
4. Use appropriate JOINs when querying multiple tables
5. Apply proper WHERE clauses, GROUP BY, ORDER BY, and HAVING as needed
6. Use PostgreSQL-specific functions when beneficial (string_agg, extract, etc.)
7. Always include a LIMIT clause with maximum {max_results} rows
8. For date/time queries, use PostgreSQL date functions
9. Handle NULL values appropriately
10. If the question cannot be answered with available data, return exactly: "NO_QUERY_POSSIBLE"

EXAMPLES OF GOOD QUERIES:
- SELECT COUNT(*) FROM users LIMIT {max_results};
- SELECT name, email FROM customers WHERE city = 'New York' ORDER BY name LIMIT {max_results};
- SELECT category, AVG(price) as avg_price FROM products GROUP BY category ORDER BY avg_price DESC LIMIT {max_results};

Return ONLY the SQL query without any explanation, comments, or markdown formatting:""".replace("{max_results}", str(settings.MAX_QUERY_RESULTS))
        )
        
        # Enhanced interpretation prompt
        self.interpret_prompt = PromptTemplate(
            input_variables=["question", "sql_result", "sql_query", "row_count"],
            template="""Provide a clear, comprehensive answer to the user's question based on the query results.

ORIGINAL QUESTION: {question}
SQL QUERY EXECUTED: {sql_query}
QUERY RESULTS: {sql_result}
NUMBER OF ROWS: {row_count}

INSTRUCTIONS:
1. Provide a direct, natural language answer that addresses the user's question
2. Include specific numbers, names, and details from the results
3. If no data was found, clearly explain this and suggest possible reasons
4. For numerical results, provide context (e.g., "This represents X% of total")
5. For lists or tables, summarize key patterns or insights
6. Keep the answer concise but informative (2-4 sentences)
7. Do not mention the SQL query or technical details in your response
8. Use business-friendly language that non-technical users can understand

EXAMPLES:
- If asked about count: "There are 1,250 customers in the database."
- If asked about trends: "Sales have increased by 15% compared to last month, with the highest growth in the electronics category."
- If no results: "No customers were found matching those criteria. This might be because the filters were too specific or the data for that time period isn't available."

Natural Language Answer:"""
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
        """Process a natural language question and return structured results."""
        start_time = time.time()
        
        try:
            # Validate input
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            # Get database schema
            table_info = self._get_formatted_table_info()
            
            # Generate SQL query
            logger.info(f"Generating SQL for question: {question[:100]}...")
            
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
                    return self._create_error_response(
                        question, sql_query, f"Security validation failed: {validation_message}", 
                        start_time, validation_result
                    )
            
            # Execute SQL query
            try:
                logger.info("Executing SQL query...")
                result = self.db.run(sql_query)
                row_count = len(result) if isinstance(result, list) else 1 if result else 0
                logger.info(f"Query executed successfully, {row_count} rows returned")
            except Exception as e:
                return self._create_error_response(
                    question, sql_query, f"SQL execution error: {str(e)}", 
                    start_time, validation_result
                )
            
            # Interpret results
            logger.info("Interpreting query results...")
            interpretation = await asyncio.to_thread(
                self.interpret_chain.run,
                question=question,
                sql_result=str(result)[:2000],  # Limit result size for LLM
                sql_query=sql_query,
                row_count=row_count
            )
            
            execution_time = time.time() - start_time
            
            return {
                "question": question,
                "sql_query": sql_query,
                "result": result,
                "interpretation": interpretation.strip(),
                "execution_time": execution_time,
                "is_safe": validation_result["is_safe"],
                "validation_message": validation_result["message"],
                "row_count": row_count,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return self._create_error_response(
                question, "", f"Processing error: {str(e)}", 
                start_time, {"is_safe": False, "message": str(e)}
            )
    
    def _clean_sql_query(self, sql_response: str) -> str:
        """Clean and format SQL query from LLM response."""
        sql_query = sql_response.strip()
        
        # Remove markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        # Remove comments
        sql_query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        
        # Ensure single statement
        sql_query = sql_query.split(';')[0].strip()
        
        return sql_query
    
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
        """Get formatted table information for the LLM prompt."""
        try:
            tables = self.get_table_info()
            formatted_info = []
            
            for table in tables:
                table_section = f"\nTable: {table.schema_name}.{table.table_name}"
                if table.row_count:
                    table_section += f" (approx. {table.row_count:,} rows)"
                
                columns = []
                for col in table.columns:
                    col_info = f"  - {col.name}: {col.type}"
                    if col.primary_key:
                        col_info += " (PK)"
                    if not col.nullable:
                        col_info += " NOT NULL"
                    if col.foreign_key:
                        col_info += f" -> {col.foreign_key}"
                    columns.append(col_info)
                
                table_section += "\n" + "\n".join(columns)
                formatted_info.append(table_section)
            
            return "\n".join(formatted_info)
        except Exception as e:
            logger.error(f"Error formatting table info: {e}")
            return self.db.get_table_info()
    
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