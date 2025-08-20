"""Enhanced database manager for LangGraph system."""

import asyncio
import logging
from typing import List, Any, Dict, Optional
from datetime import datetime

from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities import SQLDatabase  # Fixed import

from app.config import settings
from app.models import TableInfo, ColumnInfo

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced database manager for LangGraph system."""
    
    def __init__(self, database_uri: str):
        """Initialize database manager."""
        self.database_uri = database_uri
        self.db = None
        self.engine = None
        self.metadata = None
        self._table_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection."""
        try:
            self.db = SQLDatabase.from_uri(self.database_uri)
            self.engine = self.db._engine
            self.metadata = MetaData()
            
            # Test connection
            self.db.run("SELECT 1")
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    async def execute_query(self, sql_query: str) -> Any:
        """Execute SQL query asynchronously."""
        try:
            result = await asyncio.to_thread(self.db.run, sql_query)
            return result
        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {e}")
            raise
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    def get_table_info(self, use_cache: bool = True) -> List[TableInfo]:
        """Get detailed information about all database tables with caching."""
        current_time = datetime.now()
        
        # Check cache validity
        if (use_cache and self._table_cache and self._cache_timestamp and 
            (current_time - self._cache_timestamp).seconds < self._cache_ttl):
            logger.debug("Using cached table information")
            return list(self._table_cache.values())
        
        try:
            inspector = inspect(self.engine)
            tables_info = []
            
            schema_name = settings.DB_SCHEMA
            table_names = inspector.get_table_names(schema=schema_name)
            
            # Filter tables if ALLOWED_TABLES is configured
            if settings.ALLOWED_TABLES:
                table_names = [t for t in table_names if t in settings.ALLOWED_TABLES]
            
            for table_name in table_names:
                try:
                    table_info = self._get_single_table_info(inspector, table_name, schema_name)
                    if table_info:
                        tables_info.append(table_info)
                        self._table_cache[table_name] = table_info
                
                except Exception as e:
                    logger.warning(f"Error getting info for table {table_name}: {e}")
                    continue
            
            # Update cache timestamp
            self._cache_timestamp = current_time
            
            logger.info(f"Retrieved information for {len(tables_info)} tables")
            return tables_info
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return []
    
    def _get_single_table_info(self, inspector, table_name: str, schema_name: str) -> Optional[TableInfo]:
        """Get information for a single table."""
        try:
            # Get column information
            columns = []
            db_columns = inspector.get_columns(table_name, schema=schema_name)
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)
            primary_keys = inspector.get_pk_constraint(table_name, schema=schema_name)
            
            # Create foreign key mapping
            fk_mapping = {}
            for fk in foreign_keys:
                for local_col, referred_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fk_mapping[local_col] = f"{fk['referred_table']}.{referred_col}"
            
            # Get primary key columns
            pk_columns = primary_keys.get("constrained_columns", []) if primary_keys else []
            
            for column in db_columns:
                columns.append(ColumnInfo(
                    name=column["name"],
                    type=str(column["type"]),
                    nullable=column.get("nullable", True),
                    primary_key=column["name"] in pk_columns,
                    foreign_key=fk_mapping.get(column["name"]),
                    default_value=str(column.get("default")) if column.get("default") else None
                ))
            
            # Get approximate row count
            row_count = self._get_table_row_count(table_name)
            
            return TableInfo(
                table_name=table_name,
                schema_name=schema_name,
                columns=columns,
                row_count=row_count
            )
            
        except Exception as e:
            logger.error(f"Error getting info for table {table_name}: {e}")
            return None
    
    def _get_table_row_count(self, table_name: str) -> Optional[int]:
        """Get approximate row count for a table."""
        try:
            row_count_result = self.db.run(
                f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{table_name}'"
            )
            if row_count_result and row_count_result[0]:
                return int(row_count_result[0][0]) if row_count_result[0][0] is not None else 0
            return None
        except Exception:
            return None
    
    def get_tables_for_agent(self, agent_name: str) -> List[TableInfo]:
        """Get tables relevant to a specific agent."""
        from app.utils.table_mappings import get_tables_for_agent
        
        all_tables = self.get_table_info()
        relevant_table_names = get_tables_for_agent(agent_name)
        
        return [table for table in all_tables if table.table_name in relevant_table_names]
    
    def get_table_schema_summary(self, table_names: List[str] = None) -> str:
        """Get a formatted summary of table schemas."""
        tables = self.get_table_info()
        
        if table_names:
            tables = [table for table in tables if table.table_name in table_names]
        
        schema_summary = []
        for table in tables:
            table_desc = f"Table: {table.table_name}"
            if table.row_count is not None:
                table_desc += f" ({table.row_count:,} rows)"
            
            columns = []
            for col in table.columns:
                col_desc = f"  - {col.name}: {col.type}"
                if col.primary_key:
                    col_desc += " (PK)"
                if col.foreign_key:
                    col_desc += f" -> {col.foreign_key}"
                if not col.nullable:
                    col_desc += " NOT NULL"
                columns.append(col_desc)
            
            schema_summary.append(table_desc + "\n" + "\n".join(columns))
        
        return "\n\n".join(schema_summary)
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            await self.execute_query("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear the table information cache."""
        self._table_cache.clear()
        self._cache_timestamp = None
        logger.info("Table cache cleared")