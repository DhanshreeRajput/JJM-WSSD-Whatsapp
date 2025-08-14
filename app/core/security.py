"""Security validation for SQL queries."""

import re
import logging
from typing import Tuple, List
from app.config import settings

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validates SQL queries for security and safety."""
    
    def __init__(self):
        """Initialize security validator with forbidden patterns."""
        
        # Forbidden SQL keywords that could modify data or structure
        self.forbidden_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE', 'REPLACE',
            'MERGE', 'CALL', 'ANALYZE', 'VACUUM', 'REINDEX', 'COPY',
            'IMPORT', 'EXPORT', 'BACKUP', 'RESTORE', 'SHUTDOWN'
        ]
        
        # Dangerous functions that could be exploited
        self.dangerous_functions = [
            'pg_read_file', 'pg_write_file', 'pg_ls_dir', 'pg_sleep',
            'dblink', 'postgres_fdw', 'lo_import', 'lo_export',
            'pg_execute', 'pg_eval'
        ]
        
        # Patterns that might indicate SQL injection attempts
        self.injection_patterns = [
            r";\s*(DROP|DELETE|INSERT|UPDATE)",  # Multiple statements
            r"UNION\s+SELECT.*--",               # Union-based injection
            r"'\s*OR\s+'.*'='",                 # Always true conditions
            r"--\s*$",                          # Comment at end (suspicious)
            r"/\*.*\*/",                        # Block comments
            r"xp_",                             # SQL Server extended procedures
            r"sp_",                             # SQL Server system procedures
        ]
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for security and safety.
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            Tuple of (is_safe: bool, message: str)
        """
        try:
            sql_upper = sql_query.upper().strip()
            
            # Check if query is empty or invalid
            if not sql_query.strip():
                return False, "Empty query provided"
            
            if sql_query.strip() == "NO_QUERY_POSSIBLE":
                return False, "Cannot generate appropriate query for this question"
            
            # 1. Check if it's a SELECT query
            if not sql_upper.startswith('SELECT'):
                return False, "Only SELECT queries are allowed"
            
            # 2. Check for forbidden keywords
            for keyword in self.forbidden_keywords:
                if re.search(rf'\b{keyword}\b', sql_upper):
                    return False, f"Forbidden keyword detected: {keyword}"
            
            # 3. Check for dangerous functions
            for func in self.dangerous_functions:
                if func.upper() in sql_upper:
                    return False, f"Dangerous function detected: {func}"
            
            # 4. Check for SQL injection patterns
            for pattern in self.injection_patterns:
                if re.search(pattern, sql_query, re.IGNORECASE):
                    return False, "Potential SQL injection pattern detected"
            
            # 5. Check for multiple statements (semicolon check)
            statements = [stmt.strip() for stmt in sql_query.split(';') if stmt.strip()]
            if len(statements) > 1:
                return False, "Multiple statements detected - only single SELECT allowed"
            
            # 6. Validate parentheses balance
            if sql_query.count('(') != sql_query.count(')'):
                return False, "Unbalanced parentheses in query"
            
            # 7. Check quote balance
            single_quotes = sql_query.count("'")
            if single_quotes % 2 != 0:
                return False, "Unbalanced single quotes in query"
            
            # 8. Check for table restrictions
            if settings.ALLOWED_TABLES:
                if not self._validate_table_access(sql_query):
                    return False, "Query accesses unauthorized tables"
            
            # 9. Check query complexity (prevent extremely complex queries)
            if self._is_query_too_complex(sql_query):
                return False, "Query is too complex - please simplify"
            
            # 10. Validate LIMIT clause exists
            if not re.search(r'\bLIMIT\s+\d+', sql_upper):
                logger.warning("Query without LIMIT clause detected")
            
            return True, "Query passed all security validations"
            
        except Exception as e:
            logger.error(f"Error during security validation: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_table_access(self, sql_query: str) -> bool:
        """Check if query only accesses allowed tables."""
        try:
            # Extract table names from FROM and JOIN clauses
            tables_in_query = self._extract_table_names(sql_query)
            
            # Check if all tables are in allowed list
            for table in tables_in_query:
                if table not in settings.ALLOWED_TABLES:
                    logger.warning(f"Unauthorized table access attempt: {table}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating table access: {e}")
            return False
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        
        # Patterns to match table names in FROM and JOIN clauses
        patterns = [
            r'\bFROM\s+(\w+)',
            r'\bJOIN\s+(\w+)',
            r'\bINNER\s+JOIN\s+(\w+)',
            r'\bLEFT\s+JOIN\s+(\w+)',
            r'\bRIGHT\s+JOIN\s+(\w+)',
            r'\bFULL\s+JOIN\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_query, re.IGNORECASE)
            tables.extend(matches)
        
        # Remove duplicates and schema prefixes
        unique_tables = []
        for table in tables:
            # Remove schema prefix if present
            table_name = table.split('.')[-1]
            if table_name not in unique_tables:
                unique_tables.append(table_name)
        
        return unique_tables
    
    def _is_query_too_complex(self, sql_query: str) -> bool:
        """Check if query is too complex."""
        # Count various complexity indicators
        complexity_score = 0
        
        # Count subqueries
        complexity_score += sql_query.upper().count('SELECT') - 1  # Subtract main SELECT
        
        # Count JOINs
        complexity_score += len(re.findall(r'\bJOIN\b', sql_query, re.IGNORECASE))
        
        # Count functions
        complexity_score += len(re.findall(r'\w+\s*\(', sql_query))
        
        # Count CASE statements
        complexity_score += sql_query.upper().count('CASE')
        
        # Count window functions
        complexity_score += sql_query.upper().count('OVER')
        
        # Set complexity threshold
        MAX_COMPLEXITY = 15
        
        if complexity_score > MAX_COMPLEXITY:
            logger.warning(f"High complexity query detected (score: {complexity_score})")
            return True
        
        return False
    
    def sanitize_query_for_logging(self, sql_query: str) -> str:
        """Sanitize SQL query for safe logging (remove sensitive data)."""
        try:
            # Replace string literals with placeholders
            sanitized = re.sub(r"'[^']*'", "'***'", sql_query)
            
            # Replace numbers that might be sensitive
            sanitized = re.sub(r'\b\d{4,}\b', '***', sanitized)
            
            return sanitized
        except Exception:
            return "Error sanitizing query for logging"


# Global security validator instance
security_validator = SecurityValidator()