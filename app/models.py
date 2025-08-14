"""Pydantic models for the SQL QA System."""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(..., description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    schema: str = Field("public", description="Database schema")


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")
    use_safety: bool = Field(True, description="Enable safety validation")
    limit_results: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question content."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class QueryResponse(BaseModel):
    """Response model for query results."""
    question: str = Field(..., description="Original question")
    sql_query: str = Field(..., description="Generated SQL query")
    result: Any = Field(..., description="Query execution result")
    interpretation: str = Field(..., description="Natural language interpretation")
    execution_time: float = Field(..., ge=0, description="Query execution time in seconds")
    is_safe: bool = Field(..., description="Safety validation result")
    validation_message: Optional[str] = Field(None, description="Validation details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    row_count: Optional[int] = Field(None, description="Number of rows returned")


class ColumnInfo(BaseModel):
    """Database column information."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column allows NULL values")
    primary_key: bool = Field(False, description="Whether column is primary key")
    foreign_key: Optional[str] = Field(None, description="Foreign key reference if applicable")
    default_value: Optional[str] = Field(None, description="Default value if any")


class TableInfo(BaseModel):
    """Database table information."""
    table_name: str = Field(..., description="Table name")
    schema_name: str = Field(..., description="Schema name")
    columns: List[ColumnInfo] = Field(..., description="Table columns")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    table_size: Optional[str] = Field(None, description="Table size")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall system status")
    database_status: str = Field(..., description="Database connection status")
    ollama_status: str = Field(..., description="Ollama LLM status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    version: str = Field("1.0.0", description="Application version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class QueryHistory(BaseModel):
    """Query history model."""
    id: int = Field(..., description="Query ID")
    question: str = Field(..., description="Original question")
    sql_query: str = Field(..., description="Generated SQL query")
    execution_time: float = Field(..., description="Execution time")
    success: bool = Field(..., description="Whether query was successful")
    timestamp: datetime = Field(..., description="Query timestamp")
    user_ip: Optional[str] = Field(None, description="User IP address")


class SystemStats(BaseModel):
    """System statistics model."""
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    average_execution_time: float = Field(..., description="Average query execution time")
    uptime: float = Field(..., description="System uptime in seconds")
    database_tables: int = Field(..., description="Number of database tables")
    last_query_time: Optional[datetime] = Field(None, description="Last query timestamp")


class BatchQuestionRequest(BaseModel):
    """Batch question request model."""
    questions: List[str] = Field(..., min_items=1, max_items=10, description="List of questions")
    use_safety: bool = Field(True, description="Enable safety validation")
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate each question in the batch."""
        validated = []
        for question in v:
            question = question.strip()
            if not question:
                raise ValueError("Questions cannot be empty")
            if len(question) > 1000:
                raise ValueError("Questions must be less than 1000 characters")
            validated.append(question)
        return validated


class BatchQueryResponse(BaseModel):
    """Batch query response model."""
    results: List[QueryResponse] = Field(..., description="Individual query results")
    total_execution_time: float = Field(..., description="Total batch execution time")
    successful_count: int = Field(..., description="Number of successful queries")
    failed_count: int = Field(..., description="Number of failed queries")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch completion timestamp")