"""Configuration management for SQL QA System."""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_SCHEMA: str = "public"
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_K: int = 10
    OLLAMA_TOP_P: float = 0.3
    OLLAMA_NUM_PREDICT: int = 2048
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG_MODE: bool = False
    SECRET_KEY: str = "change-me-in-production"
    
    # Security Settings
    ENABLE_SAFETY_BY_DEFAULT: bool = True
    MAX_QUERY_RESULTS: int = 100
    QUERY_TIMEOUT_SECONDS: int = 30
    ALLOWED_TABLES: str = ""  # Comma-separated list
    
    @validator('ALLOWED_TABLES')
    def parse_allowed_tables(cls, v):
        """Parse comma-separated table names."""
        if not v:
            return []
        return [table.strip() for table in v.split(',') if table.strip()]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/sql_qa.log"
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # CORS Configuration
    CORS_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    @validator('CORS_ORIGINS')
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(',') if origin.strip()]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()