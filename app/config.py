"""Configuration management for SQL QA System - Fixed for Pydantic v2 and URL encoding."""

import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration (using your existing variables)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "wssd"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "root@123"
    DB_SCHEMA: str = "public"
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components with proper URL encoding."""
        import urllib.parse
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{encoded_password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Legacy database properties for backward compatibility
    @property
    def DB_HOST(self) -> str:
        return self.POSTGRES_HOST
    
    @property
    def DB_PORT(self) -> int:
        return self.POSTGRES_PORT
    
    @property
    def DB_NAME(self) -> str:
        return self.POSTGRES_DB
    
    @property
    def DB_USER(self) -> str:
        return self.POSTGRES_USER
    
    @property
    def DB_PASSWORD(self) -> str:
        return self.POSTGRES_PASSWORD
    
    # Ollama Configuration (using your existing variable)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_K: int = 10
    OLLAMA_TOP_P: float = 0.3
    OLLAMA_NUM_PREDICT: int = 2048
    
    # Organization Information (using your existing variables)
    ORG_NAME: str = "Water Supply and Sanitation Department"
    ORG_STATE: str = "Government of Maharashtra"
    WEBSITE_URL: str = "https://mahajalsamadhan.in"
    HELPLINE_NUMBERS: str = "104,102"
    
    @field_validator('HELPLINE_NUMBERS')
    @classmethod
    def parse_helpline_numbers(cls, v):
        """Parse comma-separated helpline numbers."""
        if not v:
            return []
        return [num.strip() for num in v.split(',') if num.strip()]
    
    # WhatsApp Business API Configuration (for future integration)
    WHATSAPP_TOKEN: Optional[str] = None
    WHATSAPP_PHONE_NUMBER_ID: Optional[str] = None
    WHATSAPP_VERIFY_TOKEN: Optional[str] = None
    
    # pgAdmin Configuration (for reference)
    PGADMIN_DEFAULT_EMAIL: str = "admin@example.com"
    PGADMIN_DEFAULT_PASSWORD: str = "admin"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG_MODE: bool = False
    SECRET_KEY: str = "wssd-sql-qa-secret-key-change-in-production"
    
    # Security Settings
    ENABLE_SAFETY_BY_DEFAULT: bool = True
    MAX_QUERY_RESULTS: int = 100
    QUERY_TIMEOUT_SECONDS: int = 30
    ALLOWED_TABLES: str = ""  # Comma-separated list
    
    @field_validator('ALLOWED_TABLES')
    @classmethod
    def parse_allowed_tables(cls, v):
        """Parse comma-separated table names."""
        if not v:
            return []
        return [table.strip() for table in v.split(',') if table.strip()]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/wssd_sql_qa.log"
    LOG_MAX_SIZE: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # CORS Configuration
    CORS_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True
    
    @field_validator('CORS_ORIGINS')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(',') if origin.strip()]
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }


# Global settings instance
settings = Settings()