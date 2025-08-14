#!/usr/bin/env python3
"""
Application runner for the SQL QA System.
This script provides a simple way to start the application with proper configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

def setup_environment():
    """Set up environment variables if .env file exists."""
    env_file = app_dir / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
        print("   Please copy .env.example to .env and configure your settings")

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2', 
        'langchain', 'langchain_community', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        import requests
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✅ Ollama is running at {ollama_url}")
            
            # Check if the required model is available
            models = response.json().get('models', [])
            required_model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
            
            model_names = [model['name'] for model in models]
            if required_model in model_names:
                print(f"✅ Model {required_model} is available")
            else:
                print(f"⚠️  Model {required_model} not found")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"   Run: ollama pull {required_model}")
            
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False
    except ImportError:
        print("⚠️  requests package not available for Ollama check")
    
    return True

def main():
    """Main application runner."""
    print("🚀 Starting SQL Question-Answering System")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check Ollama connection
    check_ollama_connection()
    
    # Import and run the application
    try:
        import uvicorn
        from app.config import settings
        
        print(f"\n🌐 Starting server...")
        print(f"   Host: {settings.API_HOST}")
        print(f"   Port: {settings.API_PORT}")
        print(f"   Debug: {settings.DEBUG_MODE}")
        print(f"   Log Level: {settings.LOG_LEVEL}")
        
        print(f"\n📱 Web Interface:")
        print(f"   http://localhost:{settings.API_PORT}/")
        print(f"\n📚 API Documentation:")
        print(f"   http://localhost:{settings.API_PORT}/docs")
        print(f"   http://localhost:{settings.API_PORT}/redoc")
        
        print("\n🔧 Configuration:")
        if settings.DB_NAME:
            print(f"   Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
        else:
            print("   Database: Not configured (manual setup required)")
        print(f"   Ollama: {settings.OLLAMA_BASE_URL} ({settings.OLLAMA_MODEL})")
        
        print("\n" + "=" * 50)
        
        # Start the server
        uvicorn.run(
            "app.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG_MODE,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()