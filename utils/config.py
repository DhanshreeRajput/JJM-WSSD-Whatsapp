import os
import re
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# OPTIMIZED Ollama Configuration for FAST WHATSAPP
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")  # Default to fast 8B model
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")

# WhatsApp Configuration
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID") 
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# SPEED OPTIMIZATION SETTINGS
REDIS_CACHE_ENABLED = os.getenv("REDIS_CACHE_ENABLED", "true").lower() == "true"
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))  # Fast interaction
FAST_MODE = os.getenv("FAST_MODE", "true").lower() == "true"
MAX_RESPONSE_TIME = int(os.getenv("MAX_RESPONSE_TIME", 45))  # 45 seconds max
CACHE_TTL = int(os.getenv("CACHE_TTL", 1800))  # 30 minutes

def get_whatsapp_optimized_config():
    """Get WhatsApp-optimized configuration for fast responses"""
    if "8b" in MODEL_NAME.lower():
        return {
            "max_tokens": 700,          # Shorter for mobile
            "temperature": 0.1,         # Consistent answers
            "chunk_size": 250,          # Small chunks for speed
            "max_chunks": 3,            # Fewer chunks for speed
            "timeout": 30,              # Short timeout for 8B
            "num_predict": 700,         # Limit generation
            "num_ctx": 1024,            # Smaller context window
            "num_thread": -1            # Use all CPU threads
        }
    else:
        return {
            "max_tokens": 1000,
            "temperature": 0.05,
            "chunk_size": 300,
            "max_chunks": 4,
            "timeout": 60,
            "num_predict": 1000,
            "num_ctx": 2048,
            "num_thread": -1
        }

def validate_ollama_config():
    """Validate Ollama configuration with speed checks"""
    try:
        print(f"🚀 FAST MODE: {FAST_MODE}")
        print(f"⚡ Ollama Base URL: {OLLAMA_BASE_URL}")
        print(f"🤖 Model: {MODEL_NAME}")
        print(f"📱 WhatsApp Optimized: {'8b' in MODEL_NAME.lower()}")
        
        # Test Ollama connection
        start_time = time.time()
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        connection_time = round((time.time() - start_time) * 1000, 2)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if MODEL_NAME in model_names:
                print(f"✅ Model '{MODEL_NAME}' is available ({connection_time}ms)")
                
                # Test generation speed
                test_start = time.time()
                test_payload = {
                    "model": MODEL_NAME,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 10}
                }
                
                gen_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate", 
                    json=test_payload, 
                    timeout=15
                )
                gen_time = round((time.time() - test_start) * 1000, 2)
                
                if gen_response.status_code == 200:
                    print(f"⚡ Generation test: {gen_time}ms")
                    if gen_time < 5000:  # Less than 5 seconds
                        print("🔥 EXCELLENT speed for WhatsApp!")
                    elif gen_time < 10000:  # Less than 10 seconds
                        print("✅ Good speed for WhatsApp")
                    else:
                        print("⚠️ May be slow for WhatsApp users")
                    
                    return True
                else:
                    print(f"❌ Generation test failed: {gen_response.status_code}")
                    return False
            else:
                print(f"❌ Model '{MODEL_NAME}' not found")
                print(f"📥 Available models: {model_names}")
                if "llama3.1:8b" not in model_names:
                    print(f"💡 Run: ollama pull llama3.1:8b")
                return False
        else:
            print(f"❌ Ollama server not responding: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def validate_whatsapp_config():
    """Validate WhatsApp configuration"""
    required_vars = [WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN]
    missing = [var for var in required_vars if not var]
    
    if missing:
        print(f"❌ Missing WhatsApp config: {len(missing)} variables")
        return False
    
    print("✅ WhatsApp configuration valid")
    return True

def validate_all_config():
    """Validate all configuration with speed focus"""
    print("🔧 Validating FAST WHATSAPP configuration...")
    
    ollama_valid = validate_ollama_config()
    wa_valid = validate_whatsapp_config()
    
    config = get_whatsapp_optimized_config()
    print(f"⚡ Speed settings: {config['timeout']}s timeout, {config['max_tokens']} tokens")
    
    if ollama_valid and wa_valid:
        print("🚀 ALL SYSTEMS READY FOR FAST WHATSAPP!")
        return True
    else:
        print("❌ Configuration issues detected")
        return False

def detect_language(text):
    """Fast language detection for WhatsApp"""
    try:
        # Quick character-based detection for speed
        hindi_chars = bool(re.search(r'[\u0900-\u097F]', text))
        english_chars = bool(re.search(r'[a-zA-Z]', text))

        if hindi_chars and not english_chars:
            # Quick keyword check for Marathi vs Hindi
            marathi_indicators = ['आहे', 'तुम्ही', 'मी', 'करा', 'येथे']
            if any(word in text for word in marathi_indicators):
                return 'marathi'
            return 'hindi'
        elif english_chars:
            return 'english'
        else:
            return 'english'  # Default for unknown
            
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'english'

# Import time for speed testing
import time

# Export optimized settings
__all__ = [
    'OLLAMA_BASE_URL', 'MODEL_NAME', 'AI_PROVIDER',
    'WHATSAPP_TOKEN', 'WHATSAPP_PHONE_NUMBER_ID', 'WHATSAPP_VERIFY_TOKEN',
    'REDIS_HOST', 'REDIS_PORT', 'REDIS_DB', 'REDIS_PASSWORD',
    'FAST_MODE', 'MAX_RESPONSE_TIME', 'CACHE_TTL', 'RATE_LIMIT_SECONDS',
    'get_whatsapp_optimized_config', 'validate_all_config', 'detect_language'
]