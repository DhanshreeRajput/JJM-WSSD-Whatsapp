"""
INTEGRATED WHATSAPP AI BOT WITH POSTGRESQL AND OLLAMA
Combines SQL querying capabilities with WhatsApp messaging
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncpg
import requests
import os
import logging
from dotenv import load_dotenv
import re
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState

# Load environment variables
load_dotenv()

# Configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "root@123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "wssd")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID") 
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WhatsApp AI Bot with SQL Capabilities")

# Global variables
processed_messages = set()

class WhatsAppAIBot:
    def __init__(self):
        self.db_pool = None
        self.sql_agent = None
        
    async def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL)
            logger.info("✅ Connected to PostgreSQL database")
            
            # Initialize SQL agent with Ollama
            self.initialize_sql_agent()
            logger.info("✅ SQL Agent initialized with Ollama")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def initialize_sql_agent(self):
        """Initialize the SQL agent with Ollama"""
        try:
            # Initialize Ollama LLM
            llm = ChatOllama(
                model="llama3.1:8b",
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )
            
            # Initialize database connection for SQL toolkit
            db = SQLDatabase.from_uri(DATABASE_URL)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            tools = toolkit.get_tools()
            
            system_message = SystemMessage(content=self.get_sql_system_prompt())
            
            self.sql_agent = create_react_agent(
                model=llm,
                tools=tools,
                messages_modifier=system_message
            )
            
        except Exception as e:
            logger.error(f"❌ SQL Agent initialization failed: {e}")
            raise
    
    def get_sql_system_prompt(self):
        """Generate system prompt for SQL agent"""
        return """
        You are a helpful AI assistant specializing in government schemes and services data analysis.
        You have access to a PostgreSQL database with information about districts, schemes, and services.
        
        When users ask questions:
        1. If it's a greeting, respond warmly in their language
        2. If it's a data query, use SQL to find the answer
        3. Always provide helpful, accurate information
        4. Support Hindi, Marathi, and English languages
        
        For SQL queries:
        - Only use existing tables and columns
        - Never make assumptions about column names
        - Provide clear, concise answers
        - If data isn't found, suggest alternatives or helplines
        
        Always end responses with: "📞 For more help: Call 104/102 helpline"
        
        Keep responses under 300 words for WhatsApp compatibility.
        """
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection"""
        text = text.lower()
        
        # Hindi patterns
        hindi_patterns = [
            'योजना', 'सरकार', 'आवेदन', 'जानकारी', 'सहायता', 'लाभ',
            'kaise', 'kya', 'yojana', 'sarkar', 'scheme'
        ]
        
        # Marathi patterns
        marathi_patterns = [
            'योजना', 'शासन', 'अर्ज', 'माहिती', 'मदत', 'लाभ',
            'kasa', 'kuthe', 'yojana', 'shasan'
        ]
        
        if any(word in text for word in hindi_patterns):
            return 'hi'
        elif any(word in text for word in marathi_patterns):
            return 'mr'
        else:
            return 'en'
    
    def is_greeting(self, text: str) -> bool:
        """Check if message is a greeting"""
        greetings = [
            'hi', 'hello', 'hey', 'namaste', 'namaskar',
            'नमस्ते', 'नमस्कार', 'हैलो', 'हाय'
        ]
        return any(greeting in text.lower() for greeting in greetings)
    
    def is_data_query(self, text: str) -> bool:
        """Check if message requires database query"""
        query_keywords = [
            'show', 'list', 'find', 'search', 'how many', 'count',
            'what', 'which', 'where', 'when', 'who',
            'scheme', 'district', 'beneficiary', 'application',
            'योजना', 'जिला', 'लाभार्थी', 'आवेदन',
            'दिखाओ', 'बताओ', 'खोजो', 'कितने'
        ]
        return any(keyword in text.lower() for keyword in query_keywords)
    
    def get_greeting_response(self, language: str) -> str:
        """Get greeting response based on language"""
        greetings = {
            'en': """Hello! 👋 I'm your AI assistant for government schemes and services. 

I can help you with:
• Information about government schemes
• District-wise data
• Eligibility criteria
• Application processes

Just ask me anything!""",
            
            'hi': """नमस्ते! 👋 मैं सरकारी योजनाओं की AI सहायिका हूं।

मैं आपकी मदद कर सकती हूं:
• सरकारी योजनाओं की जानकारी
• जिलेवार डेटा
• पात्रता मापदंड
• आवेदन प्रक्रिया

कुछ भी पूछिए!""",
            
            'mr': """नमस्कार! 👋 मी सरकारी योजनांची AI सहाय्यक आहे।

मी तुमची मदत करू शकते:
• सरकारी योजनांची माहिती
• जिल्हानिहाय डेटा
• पात्रता निकष
• अर्ज प्रक्रिया

काहीही विचारा!"""
        }
        
        response = greetings.get(language, greetings['en'])
        return response + "\n\n📞 For more help: Call 104/102 helpline"
    
    async def query_database_simple(self, query: str, language: str) -> str:
        """Simple database search for non-complex queries"""
        try:
            async with self.db_pool.acquire() as conn:
                # Generic search across common tables
                search_query = """
                SELECT * FROM (
                    SELECT 'scheme' as type, scheme_name as title, description, details as content
                    FROM schemes 
                    WHERE scheme_name ILIKE $1 OR description ILIKE $1
                    
                    UNION ALL
                    
                    SELECT 'district' as type, district_name as title, '' as description, 
                           CONCAT('Population: ', population, ', Area: ', area) as content
                    FROM districts 
                    WHERE district_name ILIKE $1
                ) results
                LIMIT 5
                """
                
                search_pattern = f"%{query}%"
                results = await conn.fetch(search_query, search_pattern)
                
                if results:
                    response_parts = []
                    for row in results:
                        title = row.get('title', '')
                        content = row.get('content', '') or row.get('description', '')
                        if title and content:
                            response_parts.append(f"• {title}: {content[:200]}")
                    
                    if response_parts:
                        response = "\n\n".join(response_parts[:3])
                        
                        helplines = {
                            'en': "\n\n📞 For more help: Call 104/102 helpline",
                            'hi': "\n\n📞 अधिक सहायता के लिए: 104/102 हेल्पलाइन पर कॉल करें",
                            'mr': "\n\n📞 अधिक मदतीसाठी: 104/102 हेल्पलाइन वर कॉल करा"
                        }
                        
                        return response + helplines.get(language, helplines['en'])
                
                # No results found
                no_results = {
                    'en': "Sorry, I couldn't find specific information. Please try rephrasing your question or contact 104/102 helpline.",
                    'hi': "क्षमा करें, मुझे विशिष्ट जानकारी नहीं मिली। कृपया प्रश्न दोबारा पूछें या 104/102 हेल्पलाइन से संपर्क करें।",
                    'mr': "क्षमा करा, मला विशिष्ट माहिती सापडली नाही. कृपया प्रश्न पुन्हा विचारा किंवा 104/102 हेल्पलाइन शी संपर्क करा."
                }
                return no_results.get(language, no_results['en'])
                
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return "Technical issue occurred. Please try again or contact 104/102 helpline."
    
    async def query_with_sql_agent(self, query: str, language: str) -> str:
        """Use SQL agent for complex queries"""
        try:
            if not self.sql_agent:
                return await self.query_database_simple(query, language)
            
            # Execute query with SQL agent
            graph_config = {"configurable": {"thread_id": "1"}}
            result = self.sql_agent.invoke({"messages": query}, config=graph_config)
            
            response = result["messages"][-1].content
            
            # Ensure response isn't too long for WhatsApp
            if len(response) > 1000:
                response = response[:900] + "..."
            
            # Add helpline
            helplines = {
                'en': "\n\n📞 For more help: Call 104/102 helpline",
                'hi': "\n\n📞 अधिक सहायता के लिए: 104/102 हेल्पलाइन पर कॉल करें",
                'mr': "\n\n📞 अधिक मदतीसाठी: 104/102 हेल्पलाइन वर कॉल करा"
            }
            
            return response + helplines.get(language, helplines['en'])
            
        except Exception as e:
            logger.error(f"SQL Agent error: {e}")
            return await self.query_database_simple(query, language)
    
    async def generate_response(self, user_message: str) -> str:
        """Generate AI response based on message type"""
        language = self.detect_language(user_message)
        
        # Handle greetings
        if self.is_greeting(user_message):
            return self.get_greeting_response(language)
        
        # Handle data queries with SQL agent for complex queries
        if self.is_data_query(user_message) and len(user_message.split()) > 5:
            return await self.query_with_sql_agent(user_message, language)
        
        # Handle simple queries with direct database search
        return await self.query_database_simple(user_message, language)
    
    def send_whatsapp_message(self, phone_number: str, message: str) -> bool:
        """Send message via WhatsApp Business API"""
        try:
            url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            
            headers = {
                'Authorization': f'Bearer {WHATSAPP_TOKEN}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "messaging_product": "whatsapp",
                "to": phone_number,
                "type": "text",
                "text": {"body": message}
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return False

# Initialize bot
bot = WhatsAppAIBot()

@app.on_event("startup")
async def startup():
    """Initialize database connection and SQL agent on startup"""
    await bot.connect_database()
    logger.info("🚀 WhatsApp AI Bot with SQL capabilities started successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "message": "WhatsApp AI Bot with SQL capabilities is running",
        "database": "connected" if bot.db_pool else "disconnected",
        "sql_agent": "active" if bot.sql_agent else "inactive"
    }

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verify WhatsApp webhook"""
    params = dict(request.query_params)
    
    if (params.get("hub.mode") == "subscribe" and 
        params.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN):
        return int(params.get("hub.challenge"))
    
    return JSONResponse(status_code=403, content={"error": "Verification failed"})

@app.post("/webhook")
async def handle_whatsapp_message(request: Request):
    """Handle incoming WhatsApp messages"""
    global processed_messages
    
    try:
        data = await request.json()
        
        # Extract message data
        if not data.get("entry"):
            return {"status": "no_entry"}
        
        changes = data["entry"][0].get("changes", [])
        if not changes:
            return {"status": "no_changes"}
        
        messages = changes[0].get("value", {}).get("messages", [])
        if not messages:
            return {"status": "no_messages"}
        
        message = messages[0]
        
        # Only process text messages
        if message.get("type") != "text":
            return {"status": "not_text"}
        
        # Extract message details
        phone_number = message.get("from")
        message_id = message.get("id")
        text = message.get("text", {}).get("body", "").strip()
        
        # Avoid processing duplicate messages
        if message_id in processed_messages:
            return {"status": "duplicate"}
        
        processed_messages.add(message_id)
        
        # Keep memory usage low
        if len(processed_messages) > 1000:
            processed_messages = set(list(processed_messages)[-500:])
        
        if text and phone_number:
            logger.info(f"📨 Received: '{text}' from {phone_number}")
            
            # Generate AI response
            ai_response = await bot.generate_response(text)
            
            # Send response
            success = bot.send_whatsapp_message(phone_number, ai_response)
            
            if success:
                logger.info(f"✅ Sent response to {phone_number}")
                return {"status": "success", "message_sent": True}
            else:
                logger.error(f"❌ Failed to send response to {phone_number}")
                return {"status": "send_failed"}
        
        return {"status": "processed"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/test")
async def test_bot(request: Request):
    """Test the bot with a sample message"""
    try:
        data = await request.json()
        test_message = data.get("message", "Hello")
        
        response = await bot.generate_response(test_message)
        
        return {
            "test_message": test_message,
            "bot_response": response,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting WhatsApp AI Bot with SQL capabilities...")
    uvicorn.run(app, host="0.0.0.0", port=8000)