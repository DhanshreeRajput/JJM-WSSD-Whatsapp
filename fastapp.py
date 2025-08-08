from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import os
import logging
import re
import requests
import PyPDF2
import hashlib
from contextlib import asynccontextmanager
import io

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Global variables
KNOWLEDGE_BASE = []
PROCESSED_MESSAGES = set()

# Simple cache
cache = {}

def detect_language(text: str) -> str:
    """Simple language detection"""
    text = text.strip().lower()
    
    # Hindi indicators
    hindi_words = ['ke baare', 'dijiye', 'jankari', 'kaise', 'batao', 'mujhe', 'chahiye', 'है', 'आप', 'मुझे', 'जानकारी']
    # Marathi indicators
    marathi_words = ['baddal', 'mahiti', 'dya', 'kay ahe', 'mhje', 'kasa', 'tumhi', 'mala', 'आहे', 'तुम्ही', 'माहिती']
    
    hindi_count = sum(1 for word in hindi_words if word in text)
    marathi_count = sum(1 for word in marathi_words if word in text)
    
    if hindi_count > 0:
        return 'hi'
    elif marathi_count > 0:
        return 'mr'
    else:
        return 'en'

def search_knowledge_base(query: str, language: str) -> str:
    """Simple direct search in knowledge base"""
    if not KNOWLEDGE_BASE:
        responses = {
            'en': "No knowledge base available. Please upload documents first.\n\n**For more details, please contact the 104/102 helpline numbers.**",
            'hi': "ज्ञान आधार उपलब्ध नहीं है। कृपया पहले दस्तावेज़ अपलोड करें।\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "ज्ञान आधार उपलब्ध नाही. कृपया प्रथम कागदपत्रे अपलोड करा.\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
        }
        return responses.get(language, responses['en'])
    
    # Search for query terms in knowledge base
    query_lower = query.lower()
    search_terms = ['jssk', 'cgis', 'esic', 'scheme', 'yojana', 'apply', 'application', 'eligibility']
    
    relevant_content = []
    for doc in KNOWLEDGE_BASE:
        content = doc.get('content', '').lower()
        if any(term in content for term in search_terms if term in query_lower):
            relevant_content.append(doc.get('content', ''))
    
    if not relevant_content:
        responses = {
            'en': "Information about this topic is not available in the knowledge base. Please contact 104/102 helpline for assistance.",
            'hi': "इस विषय की जानकारी ज्ञान आधार में उपलब्ध नहीं है। सहायता के लिए कृपया 104/102 हेल्पलाइन से संपर्क करें।",
            'mr': "या विषयाची माहिती ज्ञान आधारात उपलब्ध नाही. मदतीसाठी कृपया 104/102 हेल्पलाइन शी संपर्क साधा."
        }
        return responses.get(language, responses['en'])
    
    # Use Ollama to generate response from found content
    context = "\n".join(relevant_content[:2])  # Use top 2 relevant docs
    
    prompts = {
        'en': f"""Based ONLY on this context, answer the question in English:

Context: {context}

Question: {query}

Answer in English using only the information from the context above. If the context doesn't contain the specific information requested, say "Information not available in knowledge base."

Answer:""",
        
        'hi': f"""केवल इस संदर्भ के आधार पर, प्रश्न का उत्तर हिंदी में दें:

संदर्भ: {context}

प्रश्न: {query}

केवल ऊपर दिए गए संदर्भ की जानकारी का उपयोग करके हिंदी में उत्तर दें। यदि संदर्भ में विशिष्ट जानकारी नहीं है, तो कहें "ज्ञान आधार में जानकारी उपलब्ध नहीं है।"

उत्तर:""",
        
        'mr': f"""केवळ या संदर्भावर आधारित, प्रश्नाचे उत्तर मराठीत द्या:

संदर्भ: {context}

प्रश्न: {query}

फक्त वरील संदर्भातील माहिती वापरून मराठीत उत्तर द्या. जर संदर्भात विशिष्ट माहिती नाही, तर सांगा "ज्ञान आधारात माहिती उपलब्ध नाही."

उत्तर:"""
    }
    
    prompt = prompts.get(language, prompts['en'])
    
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 800,
                "num_ctx": 2048
            }
        }
        
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            
            if answer and len(answer) > 10:
                # Add helpline
                helplines = {
                    'en': "\n\n**For more details, please contact the 104/102 helpline numbers.**",
                    'hi': "\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                    'mr': "\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
                }
                return answer + helplines.get(language, helplines['en'])
            
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Fallback
    responses = {
        'en': "Unable to process your request. Please contact 104/102 helpline for assistance.",
        'hi': "आपका अनुरोध संसाधित नहीं कर सकते। सहायता के लिए कृपया 104/102 हेल्पलाइन से संपर्क करें।",
        'mr': "तुमची विनंती प्रक्रिया करू शकत नाही. मदतीसाठी कृपया 104/102 हेल्पलाइन शी संपर्क साधा."
    }
    return responses.get(language, responses['en'])

def get_response(query: str) -> str:
    """Main response function"""
    # Check cache
    cache_key = hashlib.md5(query.lower().encode()).hexdigest()
    if cache_key in cache:
        return cache[cache_key]
    
    # Detect language
    language = detect_language(query)
    print(f"Query: '{query}' | Language: {language}")
    
    # Handle greetings
    if any(word in query.lower() for word in ['hi', 'hello', 'namaste', 'नमस्ते', 'नमस्कार']):
        responses = {
            'en': "Hello! I can help with government schemes. What would you like to know?\n\n**For more details, please contact the 104/102 helpline numbers.**",
            'hi': "नमस्ते! मैं सरकारी योजनाओं की जानकारी दे सकती हूं। आप क्या जानना चाहते हैं?\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
            'mr': "नमस्कार! मी सरकारी योजनांची माहिती देऊ शकते. तुम्हाला काय जाणून घ्यायचे आहे?\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
        }
        response = responses.get(language, responses['en'])
        cache[cache_key] = response
        return response
    
    # Search knowledge base
    response = search_knowledge_base(query, language)
    cache[cache_key] = response
    return response

def send_whatsapp_message(to: str, message: str) -> bool:
    """Send WhatsApp message"""
    try:
        headers = {'Authorization': f'Bearer {WHATSAPP_TOKEN}', 'Content-Type': 'application/json'}
        payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": message}}
        url = f"https://graph.facebook.com/v12.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        return response.status_code == 200
    except:
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Simple WhatsApp Bot...")
    yield
    print("🛑 Shutting down...")

app = FastAPI(title="Simple WhatsApp Bot", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    input_text: str

@app.get("/")
async def root():
    return {
        "message": "Simple WhatsApp Bot",
        "model": MODEL_NAME,
        "knowledge_base_size": len(KNOWLEDGE_BASE)
    }

@app.post("/upload/")
async def upload_files(pdf_file: Optional[UploadFile] = File(None), txt_file: Optional[UploadFile] = File(None)):
    """Upload files to knowledge base"""
    global KNOWLEDGE_BASE
    
    if not pdf_file and not txt_file:
        raise HTTPException(status_code=400, detail="Upload at least one file")

    # Process PDF
    if pdf_file:
        pdf_bytes = await pdf_file.read()
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            KNOWLEDGE_BASE.append({"filename": pdf_file.filename, "content": text})
            print(f"✅ PDF processed: {pdf_file.filename}")
        except Exception as e:
            print(f"❌ PDF error: {e}")

    # Process TXT
    if txt_file:
        txt_bytes = await txt_file.read()
        try:
            text = txt_bytes.decode('utf-8')
            KNOWLEDGE_BASE.append({"filename": txt_file.filename, "content": text})
            print(f"✅ TXT processed: {txt_file.filename}")
        except Exception as e:
            print(f"❌ TXT error: {e}")

    return {"message": "Files uploaded", "knowledge_base_size": len(KNOWLEDGE_BASE)}

@app.post("/query/")
async def query_endpoint(req: QueryRequest):
    """Query endpoint"""
    response = get_response(req.input_text)
    return {"query": req.input_text, "response": response, "language": detect_language(req.input_text)}

@app.get("/webhook")
async def verify_whatsapp(request: Request):
    """WhatsApp verification"""
    params = dict(request.query_params)
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
        return int(params.get("hub.challenge"))
    return JSONResponse(status_code=403, content={"error": "Verification failed"})

@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Handle WhatsApp messages"""
    global PROCESSED_MESSAGES
    
    try:
        data = await request.json()
        
        if not data.get("entry") or not data["entry"][0].get("changes"):
            return {"status": "ok"}
        
        messages = data["entry"][0]["changes"][0].get("value", {}).get("messages")
        if not messages:
            return {"status": "ok"}
        
        message_obj = messages[0]
        if message_obj.get("type") != "text":
            return {"status": "ok"}
        
        user_number = message_obj.get("from")
        message_id = message_obj.get("id")
        user_msg = message_obj.get("text", {}).get("body", "").strip()
        
        # Skip duplicates and own messages
        if message_id in PROCESSED_MESSAGES or user_number == WHATSAPP_PHONE_NUMBER_ID:
            return {"status": "duplicate"}
        
        if user_msg:
            print(f"📝 Processing: '{user_msg}'")
            PROCESSED_MESSAGES.add(message_id)
            
            # Keep only recent messages
            if len(PROCESSED_MESSAGES) > 100:
                PROCESSED_MESSAGES = set(list(PROCESSED_MESSAGES)[-50:])
            
            # Generate and send response
            response = get_response(user_msg)
            success = send_whatsapp_message(user_number, response)
            
            return {"status": "processed", "send_success": success}
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/test/{query}")
async def test_query(query: str):
    """Test any query"""
    language = detect_language(query)
    response = get_response(query)
    return {
        "query": query,
        "detected_language": language,
        "response": response,
        "knowledge_base_size": len(KNOWLEDGE_BASE)
    }

@app.get("/debug/kb")
async def debug_knowledge_base():
    """Debug knowledge base"""
    return {
        "total_documents": len(KNOWLEDGE_BASE),
        "documents": [
            {
                "filename": doc.get("filename", "unknown"),
                "content_length": len(doc.get("content", "")),
                "preview": doc.get("content", "")[:200] + "..."
            }
            for doc in KNOWLEDGE_BASE
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple WhatsApp Bot...")
    uvicorn.run(app, host="0.0.0.0", port=8080)