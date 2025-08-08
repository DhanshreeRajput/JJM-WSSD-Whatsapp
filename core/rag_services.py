import re
import time
import hashlib
import requests
from typing import Optional, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.retrievers import TFIDFRetriever
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Simple cache
_query_cache = {}

class SimpleLLM(LLM):
    """Simple Ollama LLM"""
    
    model_name: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_url = base_url
    
    @property
    def _llm_type(self) -> str:
        return "simple_ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Call Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1000,
                    "num_ctx": 2048
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return "Technical issue. Please try again."
                
        except Exception as e:
            print(f"LLM error: {e}")
            return "Service unavailable. Please try again."

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

def get_prompt_template():
    """Simple but effective prompt template"""
    template = """You are a government schemes assistant. Use ONLY the provided context to answer.

LANGUAGE RULES:
- If question is in English → Answer in English
- If question has Hindi words (ke baare, dijiye, jankari) → Answer in Hindi Devanagari (हिंदी)
- If question has Marathi words (baddal, mahiti, dya) → Answer in Marathi Devanagari (मराठी)

RESPONSE RULES:
- Use ONLY information from the context below
- If context doesn't have the information, say "Information not available in knowledge base"
- Give detailed answers when information is found
- Include eligibility, benefits, application process if mentioned

Context:
{context}

Question: {question}

Answer (in same language as question, using only context information):"""

    return PromptTemplate(template=template, input_variables=["context", "question"])

def build_rag_chain_from_documents(documents, ollama_model="llama3.1:8b", **kwargs):
    """Build basic RAG chain"""
    print(f"Building RAG chain with {ollama_model}")
    
    if not documents:
        raise ValueError("No documents provided")
    
    # Convert to LangChain documents
    langchain_docs = []
    for doc in documents:
        content = doc.get("content", "").strip()
        if content and len(content) > 50:
            langchain_docs.append(Document(
                page_content=content,
                metadata={"filename": doc.get("filename", "unknown")}
            ))
    
    if not langchain_docs:
        raise ValueError("No valid documents")
    
    print(f"Processing {len(langchain_docs)} documents")
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    splits = splitter.split_documents(langchain_docs)
    print(f"Created {len(splits)} chunks")
    
    # Create retriever
    retriever = TFIDFRetriever.from_documents(splits, k=5)
    
    # Create LLM
    llm = SimpleLLM(model_name=ollama_model)
    
    # Create prompt
    prompt = get_prompt_template()
    
    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ Basic RAG chain ready")
    return chain

def process_scheme_query_with_retry(rag_chain, user_query, max_retries=1, **kwargs):
    """Process query with basic RAG"""
    
    # Check cache
    query_hash = hashlib.md5(user_query.lower().encode()).hexdigest()
    if query_hash in _query_cache:
        print("Cache hit")
        return _query_cache[query_hash], "", detect_language(user_query), {}
    
    language = detect_language(user_query)
    print(f"Processing '{user_query}' in language: {language}")
    
    try:
        # Process with RAG
        result = rag_chain.invoke({"query": user_query})
        
        if isinstance(result, dict):
            response_text = result.get('result', '')
        else:
            response_text = str(result)
        
        if not response_text or len(response_text.strip()) < 20:
            # No info found responses
            no_info_responses = {
                'hi': "इस विषय की जानकारी ज्ञान आधार में उपलब्ध नहीं है। सहायता के लिए कृपया 104/102 हेल्पलाइन से संपर्क करें।",
                'mr': "या विषयाची माहिती ज्ञान आधारात उपलब्ध नाही. मदतीसाठी कृपया 104/102 हेल्पलाइन शी संपर्क साधा.",
                'en': "Information about this topic is not available in the knowledge base. Please contact 104/102 helpline for assistance."
            }
            response_text = no_info_responses.get(language, no_info_responses['en'])
        else:
            # Add helpline to valid responses
            helplines = {
                'en': "\n\n**For more details, please contact the 104/102 helpline numbers.**",
                'hi': "\n\n**अधिक जानकारी के लिए कृपया 104/102 हेल्पलाइन नंबर पर संपर्क करें।**",
                'mr': "\n\n**अधिक माहितीसाठी, कृपया 104/102 हेल्पलाइन क्रमांकावर संपर्क साधा.**"
            }
            # Remove existing helpline
            for ending in helplines.values():
                response_text = response_text.replace(ending.strip(), "")
            # Add correct helpline
            response_text = response_text.strip() + helplines.get(language, helplines['en'])
        
        # Cache result
        _query_cache[query_hash] = response_text
        
        # Keep cache manageable
        if len(_query_cache) > 50:
            old_keys = list(_query_cache.keys())[:10]
            for key in old_keys:
                del _query_cache[key]
        
        return response_text, "", language, {}
        
    except Exception as e:
        print(f"RAG error: {e}")
        error_responses = {
            'hi': "तकनीकी समस्या हुई। कृपया पुनः प्रयास करें।",
            'mr': "तांत्रिक समस्या झाली. कृपया पुन्हा प्रयत्न करा.",
            'en': "Technical issue occurred. Please try again."
        }
        return error_responses.get(language, error_responses['en']), "", language, {}

def clear_query_cache():
    """Clear cache"""
    global _query_cache
    _query_cache.clear()

def get_cache_stats():
    """Get cache stats"""
    return {"cache_size": len(_query_cache)}

def get_model_options():
    """Get model options"""
    return {"llama3.1:8b": "Fast model for WhatsApp"}

print("✅ Basic RAG services loaded")