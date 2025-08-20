# LangGraph Multi-Agent SQL Question-Answering System
## Water Supply and Sanitation Department (WSSD) - Government of Maharashtra

This project provides a production-ready FastAPI application with LangGraph multi-agent architecture that allows users to ask natural language questions about PostgreSQL databases using Ollama's Llama 3.1 8B model.

## 🎯 Multi-Agent Architecture

The system uses **6 specialized agents** for intelligent query processing:

- **🎯 Router Agent**: Intelligently routes questions to appropriate specialists
- **📍 Location Agent**: Districts, circles, blocks, villages, administrative boundaries
- **👥 User Agent**: Citizens, registrations, accounts, user management
- **📝 Grievance Agent**: Complaints, grievances, issues, resolutions
- **🏛️ Schemes Agent**: Government schemes, programs, initiatives
- **📊 Tracker Agent**: Status tracking, progress monitoring, logs

## Project Structure

```
JJM-postgres_Whatsapp/
├── app/
│   ├── __init__.py
│   ├── config.py                    # Configuration with LangGraph settings
│   ├── dependencies.py              # Updated for LangGraph support
│   ├── main.py                      # Updated FastAPI with multi-agent
│   ├── models.py                    # Enhanced Pydantic models
│   ├── api/
│   │   ├── __init__.py
│   │   ├── middleware.py
│   │   └── routes.py                # Updated routes for LangGraph
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py              # Enhanced database manager
│   │   ├── langgraph_system.py      # 🆕 Main LangGraph orchestrator
│   │   ├── security.py              # Existing security validation
│   │   └── sql_qa.py                # Legacy system (kept for compatibility)
│   ├── agents/                      # 🆕 Multi-Agent System
│   │   ├── __init__.py
│   │   ├── location_agent.py        # 🆕 Location specialist agent
│   │   ├── user_agent.py            # 🆕 User specialist agent
│   │   ├── grievance_agent.py       # 🆕 Grievance specialist agent
│   │   ├── schemes_agent.py         # 🆕 Schemes specialist agent
│   │   └── tracker_agent.py         # 🆕 Tracker specialist agent
│   ├── utils/                       # 🆕 Utilities
│   │   ├── __init__.py
│   │   └── table_mappings.py        # 🆕 Table-agent mappings
│   ├── static/
│   │   ├── favicon.ico
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js
│   └── templates/
│       └── index.html               # Updated web interface
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── USAGE.md
├── scripts/
│   ├── migrate_data.py
│   ├── setup_database.py
│   └── test_connection.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_sql_qa.py
├── utils/
│   ├── __init__.py
│   └── config.py
├── fastapp.py                       # WhatsApp integration
├── fetch_tables.py
├── readme.md
├── relevant_tables.txt              # Agent-table mappings
├── requirements.txt                 # Updated with LangGraph
├── run.py                          # Application runner
├── setup.py
└── table_names.txt
```

## 🚀 New Features (v3.0.0)

- **🧠 LangGraph Multi-Agent Architecture**: Intelligent question routing
- **🎯 Specialized Agents**: Domain-specific expertise for better accuracy
- **📊 Agent Analytics**: Monitor which agent handles each query
- **🔄 Parallel Processing**: Concurrent query processing capabilities
- **📈 Enhanced Performance**: Optimized for WSSD database structure
- **🛡️ Maintained Security**: All existing security features preserved
- **🔄 Backward Compatibility**: Existing APIs continue to work

## 📋 Prerequisites

### Required Software
- Python 3.11+
- PostgreSQL database with WSSD schema
- Ollama with Llama 3.1 8B model
- pgAdmin (for database management)

### New Dependencies
```bash
# Core requirement for multi-agent system
langgraph>=0.0.55

# Existing dependencies maintained
langchain<=0.0.352
langchain-community<=0.0.38
fastapi==0.104.1
```

## ⚙️ Installation & Setup

### 1. **Install LangGraph and Dependencies**
```bash
# Install new dependencies
pip install langgraph

# Or install all requirements
pip install -r requirements.txt
```

### 2. **Install Ollama and Model**
```bash
# Install Ollama (visit https://ollama.ai for installation)
ollama serve
ollama pull llama3.1:8b
```

### 3. **Database Setup**
```bash
# Using Docker (Recommended)
docker-compose up -d

# Access pgAdmin at http://localhost:8080
# Default credentials: admin@example.com / admin
```

### 4. **Environment Configuration**
```bash
# Copy and edit environment file
cp .env.example .env
```

Edit `.env` with your WSSD database settings:
```bash
# WSSD Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=wssd
POSTGRES_USER=postgres
POSTGRES_PASSWORD=root@123
DB_SCHEMA=public

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Organization Information
ORG_NAME=Water Supply and Sanitation Department
ORG_STATE=Government of Maharashtra
WEBSITE_URL=https://mahajalsamadhan.in
HELPLINE_NUMBERS=104,102
```

## 🏃‍♂️ Running the LangGraph System

### **Method 1: Direct Python (Recommended)**
```bash
python run.py
```

### **Method 2: Docker Compose**
```bash
cd docker
docker-compose up -d
```

### **Method 3: Direct uvicorn**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 System Configuration

### **Step 1: Configure Database Connection**
```bash
curl -X POST "http://localhost:8000/api/configure_database" \
     -H "Content-Type: application/json" \
     -d '{
       "host": "localhost",
       "port": 5432,
       "database": "wssd",
       "username": "postgres", 
       "password": "root@123",
       "schema": "public"
     }'
```

Expected Response:
```json
{
  "message": "LangGraph multi-agent system configured successfully",
  "system_type": "LangGraph Multi-Agent",
  "agents_initialized": ["router", "location", "user", "grievance", "schemes", "tracker"],
  "tables_found": 25
}
```

### **Step 2: Verify System Health**
```bash
curl http://localhost:8000/health
```

## 🧪 Testing Multi-Agent Intelligence

### **Location Queries** (→ Location Agent)
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Show me all districts in Maharashtra",
       "use_safety": true
     }'
```

### **User Queries** (→ User Agent)
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How many citizens are registered?",
       "use_safety": true
     }'
```

### **Grievance Queries** (→ Grievance Agent)
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "List recent grievances by category",
       "use_safety": true
     }'
```

### **Schemes Queries** (→ Schemes Agent)
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What government schemes are available?",
       "use_safety": true
     }'
```

### **Tracker Queries** (→ Tracker Agent)
```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Track grievance resolution status",
       "use_safety": true
     }'
```

## 📊 WSSD Database Structure

### **Agent-Table Mapping**
```
🏛️ Location Agent:
├── districts, divisions, sub_divisions
├── regions, region_circles, circles
├── blocks, villages, grampanchayats
└── habitations, states

👥 User Agent:
├── users (contains all location IDs)
├── citizen_users
└── grievance_users

📝 Grievance Agent:
├── grievances
├── grievance_categories
└── sub_grievance_categories

🎯 Schemes Agent:
├── scheme_categories
├── scheme_types
└── schemes

📊 Tracker Agent:
├── grievance_resolve_tracks
├── grievance_resolve_track_logs
└── grievance_assigned_accept_reject_users
```

## 🌐 Web Interface

Visit: **http://localhost:8000**

The updated web interface now shows:
- **Multi-agent status** and capabilities
- **Real-time agent routing** information
- **Enhanced chat experience** with agent insights
- **System performance** metrics

## 📖 WSSD-Specific Usage Examples

### **Administrative Queries**
- "How many districts are under our jurisdiction?"
- "List all circles in Pune district"
- "Show me block-wise distribution"
- "Which villages are in Baramati block?"

### **Citizen Management**
- "How many citizens registered this month?"
- "Show user distribution by district"
- "List citizens from Mumbai district"
- "What's the total user count?"

### **Grievance Management**
- "Show pending grievances"
- "List grievances by category"
- "How many complaints resolved this week?"
- "Show water supply related complaints"

### **Scheme Information**
- "What water supply schemes are active?"
- "List all government schemes"
- "Show scheme implementation by district"
- "Which schemes have highest enrollment?"

### **Progress Tracking**
- "Track grievance ID 123 status"
- "Show resolution timeline for recent complaints"
- "List assigned but unresolved issues"
- "Track scheme implementation progress"

## 🛡️ Enhanced Security Features

- **Agent-Level Security**: Each agent validates queries for its domain
- **SQL Injection Protection**: Comprehensive pattern detection
- **Table Access Control**: Agent-specific table restrictions
- **Query Complexity Limits**: Prevents resource-intensive queries
- **Audit Logging**: Track which agent processed each query
- **Safe Defaults**: Security enabled by default

## 📝 Enhanced API Documentation

### **New Endpoints**
- `GET /api/system_info` - Get multi-agent system information
- `GET /health` - Enhanced health check with agent status

### **Enhanced Response Format**
```json
{
  "question": "Show me districts",
  "sql_query": "SELECT district_name FROM districts LIMIT 100",
  "result": [...],
  "interpretation": "I found 15 districts in Maharashtra...",
  "execution_time": 1.23,
  "is_safe": true,
  "current_agent": "location",
  "row_count": 15,
  "timestamp": "2024-01-20T10:30:00"
}
```

## 📊 Monitoring & Analytics

### **Agent Performance**
```bash
curl http://localhost:8000/api/system_info
```

### **System Health**
```bash
curl http://localhost:8000/health
```

### **Logs**
```bash
tail -f logs/wssd_langgraph_qa.log
```

## 🐳 Docker Deployment

### **WSSD Production Setup**
```bash
cd docker
cp ../.env.example .env
# Edit .env with production WSSD database credentials
docker-compose up -d
```

### **Services Available**
- **API**: http://localhost:8000
- **pgAdmin**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs

## 📋 Troubleshooting

### **LangGraph Issues**
```bash
# Install/upgrade LangGraph
pip install --upgrade langgraph

# Check imports
python -c "import langgraph; print('LangGraph installed successfully')"
```

### **Agent Routing Issues**
```bash
# Check agent mappings
curl http://localhost:8000/api/system_info

# View routing logs
grep "Router decision" logs/wssd_langgraph_qa.log
```

### **Database Connection**
```bash
# Test WSSD database
python scripts/setup_database.py

# Check table access
curl http://localhost:8000/api/tables
```

## 🚀 Migration from Legacy System

The LangGraph system is **fully backward compatible**:

- ✅ **Existing APIs work unchanged**
- ✅ **Same configuration format**
- ✅ **All security features maintained**
- ✅ **Web interface enhanced but compatible**
- ✅ **Database connections unchanged**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add agent-specific functionality
4. Test with WSSD database
5. Submit a pull request


## 🆘 Support

For WSSD-specific support:
- **Helpline**: 104, 102
- **Website**: https://mahajalsamadhan.in
- **API Documentation**: http://localhost:8000/docs
- **System Logs**: `logs/wssd_langgraph_qa.log`

---

**Government of Maharashtra | Water Supply and Sanitation Department**  
*Empowering Citizens through Intelligent Data Access*