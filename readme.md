# SQL Question-Answering System
## PostgreSQL + pgAdmin Setup via Docker

This project sets up a PostgreSQL server and pgAdmin using Docker Compose.  
It also demonstrates how to load a `.pgsql` file into the database and view the data using pgAdmin.


## Project Structure

```
JJM-postgres_Whatsapp/
.
├── app
│   ├── __init__.py
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── models.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── middleware.py
│   │   └── routes.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── security.py
│   │   └── sql_qa.py
│   ├── static
│   │   ├── favicon.ico
│   │   ├── css
│   │   │   └── style.css
│   │   └── js
│   │       └── app.js
│   └── templates
│       └── index.html
├── docker
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── USAGE.md
├── scripts
│   ├── migrate_data.py
│   ├── setup_database.py
│   └── test_connection.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_sql_qa.py
└── utils
|    ├── __init__.py
|    └── config.py
|     ├── fastapp.py
├── fetch_tables.py
├── readme.md
├── relevant_tables.txt
├── requirements.txt
├── run.py
├── setup.py
├── sql_agent_architecture.png
├── table_names.txt      
```

## Requirements

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- VS Code Dev Containers Tool



## Step 1: Setup Environment Variables

Create a `.env` file in the root directory with the following content:

```
POSTGRES_USER=admin
POSTGRES_PASSWORD=root@123
POSTGRES_DB=postgres

PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=admin
```



## Step 2: Docker Compose Configuration

Create a file named `docker-compose.yml` with this content:

```yaml
version: '3.8'

services:
  db:
    image: postgres:14
    container_name: pg_container
    restart: always
    env_file:
      - .env
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin_snapshot
    restart: always
    env_file:
      - .env
    ports:
      - "8080:80"
    depends_on:
      - db

volumes:
  postgres_data:
```

## Step 3: Start Containers

To start everything:

```sh
docker-compose up -d
```

If you previously started the containers with different credentials and want a clean reset:

```sh
docker-compose down -v
docker-compose up -d
```

## Step 4: Access pgAdmin

Open your browser and go to:  
[http://localhost:8080](http://localhost:8080)

Login using:

- **Email:** admin@example.com  
- **Password:** admin


## Step 5: Connect to PostgreSQL in pgAdmin

After logging in to pgAdmin, register a new server using:

- **Name:** Postgres DB (or any name you prefer)

**Connection tab:**

- **Host name/address:** db
- **Port:** 5432
- **Username:** admin
- **Password:** root@123
- **Save Password:** Yes

Click **Save**.

You should now see the `postgres` database under this server.


## Step 6: Import .pgsql File

**Option A – Using Docker CLI:**

```sh
docker cp jjm-ai_11082025.pgsql pg_container:/jjm-ai_11082025.pgsql
docker exec -it pg_container bash
psql -U admin -d postgres -f /jjm-ai_11082025.pgsql
```

**Option B – Using pgAdmin GUI:**

1. In the pgAdmin browser, navigate to:  
   `Servers → Postgres DB → Databases → postgres`
2. Right-click on the database and open **Query Tool**
3. Copy the contents of your `.pgsql` file and paste it in
4. Click the **Execute ▶️** button


## Step 7: View Tables/Data in pgAdmin

1. Expand:  
   `Servers → Postgres DB → Databases → postgres → Schemas → public → Tables`
2. Right-click any table → **View/Edit Data → All Rows**



## Cleanup

To stop and remove containers and associated data:

```sh
docker-compose down -v
```
---
---
## SQL Agent
A production-ready FastAPI application that allows users to ask natural language questions about PostgreSQL databases using Ollama's Llama 3.1 8B model.

## 🚀 Features

- **Natural Language Interface**: Ask questions in plain English
- **Enhanced Security**: Comprehensive SQL injection protection and query validation
- **PostgreSQL Integration**: Full support for PostgreSQL databases with schema discovery
- **Ollama Integration**: Uses local Llama 3.1 8B model for privacy and performance
- **Production Ready**: Comprehensive logging, error handling, and monitoring
- **RESTful API**: Full FastAPI with automatic documentation
- **Web Interface**: Built-in web interface for easy interaction
- **Batch Processing**: Process multiple questions simultaneously
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 📋 Prerequisites

### Required Software
- Python 3.11+
- PostgreSQL database
- Ollama with Llama 3.1 8B model
- pgAdmin (for database management)

### Installation Steps

1. **Install Ollama and Llama 3.1 8B**:
   ```bash
   # Install Ollama (visit https://ollama.ai for installation)
   ollama serve
   ollama pull llama3.1:8b
   ```

2. **Clone and Setup**:
   ```bash
   git clone 
   cd sql_qa_system
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Copy environment configuration
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Configure Database**:
   - Use pgAdmin to connect to your PostgreSQL server
   - Execute your .pgsql file to create tables and data
   - Note your connection details for the .env file

## ⚙️ Configuration

Edit the `.env` file with your settings:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Security Settings
ENABLE_SAFETY_BY_DEFAULT=true
MAX_QUERY_RESULTS=100
```

## 🏃‍♂️ Running the Application

### Method 1: Direct Python
```bash
python run.py
```

### Method 2: Docker Compose (Recommended)
```bash
cd docker
docker-compose up -d
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Examples

### Using the Web Interface
1. Open http://localhost:8000
2. Configure your database connection
3. Ask natural language questions like:
   - "How many customers do we have?"
   - "What's our total revenue this month?"
   - "Show me the top 5 products by sales"
   - "Which customers haven't ordered in the last 90 days?"

### Using the API

```python
import requests

# Configure database
config = {
    "host": "localhost",
    "port": 5432,
    "database": "your_db",
    "username": "your_user",
    "password": "your_password"
}

requests.post("http://localhost:8000/api/configure_database", json=config)

# Ask a question
question = {"question": "How many users are registered?", "use_safety": True}
response = requests.post("http://localhost:8000/api/ask", json=question)
print(response.json()["interpretation"])
```

### Batch Processing
```python
questions = {
    "questions": [
        "What's our total revenue?",
        "How many active customers do we have?",
        "What's the average order value?"
    ],
    "use_safety": True
}

response = requests.post("http://localhost:8000/api/ask_batch", json=questions)
```

## 🛡️ Security Features

- **Query Validation**: Prevents destructive SQL operations
- **SQL Injection Protection**: Comprehensive pattern detection
- **Table Access Control**: Configurable table restrictions
- **Query Complexity Limits**: Prevents resource-intensive queries
- **Audit Logging**: All queries are logged for review
- **Safe Defaults**: Security enabled by default

## 📊 Example Questions You Can Ask

### Basic Queries
- "How many records are in the customers table?"
- "What's the total count of orders?"
- "Show me the first 10 products"

### Analytics Queries
- "What's our monthly revenue trend?"
- "Which customers have spent the most money?"
- "What are the top-selling products?"
- "Show me sales by category"

### Business Intelligence
- "What's our customer retention rate?"
- "Which regions have the highest growth?"
- "What's the average order processing time?"
- "Show me seasonal sales patterns"

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black app/
isort app/
flake8 app/
```

### Adding New Features
1. Add models in `app/models.py`
2. Implement logic in `app/core/`
3. Create API endpoints in `app/api/routes.py`
4. Add tests in `tests/`

## 📝 API Documentation

The API provides the following endpoints:

- `POST /api/configure_database` - Configure database connection
- `POST /api/ask` - Ask a single question
- `POST /api/ask_batch` - Ask multiple questions
- `GET /api/tables` - Get database schema information
- `GET /api/health` - Health check
- `POST /api/upload_sql_file` - Upload SQL files
- `GET /api/system_info` - Get system information

## 🐳 Docker Deployment

### Using Docker Compose (Includes PostgreSQL)
```bash
cd docker
cp ../.env.example .env  # Configure your settings
docker-compose up -d
```

### Using Existing Database
```bash
# Build the image
docker build -f docker/Dockerfile -t sql-qa-system .

# Run the container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name sql-qa-system \
  sql-qa-system
```

## 📋 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

2. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check connection credentials in .env
   - Ensure database exists and is accessible

3. **Model Not Found**
   ```bash
   # Pull the required model
   ollama pull llama3.1:8b
   ```

4. **Permission Denied**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER logs/ uploads/
   ```

### Logs
- Application logs: `logs/sql_qa.log`
- Docker logs: `docker-compose logs sql-qa-system`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the [API documentation](http://localhost:8000/docs)
- Review the logs in `logs/sql_qa.log`
- Open an issue on GitHub