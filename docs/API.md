# API Documentation - LangGraph Multi-Agent SQL QA System with SQL-to-NLP

## Overview

This document describes the API endpoints for the LangGraph Multi-Agent SQL Question-Answering System with SQL-to-NLP conversion capabilities.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required for the API endpoints.

## Core Endpoints

### Ask Question
**POST** `/api/ask`

Convert natural language questions to SQL queries and execute them.

**Request Body:**
```json
{
  "question": "Show me all districts in Maharashtra",
  "use_safety": true,
  "limit_results": 100,
  "response_style": "brief"
}
```

**Response:**
```json
{
  "question": "Show me all districts in Maharashtra",
  "sql_query": "SELECT district_name FROM districts ORDER BY district_name LIMIT 100",
  "result": [...],
  "interpretation": "I found 36 districts in Maharashtra...",
  "execution_time": 1.23,
  "is_safe": true,
  "validation_message": "Query is safe",
  "row_count": 36,
  "response_style": "brief",
  "current_agent": "location",
  "timestamp": "2024-01-20T10:30:00"
}
```

### Batch Questions
**POST** `/api/ask_batch`

Process multiple questions in a single request.

**Request Body:**
```json
{
  "questions": [
    "How many districts are there?",
    "Show me pending grievances",
    "List active schemes"
  ],
  "use_safety": true,
  "response_style": "brief"
}
```

## 🆕 SQL-to-NLP Endpoints

### Explain SQL Query
**POST** `/api/explain_sql`

Convert a SQL query to natural language description.

**Request Body:**
```json
{
  "sql_query": "SELECT d.district_name, COUNT(g.id) as grievance_count FROM districts d JOIN grievances g ON d.id = g.district_id GROUP BY d.district_name ORDER BY grievance_count DESC LIMIT 10",
  "context": "Analyzing district-wise grievance distribution",
  "include_analysis": true,
  "language": "en"
}
```

**Response:**
```json
{
  "sql_query": "SELECT d.district_name, COUNT(g.id) as grievance_count FROM districts d JOIN grievances g ON d.id = g.district_id GROUP BY d.district_name ORDER BY grievance_count DESC LIMIT 10",
  "description": "This query finds district names along with their total grievance counts by linking district and grievance information, then shows the top 10 districts with the most grievances",
  "is_safe": true,
  "validation_message": "Query is safe",
  "analysis": "Action: SELECT with aggregation\nTables: districts, grievances\nFilters: none\nRelationships: JOIN between districts and grievances\nOutput: district names with grievance counts",
  "complexity": "moderate",
  "agent": "sql_to_nlp",
  "timestamp": "2024-01-20T10:35:00",
  "language": "en"
}
```

### Batch SQL Explanation
**POST** `/api/explain_sql_batch`

Convert multiple SQL queries to natural language descriptions.

**Request Body:**
```json
{
  "sql_queries": [
    "SELECT * FROM districts LIMIT 10",
    "SELECT COUNT(*) FROM grievances WHERE status = 'pending'",
    "SELECT scheme_name FROM schemes WHERE is_active = true"
  ],
  "context": "Batch testing different query types",
  "include_analysis": false,
  "language": "en"
}
```

**Response:**
```json
{
  "results": [
    {
      "sql_query": "SELECT * FROM districts LIMIT 10",
      "description": "This query shows all district information in the system, displaying the first 10 results",
      "is_safe": true,
      "complexity": "simple",
      "agent": "sql_to_nlp"
    }
  ],
  "total_queries": 3,
  "successful_count": 3,
  "failed_count": 0,
  "total_execution_time": 2.45,
  "timestamp": "2024-01-20T10:40:00"
}
```

### Analyze Query Components
**POST** `/api/analyze_query`

Get detailed analysis of SQL query structure and components.

**Request Body:**
```json
{
  "sql_query": "SELECT d.district_name, COUNT(g.id) as total FROM districts d LEFT JOIN grievances g ON d.id = g.district_id GROUP BY d.district_name ORDER BY total DESC",
  "context": "Complex aggregation analysis"
}
```

**Response:**
```json
{
  "sql_query": "SELECT d.district_name, COUNT(g.id) as total FROM districts d LEFT JOIN grievances g ON d.id = g.district_id GROUP BY d.district_name ORDER BY total DESC",
  "explanation": "This query combines district information with grievance counts to show how many complaints each district has received",
  "components": {
    "type": "SELECT",
    "tables": ["districts", "grievances"],
    "joins": ["Joins with grievances"],
    "conditions": [],
    "has_aggregation": true,
    "has_grouping": true,
    "has_ordering": true,
    "has_limit": false
  },
  "complexity": "moderate",
  "agent": "sql_to_nlp",
  "timestamp": "2024-01-20T10:45:00"
}
```

## System Information Endpoints

### Health Check
**GET** `/health`

Get system health status including SQL-to-NLP agent.

**Response:**
```json
{
  "status": "healthy",
  "database": "healthy",
  "llm": "healthy", 
  "sql_to_nlp": "healthy",
  "agents": "healthy",
  "overall": "healthy",
  "version": "3.1.0",
  "system_type": "LangGraph Multi-Agent with SQL-to-NLP",
  "agents_initialized": ["router", "location", "user", "grievance", "schemes", "tracker", "sql_to_nlp"]
}
```

### System Information
**GET** `/api/system_info`

Get comprehensive system information including SQL-to-NLP capabilities.

### Database Tables
**GET** `/api/tables`

Get information about database tables and schema.

### Database Configuration
**POST** `/api/configure_database`

Configure database connection and initialize all agents including SQL-to-NLP.

## Error Responses

All endpoints return structured error responses:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## SQL-to-NLP Query Complexity Levels

- **simple**: Basic SELECT queries with no joins or aggregations
- **moderate**: Queries with joins, basic aggregations, or filtering
- **complex**: Multiple joins, complex aggregations, subqueries
- **very_complex**: Advanced features like window functions, CTEs, complex subqueries

## Supported Languages

- **en**: English (default)
- **hi**: Hindi
- **mr**: Marathi

## Rate Limiting

- Single queries: No limit
- Batch queries: Maximum 20 queries per request
- Concurrent requests: Limited to 3 simultaneous SQL-to-NLP conversions

## Example Use Cases

### Educational Use
```bash
# Explain a complex query to help users learn
curl -X POST "/api/explain_sql" -d '{
  "sql_query": "SELECT d.district_name, AVG(EXTRACT(DAY FROM (g.resolved_date - g.created_date))) as avg_days FROM districts d JOIN grievances g ON d.id = g.district_id WHERE g.status = '\''resolved'\'' GROUP BY d.district_name",
  "context": "Learning about resolution time analysis"
}'
```

### Documentation
```bash
# Generate documentation for database procedures
curl -X POST "/api/explain_sql_batch" -d '{
  "sql_queries": ["SELECT 1", "SELECT 2", "SELECT 3"],
  "context": "Documenting stored procedures"
}'
```

### Audit Trail
```bash
# Explain what a logged query actually did
curl -X POST "/api/analyze_query" -d '{
  "sql_query": "UPDATE grievances SET status = '\''resolved'\'' WHERE id = 123",
  "context": "Audit trail explanation"
}'
```