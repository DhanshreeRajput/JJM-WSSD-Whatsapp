#!/usr/bin/env python3
"""Test script for SQL-to-NLP functionality."""

import asyncio
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_sql_to_nlp():
    """Test the SQL to NLP conversion functionality."""
    try:
        from app.agents.sql_to_nlp_agent import SQLToNLPAgent
        from app.core.database import DatabaseManager
        from app.config import settings
        
        print("🚀 Testing SQL-to-NLP functionality...")
        print(f"Database URL configured: {bool(settings.database_url)}")
        print(f"Ollama model: {settings.OLLAMA_MODEL}")
        print(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
        print("-" * 50)
        
        # Test queries
        test_queries = [
            {
                "sql": "SELECT district_name FROM districts LIMIT 10",
                "context": "Getting list of districts",
                "description": "Simple SELECT query"
            },
            {
                "sql": "SELECT COUNT(*) as total_grievances FROM grievances WHERE status = 'pending'",
                "context": "Checking pending grievances",
                "description": "Count query with condition"
            },
            {
                "sql": "SELECT d.district_name, COUNT(g.id) as grievance_count FROM districts d JOIN grievances g ON d.id = g.district_id GROUP BY d.district_name ORDER BY grievance_count DESC LIMIT 5",
                "context": "District-wise grievance analysis",
                "description": "Complex query with JOIN and aggregation"
            }
        ]
        
        # Initialize database manager and agent
        print("📊 Initializing database manager...")
        db_manager = DatabaseManager(settings.database_url)
        
        print("🤖 Initializing SQL-to-NLP agent...")
        sql_to_nlp_agent = SQLToNLPAgent(db_manager)
        
        print("✅ Agents initialized successfully!")
        print("-" * 50)
        
        # Test individual conversions
        print("🧪 Testing individual SQL-to-NLP conversions:")
        for i, test_case in enumerate(test_queries, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"SQL: {test_case['sql']}")
            
            try:
                result = await sql_to_nlp_agent.convert_sql_to_nlp(
                    sql_query=test_case['sql'],
                    context=test_case['context'],
                    include_analysis=True
                )
                
                print(f"✅ Safety: {result['is_safe']}")
                print(f"📝 Description: {result['description']}")
                print(f"📊 Complexity: {result.get('complexity', 'unknown')}")
                
                if result.get('analysis'):
                    print(f"🔍 Analysis: {result['analysis'][:100]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test batch conversion
        print("\n" + "=" * 50)
        print("🧪 Testing batch SQL-to-NLP conversion:")
        
        batch_queries = [tc['sql'] for tc in test_queries]
        try:
            batch_results = await sql_to_nlp_agent.batch_convert(
                sql_queries=batch_queries,
                context="Batch testing of SQL-to-NLP functionality"
            )
            
            print(f"✅ Batch conversion completed")
            print(f"📊 Results: {len(batch_results)} queries processed")
            
            success_count = sum(1 for r in batch_results if r.get('is_safe', False))
            print(f"✅ Successful: {success_count}/{len(batch_results)}")
            
        except Exception as e:
            print(f"❌ Batch conversion error: {e}")
        
        # Test health check
        print("\n" + "=" * 50)
        print("🏥 Testing agent health check:")
        try:
            health = await sql_to_nlp_agent.health_check()
            print(f"✅ Health check: {'Healthy' if health else 'Failed'}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test agent info
        print("\n" + "=" * 50)
        print("ℹ️ Agent Information:")
        try:
            info = sql_to_nlp_agent.get_agent_info()
            print(f"📛 Name: {info['name']}")
            print(f"📝 Description: {info['description']}")
            print(f"🛠️ Capabilities: {len(info['capabilities'])} features")
            print(f"🔒 Safety Features: {info['safety_features']}")
            print(f"🌐 Languages: {info['languages_supported']}")
        except Exception as e:
            print(f"❌ Agent info error: {e}")
        
        print("\n🎉 SQL-to-NLP testing completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_imports():
    """Test basic imports."""
    try:
        print("🧪 Testing imports...")
        
        import langchain_community
        print("✅ langchain_community imported")
        
        import langgraph
        print("✅ langgraph imported")
        
        from app.config import settings
        print("✅ app.config imported")
        
        from app.models import SQLToNLPRequest, SQLToNLPResponse
        print("✅ SQL-to-NLP models imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔬 SQL-to-NLP Functionality Test Suite")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("❌ Import tests failed. Please check dependencies.")
        sys.exit(1)
    
    # Test async functionality
    try:
        asyncio.run(test_sql_to_nlp())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        sys.exit(1)