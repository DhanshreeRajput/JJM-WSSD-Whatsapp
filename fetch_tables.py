"""
Simple script to fetch all table names from PostgreSQL database
"""

import asyncio
import asyncpg
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def fetch_all_tables():
    """Fetch and display all table names from the database"""
    
    # Database connection
    DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    
    try:
        # Connect to database
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected to database")
        
        # Fetch all table names
        query = """
        SELECT table_name, 
               (SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = t.table_name AND table_schema = 'public') as column_count
        FROM information_schema.tables t
        WHERE table_schema = 'public' 
        ORDER BY table_name
        """
        
        tables = await conn.fetch(query)
        
        print(f"\n📊 Found {len(tables)} tables in the database:\n")
        print("-" * 60)
        print(f"{'Table Name':<40} {'Columns':<10}")
        print("-" * 60)
        
        for i, table in enumerate(tables, 1):
            table_name = table['table_name']
            column_count = table['column_count']
            print(f"{i:2}. {table_name:<38} {column_count:<10}")
        
        print("-" * 60)
        print(f"Total: {len(tables)} tables")
        
        # Also get total row counts for first few tables (sample)
        print(f"\n📈 Sample row counts:")
        print("-" * 40)
        
        for table in tables[:10]:  # First 10 tables
            try:
                count_query = f"SELECT COUNT(*) FROM {table['table_name']}"
                row_count = await conn.fetchval(count_query)
                print(f"{table['table_name']:<30} {row_count:>8} rows")
            except Exception as e:
                print(f"{table['table_name']:<30} {'Error':>8}")
        
        if len(tables) > 10:
            print(f"... and {len(tables) - 10} more tables")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_all_tables())