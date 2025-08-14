#!/usr/bin/env python3
"""
Database setup utility script.
Helps test database connections and setup.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection(host, port, database, username, password):
    """Test PostgreSQL connection."""
    try:
        # Test with psycopg2
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Connection successful!")
        logger.info(f"PostgreSQL version: {version}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

def get_database_info(host, port, database, username, password):
    """Get database information."""
    try:
        database_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(database_uri)
        
        with engine.connect() as conn:
            # Get database size
            size_query = text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as db_size;
            """)
            size_result = conn.execute(size_query).fetchone()
            
            # Get table count
            table_query = text("""
                SELECT count(*) as table_count
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            table_result = conn.execute(table_query).fetchone()
            
            # Get table list with row counts
            tables_query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins - n_tup_del as row_count
                FROM pg_stat_user_tables
                ORDER BY row_count DESC;
            """)
            tables_result = conn.execute(tables_query).fetchall()
            
        logger.info(f"📊 Database Information:")
        logger.info(f"   Database: {database}")
        logger.info(f"   Size: {size_result[0]}")
        logger.info(f"   Tables: {table_result[0]}")
        
        logger.info(f"\n📋 Tables with row counts:")
        for schema, table, rows in tables_result:
            logger.info(f"   {schema}.{table}: {rows:,} rows")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error getting database info: {e}")
        return False

def setup_sample_data(host, port, database, username, password):
    """Create sample tables for testing."""
    try:
        database_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(database_uri)
        
        sample_sql = """
        -- Create sample tables for testing
        CREATE TABLE IF NOT EXISTS customers (
            customer_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            city VARCHAR(50),
            registration_date DATE DEFAULT CURRENT_DATE,
            is_active BOOLEAN DEFAULT TRUE
        );
        
        CREATE TABLE IF NOT EXISTS products (
            product_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            category VARCHAR(50),
            price DECIMAL(10,2),
            stock_quantity INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS orders (
            order_id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            order_date DATE DEFAULT CURRENT_DATE,
            total_amount DECIMAL(10,2),
            status VARCHAR(20) DEFAULT 'pending'
        );
        
        CREATE TABLE IF NOT EXISTS order_items (
            item_id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10,2)
        );
        
        -- Insert sample data
        INSERT INTO customers (name, email, city) VALUES
        ('John Doe', 'john@example.com', 'New York'),
        ('Jane Smith', 'jane@example.com', 'Los Angeles'),
        ('Bob Johnson', 'bob@example.com', 'Chicago'),
        ('Alice Brown', 'alice@example.com', 'Houston'),
        ('Charlie Wilson', 'charlie@example.com', 'Phoenix')
        ON CONFLICT (email) DO NOTHING;
        
        INSERT INTO products (name, category, price, stock_quantity) VALUES
        ('Laptop', 'Electronics', 999.99, 50),
        ('Mouse', 'Electronics', 29.99, 200),
        ('Keyboard', 'Electronics', 79.99, 150),
        ('Monitor', 'Electronics', 299.99, 75),
        ('Headphones', 'Electronics', 149.99, 100)
        ON CONFLICT DO NOTHING;
        
        INSERT INTO orders (customer_id, total_amount, status) VALUES
        (1, 1099.98, 'completed'),
        (2, 29.99, 'completed'),
        (3, 379.98, 'pending'),
        (4, 149.99, 'completed'),
        (5, 109.98, 'shipped')
        ON CONFLICT DO NOTHING;
        """
        
        with engine.connect() as conn:
            conn.execute(text(sample_sql))
            conn.commit()
        
        logger.info("✅ Sample data created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

def main():
    """Main setup function."""
    print("🔧 Database Setup Utility")
    print("=" * 50)
    
    # Get configuration
    try:
        settings = Settings()
        if settings.DB_NAME:
            print(f"Using configuration from .env file:")
            print(f"  Host: {settings.DB_HOST}")
            print(f"  Port: {settings.DB_PORT}")
            print(f"  Database: {settings.DB_NAME}")
            print(f"  User: {settings.DB_USER}")
            
            host = settings.DB_HOST
            port = settings.DB_PORT
            database = settings.DB_NAME
            username = settings.DB_USER
            password = settings.DB_PASSWORD
        else:
            print("No configuration found in .env file")
            print("Please enter database connection details:")
            
            host = input("Host (localhost): ") or "localhost"
            port = int(input("Port (5432): ") or "5432")
            database = input("Database name: ")
            username = input("Username: ")
            password = input("Password: ")
    
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    print("\n🔍 Testing connection...")
    if not test_connection(host, port, database, username, password):
        print("❌ Cannot proceed without valid database connection")
        return
    
    print("\n📊 Getting database information...")
    get_database_info(host, port, database, username, password)
    
    # Ask if user wants to create sample data
    create_sample = input("\n❓ Create sample data for testing? (y/N): ").lower().startswith('y')
    if create_sample:
        print("\n🔨 Creating sample data...")
        setup_sample_data(host, port, database, username, password)
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Configure your .env file with the database credentials")
    print("2. Run: python run.py")
    print("3. Open: http://localhost:8000")

if __name__ == "__main__":
    main()