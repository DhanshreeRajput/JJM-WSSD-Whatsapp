#!/usr/bin/env python3
"""
Database setup utility script - Fixed for current environment.
Helps test database connections and setup.
"""

import os
import sys
import psycopg2
import urllib.parse
from psycopg2 import sql
from sqlalchemy import create_engine, text
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import settings, fall back to environment variables if it fails
try:
    from app.config import Settings
    use_config_class = True
except ImportError as e:
    print(f"⚠️  Could not import Settings class: {e}")
    print("   Falling back to direct environment variable reading...")
    use_config_class = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_config():
    """Get database configuration from settings or environment."""
    if use_config_class:
        try:
            settings = Settings()
            return {
                'host': settings.POSTGRES_HOST,
                'port': settings.POSTGRES_PORT,
                'database': settings.POSTGRES_DB,
                'username': settings.POSTGRES_USER,
                'password': settings.POSTGRES_PASSWORD
            }
        except Exception as e:
            print(f"⚠️  Error loading settings: {e}")
            print("   Falling back to environment variables...")
    
    # Fallback to direct environment variable reading
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'postgres'),
        'username': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'root@123')
    }

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
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote_plus(password)
        database_uri = f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}"
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
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
            """)
            table_result = conn.execute(table_query).fetchone()
            
            # Get table list with row counts
            tables_query = text("""
                SELECT 
                    t.table_name,
                    COALESCE(c.reltuples::bigint, 0) as estimated_rows
                FROM information_schema.tables t
                LEFT JOIN pg_class c ON c.relname = t.table_name
                WHERE t.table_schema = 'public' 
                AND t.table_type = 'BASE TABLE'
                ORDER BY estimated_rows DESC;
            """)
            tables_result = conn.execute(tables_query).fetchall()
            
        logger.info(f"📊 Database Information:")
        logger.info(f"   Database: {database}")
        logger.info(f"   Size: {size_result[0]}")
        logger.info(f"   Tables: {table_result[0]}")
        
        logger.info(f"\n📋 Tables with estimated row counts:")
        for table_name, rows in tables_result:
            logger.info(f"   {table_name}: {rows:,} rows (estimated)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error getting database info: {e}")
        return False

def setup_sample_data(host, port, database, username, password):
    """Create sample tables for testing."""
    try:
        # URL encode the password to handle special characters like @
        encoded_password = urllib.parse.quote_plus(password)
        database_uri = f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}"
        engine = create_engine(database_uri)
        
        sample_sql = """
        -- Create sample tables for testing postgres SQL QA System
        CREATE TABLE IF NOT EXISTS water_connections (
            connection_id SERIAL PRIMARY KEY,
            consumer_name VARCHAR(100) NOT NULL,
            address TEXT,
            area VARCHAR(50),
            connection_type VARCHAR(20) DEFAULT 'domestic',
            connection_date DATE DEFAULT CURRENT_DATE,
            status VARCHAR(20) DEFAULT 'active',
            monthly_charge DECIMAL(10,2)
        );
        
        CREATE TABLE IF NOT EXISTS complaints (
            complaint_id SERIAL PRIMARY KEY,
            connection_id INTEGER REFERENCES water_connections(connection_id),
            complaint_type VARCHAR(50),
            description TEXT,
            complaint_date DATE DEFAULT CURRENT_DATE,
            status VARCHAR(20) DEFAULT 'pending',
            resolved_date DATE,
            assigned_staff VARCHAR(100)
        );
        
        CREATE TABLE IF NOT EXISTS staff (
            staff_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            designation VARCHAR(50),
            department VARCHAR(50),
            contact_number VARCHAR(15),
            area_assigned VARCHAR(50),
            joining_date DATE DEFAULT CURRENT_DATE
        );
        
        CREATE TABLE IF NOT EXISTS projects (
            project_id SERIAL PRIMARY KEY,
            project_name VARCHAR(200) NOT NULL,
            area VARCHAR(50),
            project_type VARCHAR(50),
            budget DECIMAL(15,2),
            start_date DATE,
            expected_completion DATE,
            status VARCHAR(20) DEFAULT 'planned',
            contractor_name VARCHAR(100)
        );
        
        -- Insert sample data for postgres
        INSERT INTO water_connections (consumer_name, address, area, connection_type, monthly_charge) VALUES
        ('Ramesh Kumar', '123 Gandhi Nagar', 'Pune Central', 'domestic', 250.00),
        ('Sunita Patil', '456 Shivaji Road', 'Pimpri', 'domestic', 300.00),
        ('Maharashtra Hotel', '789 Main Street', 'Pune Central', 'commercial', 1500.00),
        ('Amit Sharma', '321 Nehru Colony', 'Chinchwad', 'domestic', 275.00),
        ('Green Industries', '654 Industrial Area', 'Bhosari', 'industrial', 5000.00)
        ON CONFLICT (connection_id) DO NOTHING;
        
        INSERT INTO complaints (connection_id, complaint_type, description, status) VALUES
        (1, 'Water Quality', 'Water is not clear', 'pending'),
        (2, 'Low Pressure', 'Very low water pressure in morning', 'resolved'),
        (3, 'Billing Issue', 'Incorrect bill amount', 'in_progress'),
        (4, 'No Water Supply', 'No water for 3 days', 'pending'),
        (1, 'Leakage', 'Pipeline leakage near house', 'resolved')
        ON CONFLICT DO NOTHING;
        
        INSERT INTO staff (name, designation, department, contact_number, area_assigned) VALUES
        ('Suresh Jadhav', 'Engineer', 'Water Supply', '9876543210', 'Pune Central'),
        ('Priya Deshmukh', 'Assistant Engineer', 'Quality Control', '9876543211', 'Pimpri'),
        ('Ravi Patil', 'Technician', 'Maintenance', '9876543212', 'Chinchwad'),
        ('Meera Kulkarni', 'Manager', 'Customer Service', '9876543213', 'All Areas'),
        ('Ganesh More', 'Inspector', 'Water Quality', '9876543214', 'Bhosari')
        ON CONFLICT DO NOTHING;
        
        INSERT INTO projects (project_name, area, project_type, budget, status, contractor_name) VALUES
        ('New Pipeline Installation', 'Pune Central', 'Infrastructure', 5000000.00, 'ongoing', 'ABC Contractors'),
        ('Water Quality Testing Lab', 'Pimpri', 'Facility', 2000000.00, 'completed', 'XYZ Builders'),
        ('Smart Water Meters', 'Chinchwad', 'Technology', 3000000.00, 'planned', 'Tech Solutions Pvt Ltd'),
        ('Sewage Treatment Plant', 'Bhosari', 'Treatment', 15000000.00, 'ongoing', 'Green Solutions Ltd'),
        ('Customer Service Center', 'All Areas', 'Service', 1000000.00, 'completed', 'Modern Builders')
        ON CONFLICT DO NOTHING;
        """
        
        with engine.connect() as conn:
            conn.execute(text(sample_sql))
            conn.commit()
        
        logger.info("✅ Sample postgres data created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

def main():
    """Main setup function."""
    print("🔧 postgres Database Setup Utility")
    print("=" * 50)
    print("🏢 Water Supply and Sanitation Department")
    print("🏛️  Government of Maharashtra")
    print("=" * 50)
    
    # Get configuration
    try:
        config = get_database_config()
        print(f"Using database configuration:")
        print(f"  Host: {config['host']}")
        print(f"  Port: {config['port']}")
        print(f"  Database: {config['database']}")
        print(f"  User: {config['username']}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    print("\n🔍 Testing connection...")
    if not test_connection(**config):
        print("❌ Cannot proceed without valid database connection")
        print("\n💡 Check your .env file or environment variables:")
        print("   POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        return
    
    print("\n📊 Getting database information...")
    if get_database_info(**config):
        print("\n✅ Database connection and info retrieval successful!")
    else:
        print("\n❌ Failed to get database information")
        return