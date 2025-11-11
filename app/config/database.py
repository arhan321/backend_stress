import os
import mysql.connector
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    """Database configuration for MySQL Laragon connection"""
    
    # Laragon MySQL default configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')  # Laragon default is empty
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'stress_analysis')
    
    # SQLAlchemy configuration with fixed connection settings for Laragon
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_timeout': 20,
        'max_overflow': 0,
        'connect_args': {
            'connect_timeout': 60,
            'charset': 'utf8mb4',
            'use_unicode': True,
            'autocommit': False
        }
    }

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.metadata = MetaData()
        
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            # Create database if it doesn't exist
            self._create_database_if_not_exists()
            
            # Create SQLAlchemy engine with better error handling
            self.engine = create_engine(
                DatabaseConfig.SQLALCHEMY_DATABASE_URI,
                **DatabaseConfig.SQLALCHEMY_ENGINE_OPTIONS
            )
            
            # Test the connection immediately with SQLAlchemy 2.0 syntax
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                connection.commit()
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            logger.error(f"Connection URI: {DatabaseConfig.SQLALCHEMY_DATABASE_URI}")
            return False
    
    def _create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect without specifying database
            connection = mysql.connector.connect(
                host=DatabaseConfig.MYSQL_HOST,
                port=DatabaseConfig.MYSQL_PORT,
                user=DatabaseConfig.MYSQL_USER,
                password=DatabaseConfig.MYSQL_PASSWORD,
                charset='utf8mb4',
                use_unicode=True
            )
            
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DatabaseConfig.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {DatabaseConfig.MYSQL_DATABASE}")
            
            cursor.close()
            connection.close()
            
            logger.info(f"Database '{DatabaseConfig.MYSQL_DATABASE}' ready")
            
        except Exception as e:
            logger.error(f"Failed to create database: {str(e)}")
            raise
    
    def get_session(self):
        """Get database session"""
        if self.SessionLocal is None:
            raise Exception("Database not initialized. Call initialize_database() first.")
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all tables defined in models"""
        try:
            # Import Base from models to avoid conflicts
            from models.database_models import Base
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            if self.engine is None:
                logger.error("Database engine not initialized")
                return False
                
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                connection.commit()
                
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 