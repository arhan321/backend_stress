#!/usr/bin/env python3
"""
Automated MySQL Database Setup for Laragon
This script automatically creates and configures the stress_analysis database
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import subprocess
import time
from loguru import logger
import bcrypt

class LaragonDBSetup:
    """Automated database setup for Laragon MySQL"""
    
    def __init__(self):
        self.host = 'localhost'
        self.port = 3306
        self.user = 'root'
        self.password = ''  # Laragon default
        self.database = 'stress_analysis'
        self.connection = None
        
    def check_laragon_running(self):
        """Check if Laragon MySQL is running"""
        try:
            # Try to connect to MySQL
            connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            if connection.is_connected():
                connection.close()
                logger.info("✅ Laragon MySQL is running")
                return True
        except Error as e:
            logger.error(f"❌ Laragon MySQL not running: {e}")
            return False
            
    def create_database(self):
        """Create the stress_analysis database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            
            cursor = self.connection.cursor()
            
            # Create database
            cursor.execute(f"DROP DATABASE IF EXISTS {self.database}")
            cursor.execute(f"CREATE DATABASE {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {self.database}")
            
            logger.info(f"✅ Database '{self.database}' created successfully")
            return True
            
        except Error as e:
            logger.error(f"❌ Failed to create database: {e}")
            return False
            
    def create_tables(self):
        """Create all required tables"""
        try:
            cursor = self.connection.cursor()
            
            # Read SQL file and execute
            sql_file_path = os.path.join(os.path.dirname(__file__), 'database_init.sql')
            
            if not os.path.exists(sql_file_path):
                logger.error(f"❌ SQL file not found: {sql_file_path}")
                return False
                
            with open(sql_file_path, 'r', encoding='utf-8') as file:
                sql_content = file.read()
            
            # Better SQL parsing - handle multi-line statements
            import re
            
            # Remove comments and normalize whitespace
            sql_content = re.sub(r'--[^\n]*\n', '\n', sql_content)
            sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)
            
            # Split by semicolon but preserve statements
            statements = []
            current_statement = []
            in_string = False
            escape_next = False
            
            for line in sql_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                current_statement.append(line)
                
                # Check if statement is complete (ends with ;)
                if line.endswith(';') and not in_string:
                    full_statement = ' '.join(current_statement).strip()
                    if full_statement and not full_statement.startswith('--'):
                        statements.append(full_statement)
                    current_statement = []
            
            # Execute each statement
            for statement in statements:
                if statement.strip():
                    try:
                        # Remove trailing semicolon for execution
                        clean_statement = statement.rstrip(';').strip()
                        if clean_statement:
                            cursor.execute(clean_statement)
                            logger.debug(f"Executed: {clean_statement[:50]}...")
                    except Error as e:
                        error_msg = str(e).lower()
                        if 'already exists' in error_msg or 'duplicate' in error_msg:
                            logger.debug(f"Skipping existing: {clean_statement[:50]}...")
                        else:
                            logger.warning(f"Warning executing: {clean_statement[:50]}... - {e}")
                            
            self.connection.commit()
            logger.info("✅ All tables created successfully")
            return True
            
        except Error as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False
            
    def create_admin_user(self):
        """Create default admin user with hashed password"""
        try:
            cursor = self.connection.cursor()
            
            # Hash password
            password = "admin123"
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert admin user
            insert_user_query = """
            INSERT INTO users (username, email, password_hash, full_name, role, department, position, is_active, email_verified) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash)
            """
            
            user_data = (
                'admin',
                'admin@example.com', 
                hashed_password,
                'System Administrator',
                'admin',
                'IT',
                'Administrator',
                1,
                1
            )
            
            cursor.execute(insert_user_query, user_data)
            
            # Get user ID
            cursor.execute("SELECT id FROM users WHERE username = 'admin'")
            result = cursor.fetchone()
            if result:
                admin_id = result[0]
                
                # Insert permissions
                permissions = [
                    'admin_access',
                    'user_management', 
                    'system_config',
                    'data_analysis',
                    'report_generation'
                ]
                
                for permission in permissions:
                    insert_permission_query = """
                    INSERT INTO user_permissions (user_id, permission) 
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE permission = VALUES(permission)
                    """
                    cursor.execute(insert_permission_query, (admin_id, permission))
                    
            self.connection.commit()
            logger.info("✅ Admin user created (admin@example.com / admin123)")
            return True
            
        except Error as e:
            logger.error(f"❌ Failed to create admin user: {e}")
            return False
            
    def insert_sample_data(self):
        """Insert sample employee data"""
        try:
            cursor = self.connection.cursor()
            
            sample_employees = [
                ('EMP0001', 'IT', 28, 'Developer', 7.5, 6.2, 4.0, 6.8, 7.0, 45.2),
                ('EMP0002', 'Finance', 35, 'Analyst', 8.2, 5.5, 5.2, 5.8, 6.5, 62.3),
                ('EMP0003', 'HR', 31, 'Specialist', 6.0, 7.8, 3.5, 7.5, 8.0, 32.1),
                ('EMP0004', 'Marketing', 29, 'Manager', 8.5, 5.0, 6.0, 6.0, 6.8, 58.7),
                ('EMP0005', 'Operations', 42, 'Supervisor', 7.0, 6.5, 4.8, 6.2, 7.2, 48.9),
                ('EMP0006', 'IT', 26, 'Junior Developer', 6.5, 7.0, 3.0, 7.2, 7.5, 38.5),
                ('EMP0007', 'Finance', 33, 'Senior Analyst', 8.0, 5.8, 4.5, 6.0, 6.8, 55.3),
                ('EMP0008', 'Marketing', 27, 'Coordinator', 7.2, 6.5, 4.2, 6.5, 7.0, 47.8),
                ('EMP0009', 'HR', 38, 'Manager', 7.8, 6.0, 5.0, 6.8, 7.2, 52.1),
                ('EMP0010', 'Operations', 45, 'Director', 8.8, 4.5, 6.5, 5.5, 6.0, 68.9)
            ]
            
            insert_employee_query = """
            INSERT INTO employees (employee_id, department, age, position, workload, work_life_balance, 
                                 team_conflict, management_support, work_environment, stress_level) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                department = VALUES(department),
                age = VALUES(age),
                position = VALUES(position),
                workload = VALUES(workload),
                work_life_balance = VALUES(work_life_balance),
                team_conflict = VALUES(team_conflict),
                management_support = VALUES(management_support),
                work_environment = VALUES(work_environment),
                stress_level = VALUES(stress_level)
            """
            
            cursor.executemany(insert_employee_query, sample_employees)
            
            # Insert sample dataset record
            insert_dataset_query = """
            INSERT INTO datasets (name, description, file_path, record_count, status) 
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                description = VALUES(description),
                record_count = VALUES(record_count),
                status = VALUES(status)
            """
            
            dataset_data = (
                'Sample Employee Data',
                'Initial sample dataset for testing',
                'data/samples/employee_stress_data_500.csv',
                len(sample_employees),
                'processed'
            )
            
            cursor.execute(insert_dataset_query, dataset_data)
            
            self.connection.commit()
            logger.info(f"✅ Sample data inserted ({len(sample_employees)} employees)")
            return True
            
        except Error as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            return False
            
    def create_env_file(self):
        """Create .env file with database configuration"""
        try:
            env_content = f"""# Database Configuration for Laragon
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE={self.database}

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRES=3600

# File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216
"""
            
            env_file_path = os.path.join(os.path.dirname(__file__), '.env')
            with open(env_file_path, 'w', encoding='utf-8') as file:
                file.write(env_content)
                
            logger.info("✅ .env file created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create .env file: {e}")
            return False
            
    def verify_setup(self):
        """Verify database setup"""
        try:
            cursor = self.connection.cursor()
            
            # Check tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            expected_tables = [
                'employees', 'datasets', 'analysis_results', 'stress_predictions',
                'model_training', 'system_logs', 'users', 'user_sessions',
                'user_permissions', 'audit_logs'
            ]
            
            missing_tables = [table for table in expected_tables if table not in table_names]
            
            if missing_tables:
                logger.warning(f"⚠️ Missing tables: {missing_tables}")
            else:
                logger.info("✅ All tables present")
                
            # Check admin user
            cursor.execute("SELECT username, email, role FROM users WHERE role = 'admin'")
            admin_users = cursor.fetchall()
            
            if admin_users:
                logger.info(f"✅ Admin users found: {len(admin_users)}")
                for user in admin_users:
                    logger.info(f"   - {user[0]} ({user[1]})")
            else:
                logger.warning("⚠️ No admin users found")
                
            # Check employee data
            cursor.execute("SELECT COUNT(*) FROM employees")
            employee_count = cursor.fetchone()[0]
            logger.info(f"✅ Employee records: {employee_count}")
            
            return True
            
        except Error as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
            
    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("✅ Database connection closed")
            
    def run_setup(self):
        """Run complete database setup"""
        logger.info("🚀 Starting Laragon MySQL Database Setup...")
        
        try:
            # Step 1: Check Laragon is running
            if not self.check_laragon_running():
                logger.error("❌ Please start Laragon first!")
                return False
                
            # Step 2: Create database
            if not self.create_database():
                return False
                
            # Step 3: Create tables
            if not self.create_tables():
                return False
                
            # Step 4: Create admin user
            if not self.create_admin_user():
                return False
                
            # Step 5: Insert sample data
            if not self.insert_sample_data():
                return False
                
            # Step 6: Create .env file
            if not self.create_env_file():
                return False
                
            # Step 7: Verify setup
            if not self.verify_setup():
                return False
                
            logger.info("\n🎉 DATABASE SETUP COMPLETED SUCCESSFULLY!")
            logger.info("📋 Setup Summary:")
            logger.info(f"   Database: {self.database}")
            logger.info("   Admin Login: admin@example.com / admin123")
            logger.info("   Access: http://localhost/phpmyadmin (via Laragon)")
            logger.info("   Backend API: Ready to start with 'python app.py'")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
        finally:
            self.close_connection()

def main():
    """Main function"""
    print("=" * 60)
    print("🗄️  LARAGON MYSQL DATABASE SETUP")
    print("   Employee Stress Analysis System")
    print("=" * 60)
    
    # Install required packages
    try:
        import bcrypt
    except ImportError:
        logger.info("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bcrypt"])
        import bcrypt
        
    # Run setup
    setup = LaragonDBSetup()
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed! You can now:")
        print("   1. Start the backend: python app.py")
        print("   2. Run the Flutter app")
        print("   3. Access phpMyAdmin at http://localhost/phpmyadmin")
        return 0
    else:
        print("\n❌ Setup failed! Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 