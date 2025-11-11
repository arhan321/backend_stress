#!/usr/bin/env python3
"""
Final Database Setup for Laragon MySQL
Clean and working version
"""

import mysql.connector
import bcrypt
import sys
import subprocess

def setup_laragon_database():
    """Setup complete database for Laragon"""
    
    print("=" * 60)
    print("🗄️  SETUP DATABASE LARAGON MYSQL")
    print("   Employee Stress Analysis System")
    print("=" * 60)
    
    try:
        # Install bcrypt if needed
        try:
            import bcrypt
        except ImportError:
            print("📦 Installing bcrypt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bcrypt"])
            import bcrypt
        
        # Connect to MySQL
        print("🔗 Connecting to Laragon MySQL...")
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',
            autocommit=True
        )
        
        cursor = connection.cursor()
        print("✅ Connected to MySQL")
        
        # Step 1: Create Database
        print("\n📦 Creating database...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS stress_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute("USE stress_analysis")
        print("✅ Database 'stress_analysis' ready")
        
        # Step 2: Create Tables
        print("\n📋 Creating tables...")
        
        # Employees table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id VARCHAR(50) UNIQUE NOT NULL,
            department VARCHAR(100) NOT NULL,
            age INT,
            position VARCHAR(200),
            workload DECIMAL(3,1) NOT NULL,
            work_life_balance DECIMAL(3,1) NOT NULL,
            team_conflict DECIMAL(3,1) NOT NULL,
            management_support DECIMAL(3,1) NOT NULL,
            work_environment DECIMAL(3,1) NOT NULL,
            stress_level DECIMAL(5,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: employees")
        
        # Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100) NOT NULL,
            role ENUM('admin','employee') DEFAULT 'employee',
            department VARCHAR(100),
            position VARCHAR(200),
            phone_number VARCHAR(20),
            profile_picture VARCHAR(500),
            is_active TINYINT(1) DEFAULT 1,
            email_verified TINYINT(1) DEFAULT 0,
            last_login TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            reset_token VARCHAR(255),
            reset_token_expires TIMESTAMP NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: users")
        
        # Datasets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            file_path VARCHAR(500),
            file_size INT,
            record_count INT NOT NULL DEFAULT 0,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'uploaded',
            processing_start_time TIMESTAMP NULL,
            processing_end_time TIMESTAMP NULL,
            error_message TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: datasets")
        
        # Analysis results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            dataset_id INT NOT NULL,
            analysis_type VARCHAR(100) NOT NULL,
            overall_stress_level DECIMAL(5,2) NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            factor_contributions JSON,
            model_accuracy DECIMAL(5,4),
            confidence_score DECIMAL(5,4),
            department_breakdown JSON,
            correlation_matrix JSON,
            feature_importance JSON,
            processing_time_seconds DECIMAL(8,2),
            INDEX idx_dataset_id (dataset_id),
            INDEX idx_analysis_date (analysis_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: analysis_results")
        
        # User permissions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_permissions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            permission VARCHAR(100) NOT NULL,
            granted_by INT,
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            INDEX idx_user_id (user_id),
            INDEX idx_permission (permission)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: user_permissions")
        
        # System logs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            level VARCHAR(20) NOT NULL,
            module VARCHAR(100) NOT NULL,
            message TEXT NOT NULL,
            additional_data JSON,
            INDEX idx_timestamp (timestamp),
            INDEX idx_level (level)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        print("✅ Table: system_logs")
        
        # Step 3: Create Admin User
        print("\n👤 Creating admin user...")
        
        # Check if admin user exists
        cursor.execute("SELECT id FROM users WHERE email = 'admin@example.com'")
        existing_admin = cursor.fetchone()
        
        if not existing_admin:
            # Hash password
            password = "admin123"
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert admin user
            cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role, department, position, is_active, email_verified) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                'admin',
                'admin@example.com', 
                hashed_password,
                'System Administrator',
                'admin',
                'IT',
                'Administrator',
                1,
                1
            ))
            admin_id = cursor.lastrowid
            print("✅ Admin user created")
        else:
            admin_id = existing_admin[0]
            print("✅ Admin user already exists")
        
        # Add permissions
        permissions = ['admin_access', 'user_management', 'system_config', 'data_analysis', 'report_generation']
        for permission in permissions:
            cursor.execute("""
            INSERT IGNORE INTO user_permissions (user_id, permission) 
            VALUES (%s, %s)
            """, (admin_id, permission))
        print("✅ Admin permissions added")
        
        # Step 4: Add Sample Data
        print("\n📊 Adding sample data...")
        
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
        
        for emp in sample_employees:
            cursor.execute("""
            INSERT IGNORE INTO employees (employee_id, department, age, position, workload, work_life_balance, 
                                        team_conflict, management_support, work_environment, stress_level) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, emp)
        print(f"✅ Added {len(sample_employees)} sample employees")
        
        # Add sample dataset
        cursor.execute("""
        INSERT IGNORE INTO datasets (name, description, file_path, record_count, status) 
        VALUES (%s, %s, %s, %s, %s)
        """, (
            'Sample Employee Data',
            'Initial sample dataset for testing',
            'data/samples/employee_stress_data_500.csv',
            len(sample_employees),
            'processed'
        ))
        print("✅ Sample dataset added")
        
        # Step 5: Create .env file
        print("\n⚙️ Creating configuration...")
        
        env_content = """# Database Configuration for Laragon
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=stress_analysis

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-change-in-production

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
JWT_ACCESS_TOKEN_EXPIRES=3600

# File Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216

# Logging
LOG_LEVEL=INFO
"""
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ .env file created")
        
        # Step 6: Verify Setup
        print("\n🔍 Verifying setup...")
        
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"✅ Tables created: {len(tables)}")
        
        cursor.execute("SELECT COUNT(*) FROM employees")
        emp_count = cursor.fetchone()[0]
        print(f"✅ Employee records: {emp_count}")
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        admin_count = cursor.fetchone()[0]
        print(f"✅ Admin users: {admin_count}")
        
        # Log setup completion
        cursor.execute("""
        INSERT INTO system_logs (level, module, message, additional_data) 
        VALUES (%s, %s, %s, %s)
        """, (
            'INFO',
            'SETUP',
            'Database setup completed successfully',
            '{"tables": ' + str(len(tables)) + ', "employees": ' + str(emp_count) + '}'
        ))
        
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📋 SUMMARY:")
        print(f"   Database: stress_analysis")
        print(f"   Tables: {len(tables)}")
        print(f"   Sample Data: {emp_count} employees")
        print(f"   Admin Login: admin@example.com / admin123")
        print("   phpMyAdmin: http://localhost/phpmyadmin")
        print("   Backend: python app.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Pastikan Laragon sudah berjalan")
        print("2. Start MySQL service di Laragon")
        print("3. Periksa port 3306 tidak digunakan aplikasi lain")
        return False
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
            print("✅ Connection closed")

if __name__ == "__main__":
    success = setup_laragon_database()
    
    if success:
        print("\n✅ Setup berhasil! Langkah selanjutnya:")
        print("   1. cd backend")
        print("   2. python app.py")
        print("   3. Test Flutter app")
    else:
        print("\n❌ Setup gagal! Periksa error di atas.")
    
    print("\nTekan Enter untuk keluar...")
    input() 