#!/usr/bin/env python3
"""
Quick MySQL Database Setup for Laragon
Simple and robust setup script
"""

import mysql.connector
import os

def setup_database():
    """Setup database for Laragon MySQL"""
    
    print("🚀 Setup Database MySQL untuk Laragon...")
    
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            user='root',
            password=''
        )
        
        cursor = connection.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS stress_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute("USE stress_analysis")
        print("✅ Database 'stress_analysis' dibuat")
        
        # Create basic tables
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✅ Table employees dibuat")
        
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
            is_active TINYINT(1) DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("✅ Table users dibuat")
        
        connection.commit()
        print("🎉 Setup selesai!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    setup_database() 