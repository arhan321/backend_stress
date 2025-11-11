#!/usr/bin/env python3
"""
Test admin API endpoints
"""

import requests
import json

def test_admin_api():
    print('🔐 Testing admin login...')
    
    # Test admin login first
    login_data = {
        'username_or_email': 'admin@example.com',
        'password': 'admin123'
    }
    
    try:
        response = requests.post('http://localhost:5000/api/auth/login', json=login_data)
        print(f'Login Status: {response.status_code}')
        
        if response.status_code == 200:
            login_result = response.json()
            token = login_result['token']
            print('✅ Admin login successful')
            print(f'User role: {login_result["user"]["role"]}')
            
            # Test admin users endpoint
            print('\n📊 Testing admin users endpoint...')
            headers = {'Authorization': f'Bearer {token}'}
            users_response = requests.get('http://localhost:5000/api/admin/users', headers=headers)
            print(f'Users Status: {users_response.status_code}')
            
            if users_response.status_code == 200:
                users_data = users_response.json()
                print('✅ Admin users endpoint working')
                print(f'Found {len(users_data.get("users", []))} users')
                
                for user in users_data.get('users', []):
                    print(f'  👤 {user["full_name"]} ({user["email"]}) - {user["role"]} - Datasets: {user.get("dataset_count", 0)}')
            else:
                print(f'❌ Users endpoint failed: {users_response.text}')
            
            # Test admin datasets endpoint
            print('\n📁 Testing admin datasets endpoint...')
            datasets_response = requests.get('http://localhost:5000/api/admin/datasets', headers=headers)
            print(f'Datasets Status: {datasets_response.status_code}')
            
            if datasets_response.status_code == 200:
                datasets_data = datasets_response.json()
                print('✅ Admin datasets endpoint working')
                print(f'Found {len(datasets_data.get("datasets", []))} datasets')
                
                for dataset in datasets_data.get('datasets', []):
                    uploader_name = dataset.get('uploader', {}).get('full_name', 'Unknown')
                    print(f'  📄 {dataset["name"]} by {uploader_name} - Status: {dataset["status"]}')
            else:
                print(f'❌ Datasets endpoint failed: {datasets_response.text}')
            
            # Test user datasets endpoint for tes2 (user_id 3)
            print('\n👤 Testing user datasets endpoint for tes2...')
            user_datasets_response = requests.get('http://localhost:5000/api/admin/users/3/datasets', headers=headers)
            print(f'User Datasets Status: {user_datasets_response.status_code}')
            
            if user_datasets_response.status_code == 200:
                user_datasets_data = user_datasets_response.json()
                print('✅ User datasets endpoint working')
                print(f'User: {user_datasets_data.get("user", {}).get("full_name", "Unknown")}')
                print(f'Found {len(user_datasets_data.get("datasets", []))} datasets for this user')
            else:
                print(f'❌ User datasets endpoint failed: {user_datasets_response.text}')
        
        else:
            print(f'❌ Login failed: {response.text}')
    
    except requests.exceptions.ConnectionError:
        print('❌ Connection failed - Make sure Flask server is running on localhost:5000')
    except Exception as e:
        print(f'❌ Error: {str(e)}')

if __name__ == '__main__':
    test_admin_api() 