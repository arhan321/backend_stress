#!/usr/bin/env python3
"""
Comprehensive test script for admin CRUD operations
"""

import requests
import json

BASE_URL = 'http://localhost:5000'

def test_admin_crud():
    print('🔐 Testing Admin CRUD Operations')
    print('=' * 50)
    
    # Step 1: Admin login
    print('\n1. 🔑 Admin Login...')
    login_data = {
        'username_or_email': 'admin@example.com',
        'password': 'admin123'
    }
    
    try:
        response = requests.post(f'{BASE_URL}/api/auth/login', json=login_data)
        
        if response.status_code != 200:
            print(f'❌ Login failed: {response.text}')
            return
        
        login_result = response.json()
        token = login_result['token']
        headers = {'Authorization': f'Bearer {token}'}
        
        print('✅ Admin login successful')
        print(f'   Admin role: {login_result["user"]["role"]}')
        
        # Step 2: Test Users API
        print('\n2. 👥 Testing Users Management...')
        
        # Get all users
        users_response = requests.get(f'{BASE_URL}/api/admin/users', headers=headers)
        if users_response.status_code == 200:
            users_data = users_response.json()
            print(f'✅ Get Users: Found {len(users_data["users"])} users')
            
            users = users_data['users']
            test_user = None
            
            # Find a non-admin user for testing
            for user in users:
                if user['role'] == 'employee':
                    test_user = user
                    break
            
            if test_user:
                user_id = test_user['id']
                print(f'   📄 Testing with user: {test_user["full_name"]} (ID: {user_id})')
                
                # Test edit user
                edit_data = {
                    'full_name': f'{test_user["full_name"]} (Updated)',
                    'department': 'Updated Department',
                    'position': 'Updated Position'
                }
                
                edit_response = requests.put(
                    f'{BASE_URL}/api/admin/users/{user_id}/edit',
                    headers=headers,
                    json=edit_data
                )
                
                if edit_response.status_code == 200:
                    print('✅ Edit User: Success')
                else:
                    print(f'❌ Edit User: {edit_response.text}')
                
                # Test user datasets
                datasets_response = requests.get(
                    f'{BASE_URL}/api/admin/users/{user_id}/datasets',
                    headers=headers
                )
                
                if datasets_response.status_code == 200:
                    datasets_data = datasets_response.json()
                    dataset_count = len(datasets_data['datasets'])
                    print(f'✅ User Datasets: Found {dataset_count} datasets for user')
                else:
                    print(f'❌ User Datasets: {datasets_response.text}')
                
                # Test reset password
                reset_data = {'new_password': 'newpassword123'}
                reset_response = requests.post(
                    f'{BASE_URL}/api/admin/users/{user_id}/reset-password',
                    headers=headers,
                    json=reset_data
                )
                
                if reset_response.status_code == 200:
                    print('✅ Reset Password: Success')
                else:
                    print(f'❌ Reset Password: {reset_response.text}')
                
                # Test toggle user status (deactivate then reactivate)
                toggle_response = requests.post(
                    f'{BASE_URL}/api/admin/users/{user_id}/toggle-status',
                    headers=headers
                )
                
                if toggle_response.status_code == 200:
                    toggle_data = toggle_response.json()
                    new_status = toggle_data['user']['is_active']
                    print(f'✅ Toggle Status: User is now {"active" if new_status else "inactive"}')
                    
                    # Toggle back to original status
                    toggle_back_response = requests.post(
                        f'{BASE_URL}/api/admin/users/{user_id}/toggle-status',
                        headers=headers
                    )
                    if toggle_back_response.status_code == 200:
                        print('✅ Toggle Status Back: Success')
                else:
                    print(f'❌ Toggle Status: {toggle_response.text}')
        
        else:
            print(f'❌ Get Users: {users_response.text}')
        
        # Step 3: Test Datasets API
        print('\n3. 📁 Testing Datasets Management...')
        
        # Get all datasets
        datasets_response = requests.get(f'{BASE_URL}/api/admin/datasets', headers=headers)
        if datasets_response.status_code == 200:
            datasets_data = datasets_response.json()
            datasets = datasets_data['datasets']
            print(f'✅ Get Datasets: Found {len(datasets)} datasets')
            
            if datasets:
                # Show dataset details
                for i, dataset in enumerate(datasets[:3]):  # Show first 3
                    uploader = dataset.get('uploader', {})
                    print(f'   📄 Dataset {i+1}: {dataset["name"]} by {uploader.get("full_name", "Unknown")} ({dataset["status"]})')
                
                # Test delete dataset (only if there are test datasets)
                test_dataset = None
                for dataset in datasets:
                    if 'test' in dataset['name'].lower() or dataset['name'] in ['200', '500']:
                        test_dataset = dataset
                        break
                
                if test_dataset:
                    dataset_id = test_dataset['id']
                    dataset_name = test_dataset['name']
                    
                    print(f'   🗑️  Testing delete with dataset: {dataset_name} (ID: {dataset_id})')
                    
                    delete_response = requests.delete(
                        f'{BASE_URL}/api/admin/datasets/{dataset_id}',
                        headers=headers
                    )
                    
                    if delete_response.status_code == 200:
                        print(f'✅ Delete Dataset: "{dataset_name}" deleted successfully')
                    else:
                        print(f'❌ Delete Dataset: {delete_response.text}')
                else:
                    print('   ℹ️  No test datasets found to delete')
        else:
            print(f'❌ Get Datasets: {datasets_response.text}')
        
        # Step 4: Test Dashboard Stats
        print('\n4. 📊 Testing Dashboard Stats...')
        
        stats_response = requests.get(f'{BASE_URL}/api/admin/dashboard/stats', headers=headers)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            stats = stats_data['stats']
            
            print('✅ Dashboard Stats: Success')
            print(f'   👥 Users: {stats["users"]["total"]} total, {stats["users"]["active"]} active')
            print(f'   📁 Datasets: {stats["datasets"]["total"]} total, {stats["datasets"]["processed"]} processed')
            print(f'   📈 Success Rate: {stats["datasets"]["success_rate"]:.1f}%')
        else:
            print(f'❌ Dashboard Stats: {stats_response.text}')
        
        print('\n' + '=' * 50)
        print('🎉 Admin CRUD Test Completed!')
        
    except requests.exceptions.ConnectionError:
        print('❌ Connection failed - Make sure Flask server is running on localhost:5000')
    except Exception as e:
        print(f'❌ Error: {str(e)}')

if __name__ == '__main__':
    test_admin_crud() 