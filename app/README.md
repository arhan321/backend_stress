# Employee Stress Analysis Backend

This is the Python backend for the Employee Stress Analysis application, featuring **Deep Learning** and **Neural Collaborative Filtering** algorithms for predicting and analyzing employee stress levels.

## 🎯 Features

- **Deep Learning Model**: TensorFlow-based neural network for stress level prediction
- **Neural Collaborative Filtering**: Pattern analysis and factor correlation identification  
- **MySQL Integration**: Complete database integration with XAMPP
- **Real-time Processing**: Handle large datasets (3000+ records) efficiently
- **REST API**: Complete API for Flutter app integration
- **Data Validation**: Comprehensive dataset validation and error handling

## 🏗️ Architecture

```
backend/
├── app.py                  # Main Flask application
├── setup.py               # Setup and initialization script
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
├── config/
│   └── database.py        # Database configuration and connection
├── models/
│   ├── database_models.py # SQLAlchemy database models
│   └── ml_models.py       # Deep Learning and NCF implementations
├── uploads/               # Dataset upload directory
├── data/
│   └── samples/          # Sample datasets
├── models/               # Trained ML models storage
└── logs/                # Application logs
```

## 🚀 Quick Start

### Prerequisites

1. **XAMPP** installed and running
   - MySQL service must be active
   - Default MySQL settings (host: localhost, port: 3306, user: root, password: empty)

2. **Python 3.8+** installed

### Installation

1. **Clone or navigate to the backend directory**

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start XAMPP and ensure MySQL is running**
   - Open XAMPP Control Panel
   - Start Apache and MySQL services
   - Verify MySQL is accessible on port 3306

4. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Create the MySQL database (`stress_analysis`)
   - Create all necessary tables
   - Generate sample data (500 employee records)
   - Test ML models
   - Create required directories

5. **Start the API server**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify if needed:

```env
# Database Configuration (XAMPP defaults)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=stress_analysis

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True
```

### Database Schema

The system automatically creates these tables:

- **employees**: Employee data and stress factors
- **datasets**: Uploaded dataset information
- **analysis_results**: ML analysis results
- **stress_predictions**: Individual predictions
- **model_training**: Training history and metrics
- **system_logs**: Application logs

## 📊 Machine Learning Models

### 1. Deep Learning Model (TensorFlow)

**Architecture:**
- Input layer: 6 features (workload, work-life balance, team conflict, management support, work environment, department)
- Hidden layers: 128 → 64 → 32 neurons with ReLU activation
- Batch normalization and dropout for regularization
- Output: 3-class classification (Low, Medium, High stress)

**Features:**
- Automatic feature scaling
- Early stopping and learning rate reduction
- Model versioning and persistence
- Feature importance calculation

### 2. Neural Collaborative Filtering (PyTorch)

**Purpose:**
- Identify patterns between employees and stress factors
- Predict factor correlations and recommendations
- Employee similarity analysis

**Architecture:**
- Employee and factor embeddings (32 dimensions)
- Multi-layer perceptron with dropout
- Matrix factorization for pattern discovery

## 🌐 API Endpoints

### Health Check
```http
GET /api/health
```
Returns API status and component health.

### Dataset Upload
```http
POST /api/upload-dataset
Content-Type: multipart/form-data

Parameters:
- file: CSV/Excel file with employee data
- dataset_name: (optional) Name for the dataset
- description: (optional) Dataset description
```

**Required CSV columns:**
- `employee_id`: Unique employee identifier
- `department`: Employee department
- `workload`: Scale 1-10
- `work_life_balance`: Scale 1-10
- `team_conflict`: Scale 1-10
- `management_support`: Scale 1-10
- `work_environment`: Scale 1-10
- `stress_level`: Scale 0-100

### Dataset Analysis
```http
POST /api/analyze-dataset/{dataset_id}
```
Analyzes dataset using both ML models and statistical methods.

### Stress Prediction
```http
POST /api/predict-stress
Content-Type: application/json

{
  "workload": 8.0,
  "work_life_balance": 5.0,
  "team_conflict": 6.0,
  "management_support": 4.0,
  "work_environment": 7.0,
  "department": "IT",
  "employee_id": "EMP001" // optional
}
```

### Get Datasets
```http
GET /api/datasets
```
Returns list of uploaded datasets with analysis status.

### Get Analysis Results
```http
GET /api/analysis-results/{dataset_id}
```
Returns detailed analysis results for a specific dataset.

## 📈 Sample Data

The setup script generates a realistic dataset with:

- **500 employees** across 5 departments
- **Correlated stress factors** (higher workload → higher stress)
- **Department-specific patterns** (Finance tends to have higher stress)
- **Realistic age and position distributions**

Sample employee record:
```csv
employee_id,department,position,age,workload,work_life_balance,team_conflict,management_support,work_environment,stress_level
EMP0001,IT,Senior,34,7.8,6.2,4.1,5.9,7.3,58.4
```

## 🧪 Testing the API

### Using curl

1. **Health check:**
   ```bash
   curl http://localhost:5000/api/health
   ```

2. **Upload sample dataset:**
   ```bash
   curl -X POST \
     -F "file=@data/samples/employee_stress_data_500.csv" \
     -F "dataset_name=Test Dataset" \
     http://localhost:5000/api/upload-dataset
   ```

3. **Analyze dataset (use dataset_id from upload response):**
   ```bash
   curl -X POST http://localhost:5000/api/analyze-dataset/1
   ```

4. **Predict stress:**
   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"workload":8,"work_life_balance":5,"team_conflict":6,"management_support":4,"work_environment":7,"department":"IT"}' \
     http://localhost:5000/api/predict-stress
   ```

## 🔍 Monitoring and Logging

- **Application logs**: `logs/stress_analysis.log`
- **Setup logs**: `logs/setup.log`
- **Database logs**: Stored in `system_logs` table
- **Model training metrics**: Stored in `model_training` table

## 🐛 Troubleshooting

### Common Issues

1. **MySQL Connection Failed**
   ```
   Solution: Ensure XAMPP is running and MySQL service is started
   Check MySQL is listening on port 3306
   ```

2. **Import Errors**
   ```
   Solution: Install dependencies with: pip install -r requirements.txt
   Ensure Python 3.8+ is being used
   ```

3. **Out of Memory During Training**
   ```
   Solution: Reduce batch size in model training
   Use smaller dataset for initial testing
   ```

4. **Model Training Fails**
   ```
   Solution: Check dataset format and required columns
   Ensure numeric columns are properly formatted
   Verify no missing values in dataset
   ```

### Database Issues

1. **Access Denied**
   ```sql
   -- Create MySQL user if needed
   CREATE USER 'stress_user'@'localhost' IDENTIFIED BY 'password';
   GRANT ALL PRIVILEGES ON stress_analysis.* TO 'stress_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

2. **Table Creation Fails**
   ```
   Solution: Check MySQL version compatibility
   Ensure proper permissions for database creation
   ```

## 🚀 Performance Optimization

### For Large Datasets (3000+ records)

1. **Database Optimization**
   - Indexes are automatically created on frequently queried columns
   - Use connection pooling for high concurrency
   - Consider MySQL query optimization

2. **ML Model Optimization**
   - Batch processing for large prediction requests
   - Model caching to avoid repeated loading
   - GPU acceleration (set ENABLE_GPU=True in .env)

3. **API Optimization**
   - Implement request queuing for analysis jobs
   - Add caching for frequently requested data
   - Use async processing for long-running tasks

## 🔗 Integration with Flutter App

The API is designed to work seamlessly with the Flutter frontend:

1. **Dataset Upload**: Flutter app uploads CSV files via multipart form
2. **Real-time Analysis**: Flutter polls analysis status during processing
3. **Live Updates**: API provides real-time stress predictions
4. **Error Handling**: Comprehensive error responses for Flutter UI

## 📚 Additional Resources

- **TensorFlow Documentation**: https://tensorflow.org/guide
- **PyTorch Documentation**: https://pytorch.org/docs
- **Flask API Development**: https://flask.palletsprojects.com/
- **SQLAlchemy ORM**: https://docs.sqlalchemy.org/
- **XAMPP Setup Guide**: https://www.apachefriends.org/

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs in `logs/` directory
3. Verify XAMPP MySQL service is running
4. Ensure all dependencies are installed correctly

---

**Ready to analyze employee stress with advanced machine learning! 🚀** 