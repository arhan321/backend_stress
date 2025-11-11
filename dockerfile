FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu (biar caching lebih efisien)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code ke dalam container
COPY ./app /app

# Expose Flask port
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]
