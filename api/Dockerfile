FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p /models

# Copy the model from the project if it exists
RUN if [ -f /app/output/temporal_model.pkl ]; then cp /app/output/temporal_model.pkl /models/; fi

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 