# Use slim Python image to keep the container small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifacts
COPY main.py .
COPY models/ ./models/

# Expose port 8080 (AWS Elastic Beanstalk default)
EXPOSE 8080

# Start the FastAPI app with uvicorn
# - host 0.0.0.0 makes it reachable outside the container
# - port 8080 matches the EXPOSE above
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
