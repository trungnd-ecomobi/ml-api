# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "cicd2_webservice:app", "--host", "0.0.0.0", "--port", "8000"]