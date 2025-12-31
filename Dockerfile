FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Chatterbox installation should be done via git clone in Docker command
# or you can add it here if you have a specific installation method

# Expose API port
EXPOSE 8000

# Run the API service
CMD ["python", "app.py"]

