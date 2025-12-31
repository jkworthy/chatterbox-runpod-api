#!/bin/bash
set -e

echo "=== Starting Chatterbox RunPod API ==="

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Clone and install official Chatterbox
echo "Installing Chatterbox..."
rm -rf chatterbox
git clone https://github.com/resemble-ai/chatterbox.git chatterbox
pip install -e ./chatterbox

# Create output directory
mkdir -p /workspace/audio_output

# Start the FastAPI server
echo "Starting FastAPI server on port 8000..."
python -u app.py
