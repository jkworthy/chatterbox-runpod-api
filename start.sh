#!/bin/bash
# Startup script for RunPod Chatterbox TTS API

set -e  # Exit on error

echo "🚀 Starting Chatterbox TTS API setup..."

# Clone API service repo
echo "📥 Cloning API service repository..."
git clone https://github.com/jkworthy/chatterbox-runpod-api.git /workspace || {
    echo "⚠️  Repository already exists, continuing..."
}

cd /workspace

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Clone and install Chatterbox
echo "📥 Cloning Chatterbox..."
if [ ! -d "/workspace/chatterbox" ]; then
    git clone https://github.com/chenxwh/chatterbox.git /workspace/chatterbox
fi

cd /workspace
echo "📦 Installing Chatterbox..."
pip install -e ./chatterbox

# Start the API service
echo "🎤 Starting Chatterbox TTS API..."
python app.py

