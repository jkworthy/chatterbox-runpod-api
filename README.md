# Chatterbox TTS RunPod API Service

FastAPI service for processing text chunks into MP3 audio files using Chatterbox TTS on RunPod GPU instances.

## Setup Instructions

See `/Users/kylewidner/Documents/zzy-scripts/RUNPOD-SETUP-GUIDE.md` for complete step-by-step setup.

## Files

- `app.py` - FastAPI service with endpoints for voice upload and audio generation
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image configuration
- `.dockerignore` - Files to exclude from Docker build

## API Endpoints

### Health Check
- `GET /` - Basic status
- `GET /health` - Detailed health check with GPU info

### Voice Management
- `POST /upload-voice` - Upload voice file for use in TTS

### Audio Generation
- `POST /generate` - Generate single MP3 from text
- `POST /generate-batch` - Generate multiple MP3s in batch
- `GET /download/{chunk_id}` - Download generated MP3 file

## Usage

Once deployed on RunPod, use the Mac script:
`7200-create-mp3-chunks-chatterbox-runpod.py`

## Notes

- Model loads on startup (takes 1-2 minutes)
- Voice files are stored temporarily on the pod
- Generated MP3s are stored in `/workspace/audio_output`
- Service runs on port 8000

