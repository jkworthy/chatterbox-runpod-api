#!/usr/bin/env python3
"""
Chatterbox TTS API Service for RunPod
FastAPI service that processes text chunks into MP3 audio files.
"""

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import List, Optional
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydub import AudioSegment
from pydub.effects import normalize
import uvicorn

# Import Chatterbox
from chatterbox.tts import ChatterboxTTS

app = FastAPI(title="Chatterbox TTS API")

# Global model (loaded once on startup)
model = None
model_loaded = False

# Storage for voice files and processed audio
VOICE_STORAGE = Path("/workspace/voices")
AUDIO_STORAGE = Path("/workspace/audio_output")
VOICE_STORAGE.mkdir(exist_ok=True)
AUDIO_STORAGE.mkdir(exist_ok=True)

@app.on_event("startup")
async def load_model():
    """Load Chatterbox model on startup."""
    global model, model_loaded
    
    print("🔄 Loading Chatterbox model...")
    try:
        import torch
        # Determine device (CUDA for NVIDIA GPU, CPU fallback)
        if torch.cuda.is_available():
            device = "cuda"
            print(f"📱 Using device: CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            print("📱 Using device: CPU (no GPU acceleration)")
        
        # Load model
        model = ChatterboxTTS.from_pretrained(device=device)
        model_loaded = True
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": model_loaded,
        "service": "Chatterbox TTS API"
    }

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "gpu_available": torch.cuda.is_available() if torch else False,
        "gpu_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    }

@app.post("/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    voice_name: str = Form(...)
):
    """
    Upload a voice file for use in TTS generation.
    Returns voice_id for use in generate requests.
    """
    try:
        # Save voice file
        voice_path = VOICE_STORAGE / f"{voice_name}.wav"
        with open(voice_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "voice_id": voice_name,
            "message": f"Voice file uploaded: {voice_name}.wav"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

@app.post("/generate")
async def generate_audio(
    text: str = Form(...),
    voice_id: str = Form(...),
    chunk_id: str = Form(...)
):
    """
    Generate MP3 audio from text using specified voice.
    
    Parameters:
    - text: Text to convert to speech
    - voice_id: Name of voice file (must be uploaded first)
    - chunk_id: Unique identifier for this chunk (for tracking)
    
    Returns:
    - Base64-encoded MP3 file
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Find voice file
        voice_path = VOICE_STORAGE / f"{voice_id}.wav"
        if not voice_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice file not found: {voice_id}")
        
        # Generate audio
        print(f"🎤 Generating audio for chunk: {chunk_id} ({len(text)} chars)")
        wav = model.generate(text, audio_prompt_path=str(voice_path))
        
        if wav is None or wav.numel() == 0:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Save as WAV first
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        ta.save(temp_wav.name, wav, model.sr)
        
        # Convert to MP3
        audio = AudioSegment.from_wav(temp_wav.name)
        audio = normalize(audio)  # Normalize volume
        
        # Save MP3
        output_path = AUDIO_STORAGE / f"{chunk_id}.mp3"
        audio.export(str(output_path), format='mp3', parameters=["-ar", "24000", "-ac", "1", "-q:a", "2"])
        
        # Clean up temp WAV
        os.unlink(temp_wav.name)
        
        # Read MP3 and encode as base64
        with open(output_path, "rb") as f:
            mp3_data = f.read()
        
        mp3_base64 = base64.b64encode(mp3_data).decode('utf-8')
        
        return {
            "status": "success",
            "chunk_id": chunk_id,
            "audio_base64": mp3_base64,
            "file_size": len(mp3_data)
        }
        
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@app.post("/generate-batch")
async def generate_batch(
    chunks: List[dict]
):
    """
    Generate multiple audio chunks in batch.
    
    Request body should be:
    {
        "chunks": [
            {"text": "...", "voice_id": "...", "chunk_id": "..."},
            ...
        ]
    }
    
    Returns list of results (may take time for large batches).
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for i, chunk in enumerate(chunks):
        try:
            text = chunk.get("text")
            voice_id = chunk.get("voice_id")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            
            # Use the single generate endpoint logic
            voice_path = VOICE_STORAGE / f"{voice_id}.wav"
            if not voice_path.exists():
                results.append({
                    "chunk_id": chunk_id,
                    "status": "error",
                    "error": f"Voice file not found: {voice_id}"
                })
                continue
            
            print(f"🎤 [{i+1}/{len(chunks)}] Generating: {chunk_id}")
            wav = model.generate(text, audio_prompt_path=str(voice_path))
            
            if wav is None or wav.numel() == 0:
                results.append({
                    "chunk_id": chunk_id,
                    "status": "error",
                    "error": "Failed to generate audio"
                })
                continue
            
            # Save as WAV first
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            ta.save(temp_wav.name, wav, model.sr)
            
            # Convert to MP3
            audio = AudioSegment.from_wav(temp_wav.name)
            audio = normalize(audio)
            
            output_path = AUDIO_STORAGE / f"{chunk_id}.mp3"
            audio.export(str(output_path), format='mp3', parameters=["-ar", "24000", "-ac", "1", "-q:a", "2"])
            
            # Clean up temp WAV
            os.unlink(temp_wav.name)
            
            # Read and encode
            with open(output_path, "rb") as f:
                mp3_data = f.read()
            
            mp3_base64 = base64.b64encode(mp3_data).decode('utf-8')
            
            results.append({
                "chunk_id": chunk_id,
                "status": "success",
                "audio_base64": mp3_base64,
                "file_size": len(mp3_data)
            })
            
        except Exception as e:
            results.append({
                "chunk_id": chunk_id,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "complete",
        "total": len(chunks),
        "results": results
    }

@app.get("/download/{chunk_id}")
async def download_audio(chunk_id: str):
    """Download a generated MP3 file by chunk_id."""
    file_path = AUDIO_STORAGE / f"{chunk_id}.mp3"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=f"{chunk_id}.mp3"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
