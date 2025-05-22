from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse
import jwt
from generate import generate_song
from generate_muzic import generate_song_with_muzic
from loguru import logger
import os
import uuid
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
load_dotenv()
LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")
logger.add(LOG_FILE, rotation="500 MB")

SECRET_KEY = os.getenv("SECRET_KEY", "wXr7862875dxFIi3iTgXCxejaHv5B9UVGHZYNBRFVSg=")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create output directory if it doesn't exist
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store job status
job_status = {}

class SongRequest(BaseModel):
    prompt: str = "happy rap song in C major"
    tempo: int = 120
    key: str = "C"
    mode: str = "major"
    style: str = "rap"
    use_muzic: bool = True

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"Valid token for user: {payload.get('user')}")
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/generate_song")
async def api_generate_song(
    request: SongRequest,
    background_tasks: BackgroundTasks,
    token: dict = Depends(verify_token)
):
    try:
        job_id = str(uuid.uuid4())
        output_file = os.path.join(OUTPUT_DIR, f"generated_song_{job_id}.wav")
        
        # Update job status
        job_status[job_id] = {"status": "processing"}
        
        # Start generation in background task
        if request.use_muzic:
            background_tasks.add_task(
                generate_song_with_muzic,
                prompt=request.prompt,
                output_file=output_file,
                tempo=request.tempo,
                key=request.key,
                mode=request.mode,
                style=request.style
            )
        else:
            background_tasks.add_task(
                generate_song,
                prompt=request.prompt,
                output_file=output_file,
                tempo=request.tempo,
                key=request.key,
                mode=request.mode,
                style=request.style
            )
        
        return {
            "message": "Song generation started",
            "job_id": job_id,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error generating song: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate song")

@app.get("/songs/{job_id}")
async def get_song_status(job_id: str, token: dict = Depends(verify_token)):
    try:
        output_file = os.path.join(OUTPUT_DIR, f"generated_song_{job_id}.wav")
        
        if os.path.exists(output_file):
            # Update job status
            job_status[job_id] = {"status": "completed"}
            
            return {
                "job_id": job_id,
                "status": "completed",
                "download_url": f"/songs/{job_id}/download"
            }
        else:
            # Check if job is in progress
            if job_id in job_status and job_status[job_id]["status"] == "processing":
                return {
                    "job_id": job_id,
                    "status": "processing"
                }
            else:
                raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking song status: {e}")
        raise HTTPException(status_code=500, detail="Failed to check song status")

@app.get("/songs/{job_id}/download")
async def download_song(job_id: str, token: dict = Depends(verify_token)):
    try:
        output_file = os.path.join(OUTPUT_DIR, f"generated_song_{job_id}.wav")
        
        if os.path.exists(output_file):
            return FileResponse(output_file, media_type="audio/wav", filename=f"song_{job_id}.wav")
        else:
            raise HTTPException(status_code=404, detail="Song not found")
    except Exception as e:
        logger.error(f"Error downloading song: {e}")
        raise HTTPException(status_code=500, detail="Failed to download song")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
