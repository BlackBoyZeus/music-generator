from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt
from generate import generate_song
from loguru import logger
import os
from dotenv import load_dotenv
import uvicorn

app = FastAPI()
load_dotenv()
LOG_DIR = os.getenv('LOG_DIR')
LOG_FILE = os.path.join(LOG_DIR, "api.log")
logger.add(LOG_FILE, rotation="500 MB")

SECRET_KEY = os.getenv("SECRET_KEY", "wXr7862875dxFIi3iTgXCxejaHv5B9UVGHZYNBRFVSg=")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"Valid token for user: {payload.get('user')}")
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/generate_song")
async def api_generate_song(prompt: str = "happy pop song in C major", tempo: int = 120, key: str = "C", mode: str = "major", style: str = "pop", token: dict = Depends(verify_token)):
    try:
        output_file = os.path.join(os.getenv('OUTPUT_DIR'), f"generated_song_{token.get('user')}.wav")
        generate_song(prompt=prompt, output_file=output_file, tempo=tempo, key=key, mode=mode, style=style)
        return {"message": f"Song generated successfully at {output_file}"}
    except Exception as e:
        logger.error(f"Error generating song: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate song")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
