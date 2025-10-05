from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
from pathlib import Path

app = FastAPI(title="Speech-to-Text API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded.")

@app.get("/")
async def root():
    return {"message": "Speech-to-Text API is running"}

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        print(f"Transcribing audio file: {temp_audio_path}")
        result = model.transcribe(temp_audio_path, fp16=False)

        os.unlink(temp_audio_path)
        return JSONResponse(content={
            "text": result["text"].strip(),
            "language": result.get("language", "unknown")
        })
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)