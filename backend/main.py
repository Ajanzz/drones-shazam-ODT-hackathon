# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json

# -------- data shape we send back ----------
class InferenceEvent(BaseModel):
    label: str
    score: float
    probs: dict[str, float]
    timestamp: int
    window_sec: float | None = None

app = FastAPI()

# Allow your Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # relax as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- one-shot file upload -------------
@app.post("/predict", response_model=InferenceEvent)
async def predict(file: UploadFile = File(...)):
    # TODO: run trained model on file.file (a file-like object)
    # For now we return a default result
    probs = {"drone": 0.87, "non-drone": 0.13}
    return InferenceEvent(
        label="drone",
        score=probs["drone"],
        probs=probs,
        timestamp=int(time.time() * 1000),
        window_sec=None,
    )

# -------- live streaming via WebSocket -----
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    try:
        # The frontend sends raw binary chunks (WebM/Opus). We would
        # feed each chunk into your streaming classifier here.
        window_sec = 0.3  # one chunk ~300ms
        while True:
            chunk: bytes = await ws.receive_bytes()
            # TODO: push 'chunk' through a streaming model
            # Demo output:
            msg = InferenceEvent(
                label="drone",
                score=0.78,
                probs={"drone": 0.78, "non-drone": 0.22},
                timestamp=int(time.time() * 1000),
                window_sec=window_sec,
            )
            await ws.send_text(msg.json())
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
