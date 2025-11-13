# backend/main.py
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
from audio_analyzer import DroneAudioAnalyzer
from itertools import cycle
from typing import Optional

# Toggle dummy data mode for frontend testing
USE_DUMMY_DATA = False

# Initialize audio analyzer only when using real inference
analyzer: Optional[DroneAudioAnalyzer] = None
if not USE_DUMMY_DATA:
    analyzer = DroneAudioAnalyzer()

# -------- data shape we send back ----------
class DroneMetrics(BaseModel):
    """Drone characteristics extracted from audio."""
    speed: Optional[float] = None  # m/s
    distance: Optional[float] = None  # meters
    direction: Optional[str] = None  # "approaching", "receding", "stationary", "unknown"
    flight_state: str = "unknown"  # "hovering", "flying", "transitioning", "unknown"
    has_payload: bool = False
    payload_confidence: float = 0.0

class InferenceEvent(BaseModel):
    label: str
    score: float
    probs: dict[str, float]
    timestamp: int
    window_sec: float | None = None
    metrics: Optional[DroneMetrics] = None  # New: drone characteristics


# -------- dummy data helpers ----------
_DUMMY_PRESETS = [
    {
        "score": 0.96,
        "metrics": dict(
            speed=12.4,
            distance=42.0,
            direction="approaching",
            flight_state="flying",
            has_payload=True,
            payload_confidence=0.82,
        ),
    },
    {
        "score": 0.88,
        "metrics": dict(
            speed=6.8,
            distance=65.5,
            direction="stationary",
            flight_state="hovering",
            has_payload=False,
            payload_confidence=0.18,
        ),
    },
    {
        "score": 0.91,
        "metrics": dict(
            speed=9.7,
            distance=28.3,
            direction="receding",
            flight_state="transitioning",
            has_payload=False,
            payload_confidence=0.27,
        ),
    },
]
_DUMMY_CYCLE = cycle(_DUMMY_PRESETS)


def build_dummy_event(window: float | None = None) -> InferenceEvent:
    preset = next(_DUMMY_CYCLE)
    metrics = DroneMetrics(**preset["metrics"])
    score = preset["score"]
    return InferenceEvent(
        label="drone",
        score=score,
        probs={"drone": score, "non-drone": round(1.0 - score, 2)},
        timestamp=int(time.time() * 1000),
        window_sec=window,
        metrics=metrics,
    )

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
    if USE_DUMMY_DATA:
        # Return preset dummy data to exercise the frontend UI
        await file.read()  # consume upload to keep interface consistent
        return build_dummy_event()

    assert analyzer is not None
    # Read audio file
    audio_data = await file.read()
    file_format = file.filename.split(".")[-1] if file.filename else "wav"

    # Analyze audio for drone characteristics
    metrics_dict = analyzer.analyze_audio(audio_data, format=file_format)

    # Check if it's likely a drone (not ambient noise)
    # Extract features for noise filtering
    import io
    import librosa

    try:
        audio_io = io.BytesIO(audio_data)
        audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
        features = analyzer._extract_features(audio_array, sr)
        is_drone = analyzer.filter_ambient_noise(features)
    except:
        is_drone = True  # Default to true if analysis fails

    # TODO: Integrate your trained AI model here
    # Replace the heuristic-based confidence with your model's prediction:
    #   model_output = your_model.predict(audio_array)
    #   drone_confidence = model_output["drone_probability"]
    #   probs = model_output["class_probabilities"]
    # For now we use the analyzer's confidence as a fallback
    drone_confidence = metrics_dict.get("confidence", 0.5) if is_drone else 0.1
    probs = {
        "drone": drone_confidence,
        "non-drone": 1.0 - drone_confidence,
    }

    # Create metrics object
    if "features" in locals() and features:
        has_payload, payload_conf = analyzer._detect_payload(features)
    else:
        has_payload, payload_conf = False, 0.0

    metrics = DroneMetrics(
        speed=metrics_dict.get("speed"),
        distance=metrics_dict.get("distance"),
        direction=metrics_dict.get("direction"),
        flight_state=metrics_dict.get("flight_state", "unknown"),
        has_payload=has_payload,
        payload_confidence=payload_conf,
    )

    return InferenceEvent(
        label="drone" if drone_confidence > 0.5 else "non-drone",
        score=drone_confidence,
        probs=probs,
        timestamp=int(time.time() * 1000),
        window_sec=None,
        metrics=metrics if is_drone else None,  # Only include metrics if it's a drone
    )

# -------- live streaming via WebSocket -----
@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    if USE_DUMMY_DATA:
        try:
            window_sec = 0.3  # ~300ms chunks
            while True:
                # Wait for a chunk but ignore its contents
                await ws.receive_bytes()
                msg = build_dummy_event(window=window_sec)
                await ws.send_text(msg.json())
        except WebSocketDisconnect:
            return

    # Create a new analyzer instance for this connection to maintain history
    stream_analyzer = DroneAudioAnalyzer()
    
    try:
        # The frontend sends raw binary chunks (WebM/Opus). We would
        # feed each chunk into your streaming classifier here.
        window_sec = 0.3  # one chunk ~300ms
        while True:
            chunk: bytes = await ws.receive_bytes()
            
            # Analyze audio chunk
            metrics_dict = stream_analyzer.analyze_audio(chunk, format="webm")
            
            # Extract features for noise filtering
            try:
                import io
                import librosa
                audio_io = io.BytesIO(chunk)
                audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
                features = stream_analyzer._extract_features(audio_array, sr)
                is_drone = stream_analyzer.filter_ambient_noise(features)
            except:
                is_drone = True
                features = {}
            
            # Calculate probabilities
            drone_confidence = metrics_dict.get("confidence", 0.5) if is_drone else 0.1
            probs = {
                "drone": drone_confidence,
                "non-drone": 1.0 - drone_confidence
            }
            
            # Create metrics
            has_payload, payload_conf = stream_analyzer._detect_payload(features) if features else (False, 0.0)
            metrics = DroneMetrics(
                speed=metrics_dict.get("speed"),
                distance=metrics_dict.get("distance"),
                direction=metrics_dict.get("direction"),
                flight_state=metrics_dict.get("flight_state", "unknown"),
                has_payload=has_payload,
                payload_confidence=payload_conf,
            )
            
            # TODO: Integrate your trained AI model here for streaming
            # Replace the heuristic-based confidence with your model's prediction:
            #   model_output = your_streaming_model.predict(audio_array)
            #   drone_confidence = model_output["drone_probability"]
            #   probs = model_output["class_probabilities"]
            msg = InferenceEvent(
                label="drone" if drone_confidence > 0.5 else "non-drone",
                score=drone_confidence,
                probs=probs,
                timestamp=int(time.time() * 1000),
                window_sec=window_sec,
                metrics=metrics if is_drone else None,
            )
            await ws.send_text(msg.json())
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
