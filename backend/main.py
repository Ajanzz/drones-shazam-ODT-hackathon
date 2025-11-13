# backend/main.py

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from itertools import cycle
from typing import Optional, Dict

import uvicorn
import time
import io

import numpy as np
import onnxruntime as ort
import librosa
import av  # ffmpeg-based decoder

from audio_analyzer import DroneAudioAnalyzer

# ============================================================
# Config: toggle dummy mode for frontend testing
# ============================================================

USE_DUMMY_DATA = False  # set True to test UI without model/analyzer

# ============================================================
# Data models sent back to the frontend
# ============================================================

class DroneMetrics(BaseModel):
    """Drone characteristics extracted from audio."""
    speed: Optional[float] = None          # m/s
    distance: Optional[float] = None       # meters
    direction: Optional[str] = None        # "approaching", "receding", "stationary", "unknown"
    flight_state: str = "unknown"          # "hovering", "flying", "transitioning", "unknown"
    has_payload: bool = False
    payload_confidence: float = 0.0


class InferenceEvent(BaseModel):
    label: str                              # "drone" or "non-drone"
    score: float                            # confidence for label
    probs: Dict[str, float]                 # per-class probabilities
    timestamp: int                          # ms since epoch
    window_sec: float | None = None         # analysis window length
    metrics: Optional[DroneMetrics] = None  # optional drone metrics


# ============================================================
# Dummy data helpers (for UI testing without model)
# ============================================================

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


# ============================================================
# FastAPI app + CORS
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ONNX model setup (from master branch)
# ============================================================

ONNX_MODEL_PATH = "model_detector.onnx"
CLASS_NAMES = ["non-drone", "drone"]

session: Optional[ort.InferenceSession] = None
INPUT_NAME: Optional[str] = None
OUTPUT_NAME: Optional[str] = None

if not USE_DUMMY_DATA:
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],  # or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    INPUT_NAME = input_meta.name
    OUTPUT_NAME = output_meta.name

    print("Loaded ONNX model:")
    print("  Input :", INPUT_NAME, input_meta.shape, input_meta.type)
    print("  Output:", OUTPUT_NAME, output_meta.shape, output_meta.type)


# ============================================================
# Audio / feature helpers (must match ONNX training)
# ============================================================

TARGET_SR = 16000      # must match training
N_MELS = 64            # model expects ['batch', 1, 64, 'time']
MAX_SECONDS = 3        # window length used for training


def decode_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decode any audio bytes (wav / webm / opus / m4a / mp3 / etc.)
    into a mono float32 waveform at TARGET_SR.
    """
    container = av.open(io.BytesIO(raw_bytes))
    stream = container.streams.audio[0]

    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        # if multi-channel, average to mono
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        frames.append(arr)

    if not frames:
        raise ValueError("No audio frames decoded")

    audio = np.concatenate(frames).astype(np.float32)
    orig_sr = stream.rate

    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

    # trim / pad to fixed window
    max_samples = TARGET_SR * MAX_SECONDS
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")

    return audio


def preprocess_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Convert raw audio bytes -> log-mel tensor of shape
    (1, 1, 64, time) matching ONNX input.
    """
    y = decode_audio_bytes(raw_bytes)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512,
    )
    logmel = librosa.power_to_db(mel).astype(np.float32)  # (64, time)

    # (batch, channels, 64, time)
    x = logmel[np.newaxis, np.newaxis, :, :]
    return x


def run_model(input_tensor: np.ndarray) -> Dict[str, float]:
    """
    Run ONNX model and return probs per class.
    """
    if session is None or INPUT_NAME is None or OUTPUT_NAME is None:
        raise RuntimeError("ONNX session is not initialized (check USE_DUMMY_DATA and model path).")

    input_tensor = input_tensor.astype(np.float32)

    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})
    out = outputs[0]  # shape: (batch, num_classes)
    out = np.squeeze(out)  # (num_classes,)

    # if model already has softmax, skip this and just normalize if needed
    exp = np.exp(out - np.max(out))
    probs = exp / np.sum(exp)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ============================================================
# Analyzer setup (dashboard metrics)
# ============================================================

analyzer: Optional[DroneAudioAnalyzer] = None
if not USE_DUMMY_DATA:
    analyzer = DroneAudioAnalyzer()


# ============================================================
# HTTP: one-shot /predict
# Uses: ONNX for classification + analyzer for metrics
# ============================================================

@app.post("/predict", response_model=InferenceEvent)
async def predict(file: UploadFile = File(...)):
    start = time.time()
    raw_bytes = await file.read()
    print(f"[PREDICT] file={file.filename}, bytes={len(raw_bytes)}")

    # Dummy mode: just exercise UI
    if USE_DUMMY_DATA:
        event = build_dummy_event(window=None)
        print(f"[PREDICT] dummy event -> {event}")
        return event

    assert analyzer is not None, "Analyzer not initialized"

    try:
        # --- 1) Run ONNX classifier (drone vs non-drone) ---
        x = preprocess_audio_bytes(raw_bytes)
        print(f"[PREDICT] preprocess ok in {time.time() - start:.3f}s, shape={x.shape}")

        probs = run_model(x)
        print(f"[PREDICT] inference ok in {time.time() - start:.3f}s, probs={probs}")
    except Exception as e:
        import traceback
        print("[PREDICT] ERROR during ONNX processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Derive label + score from ONNX probabilities
    label = max(probs, key=probs.get)
    score = probs[label]
    is_drone = label == "drone" and score >= 0.5

    # --- 2) Compute metrics from analyzer (speed, distance, etc.) ---
    metrics_obj: Optional[DroneMetrics] = None
    try:
        file_format = file.filename.split(".")[-1] if file.filename else "wav"
        metrics_dict = analyzer.analyze_audio(raw_bytes, format=file_format)

        # Optional: refine with analyzer features / ambient filter
        try:
            audio_io = io.BytesIO(raw_bytes)
            audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
            features = analyzer._extract_features(audio_array, sr)
            # you could combine this with ONNX output if you want:
            # is_drone = is_drone and analyzer.filter_ambient_noise(features)
        except Exception:
            features = {}

        if is_drone:
            if features:
                has_payload, payload_conf = analyzer._detect_payload(features)
            else:
                has_payload, payload_conf = False, 0.0

            metrics_obj = DroneMetrics(
                speed=metrics_dict.get("speed"),
                distance=metrics_dict.get("distance"),
                direction=metrics_dict.get("direction"),
                flight_state=metrics_dict.get("flight_state", "unknown"),
                has_payload=has_payload,
                payload_confidence=payload_conf,
            )
    except Exception as e:
        # Metrics are optional; log and continue if they fail
        import traceback
        print("[PREDICT] WARNING: analyzer metrics failed:")
        traceback.print_exc()
        metrics_obj = None

    event = InferenceEvent(
        label=label,
        score=score,
        probs=probs,
        timestamp=int(time.time() * 1000),
        window_sec=None,
        metrics=metrics_obj if is_drone else None,
    )

    print(f"[PREDICT] total {time.time() - start:.3f}s")
    return event


# ============================================================
# WS: streaming /ws/audio
# Currently uses dummy or heuristic analyzer;
# ONNX streaming can be plugged in later.
# ============================================================

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    window_sec = 0.3  # ~300ms chunks

    # Dummy mode: just ignore chunk contents and emit rotating dummy events
    if USE_DUMMY_DATA:
        print("[WS] client connected (dummy mode)")
        try:
            while True:
                await ws.receive_bytes()  # keep connection alive
                msg = build_dummy_event(window=window_sec)
                await ws.send_text(msg.json())
        except WebSocketDisconnect:
            print("[WS] client disconnected (dummy mode)")
            return

    # Real streaming mode with analyzer
    print("[WS] client connected")
    stream_analyzer = DroneAudioAnalyzer()

    try:
        while True:
            chunk: bytes = await ws.receive_bytes()
            print(f"[WS] got chunk of {len(chunk)} bytes")

            # --- 1) Analyze audio chunk for metrics / confidence ---
            metrics_dict = stream_analyzer.analyze_audio(chunk, format="webm")

            # --- 2) Extract features and filter ambient noise ---
            try:
                audio_io = io.BytesIO(chunk)
                audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
                features = stream_analyzer._extract_features(audio_array, sr)
                is_drone = stream_analyzer.filter_ambient_noise(features)
            except Exception:
                is_drone = True
                features = {}

            # --- 3) Heuristic "drone" confidence for streaming ---
            drone_confidence = metrics_dict.get("confidence", 0.5) if is_drone else 0.1
            probs = {
                "drone": float(drone_confidence),
                "non-drone": float(1.0 - drone_confidence),
            }

            # --- 4) Build metrics object ---
            if features:
                has_payload, payload_conf = stream_analyzer._detect_payload(features)
            else:
                has_payload, payload_conf = False, 0.0

            metrics_obj = DroneMetrics(
                speed=metrics_dict.get("speed"),
                distance=metrics_dict.get("distance"),
                direction=metrics_dict.get("direction"),
                flight_state=metrics_dict.get("flight_state", "unknown"),
                has_payload=has_payload,
                payload_confidence=payload_conf,
            )

            # TODO: If you want, you can also run the ONNX model here by
            # buffering enough audio and calling preprocess_audio_bytes + run_model.
            # Right now, websocket streaming uses the heuristic analyzer only.

            msg = InferenceEvent(
                label="drone" if drone_confidence > 0.5 else "non-drone",
                score=float(drone_confidence),
                probs=probs,
                timestamp=int(time.time() * 1000),
                window_sec=window_sec,
                metrics=metrics_obj if is_drone else None,
            )

            await ws.send_text(msg.json())

    except WebSocketDisconnect:
        print("[WS] client disconnected")


# ============================================================
# Entrypoint (optional, if you want to run via `python main.py`)
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
