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
    speed: Optional[float] = None          # m/s
    distance: Optional[float] = None       # meters
    direction: Optional[str] = None        # "approaching", "receding", "stationary", "unknown"
    flight_state: str = "unknown"          # "hovering", "flying", "transitioning", "unknown"
    has_payload: bool = False
    payload_confidence: float = 0.0


class InferenceEvent(BaseModel):
    # primary detector (drone vs non-drone)
    label: str                              # "drone" or "non-drone"
    score: float                            # confidence for label
    probs: Dict[str, float]                 # per-class probabilities (detector)
    timestamp: int                          # ms since epoch
    window_sec: float | None = None         # analysis window length
    metrics: Optional[DroneMetrics] = None  # optional drone metrics

    # secondary classifier (drone type A–J), only populated when label == "drone"
    drone_type: Optional[str] = None
    drone_type_score: Optional[float] = None
    drone_type_probs: Optional[Dict[str, float]] = None


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
        drone_type="drone_A",
        drone_type_score=0.9,
        drone_type_probs={f"drone_{c}": (0.9 if c == "A" else 0.01) for c in "ABCDEFGHIJ"},
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
# ONNX model setup: detector + classifier
# ============================================================

DETECTOR_MODEL_PATH = "model_detector (1).onnx"
CLASSIFIER_MODEL_PATH = "drone_classifier_20s.onnx"

DETECTOR_CLASS_NAMES = ["non-drone", "drone"]
DRONE_CLASS_NAMES = [f"drone_{c}" for c in "ABCDEFGHIJ"]

detector_session: Optional[ort.InferenceSession] = None
classifier_session: Optional[ort.InferenceSession] = None

DETECTOR_INPUT_NAME: Optional[str] = None
DETECTOR_OUTPUT_NAME: Optional[str] = None

CLASSIFIER_INPUT_NAME: Optional[str] = None
CLASSIFIER_OUTPUT_NAME: Optional[str] = None

if not USE_DUMMY_DATA:
    # detector (drone vs non-drone)
    detector_session = ort.InferenceSession(
        DETECTOR_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    det_input_meta = detector_session.get_inputs()[0]
    det_output_meta = detector_session.get_outputs()[0]
    DETECTOR_INPUT_NAME = det_input_meta.name
    DETECTOR_OUTPUT_NAME = det_output_meta.name

    print("Loaded detector ONNX model:")
    print("  Input :", DETECTOR_INPUT_NAME, det_input_meta.shape, det_input_meta.type)
    print("  Output:", DETECTOR_OUTPUT_NAME, det_output_meta.shape, det_output_meta.type)

    # classifier (drone type A–J)
    classifier_session = ort.InferenceSession(
        CLASSIFIER_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    cls_input_meta = classifier_session.get_inputs()[0]
    cls_output_meta = classifier_session.get_outputs()[0]
    CLASSIFIER_INPUT_NAME = cls_input_meta.name
    CLASSIFIER_OUTPUT_NAME = cls_output_meta.name

    print("Loaded classifier ONNX model:")
    print("  Input :", CLASSIFIER_INPUT_NAME, cls_input_meta.shape, cls_input_meta.type)
    print("  Output:", CLASSIFIER_OUTPUT_NAME, cls_output_meta.shape, cls_output_meta.type)


# ============================================================
# Audio / feature helpers (must match ONNX training)
# ============================================================

TARGET_SR = 16000      # must match training
N_MELS = 64            # model expects ['batch', 1, 64, 'time']
MAX_SECONDS = 3        # window length used for training


def decode_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode audio bytes exactly like in training (librosa.load)."""
    audio_io = io.BytesIO(raw_bytes)
    y, sr = librosa.load(audio_io, sr=TARGET_SR, mono=True)

    max_samples = TARGET_SR * MAX_SECONDS
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant")

    return y


def preprocess_audio_bytes(raw_bytes: bytes) -> np.ndarray:
    """Convert raw audio bytes -> log-mel tensor (1,1,64,T), same as training."""
    y = decode_audio_bytes(raw_bytes)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512,
    )
    logmel = librosa.power_to_db(mel).astype(np.float32)  # (64, time)

    x = logmel[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, T)
    return x


def _softmax_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.squeeze(logits).astype(np.float32)
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def run_detector(input_tensor: np.ndarray) -> Dict[str, float]:
    if detector_session is None or DETECTOR_INPUT_NAME is None or DETECTOR_OUTPUT_NAME is None:
        raise RuntimeError("Detector ONNX session is not initialized.")

    outputs = detector_session.run([DETECTOR_OUTPUT_NAME], {DETECTOR_INPUT_NAME: input_tensor})
    probs = _softmax_to_probs(outputs[0])
    return {DETECTOR_CLASS_NAMES[i]: float(probs[i]) for i in range(len(DETECTOR_CLASS_NAMES))}


def run_classifier(input_tensor: np.ndarray) -> Dict[str, float]:
    if classifier_session is None or CLASSIFIER_INPUT_NAME is None or CLASSIFIER_OUTPUT_NAME is None:
        raise RuntimeError("Classifier ONNX session is not initialized.")

    outputs = classifier_session.run([CLASSIFIER_OUTPUT_NAME], {CLASSIFIER_INPUT_NAME: input_tensor})
    probs = _softmax_to_probs(outputs[0])
    return {DRONE_CLASS_NAMES[i]: float(probs[i]) for i in range(len(DRONE_CLASS_NAMES))}


# ============================================================
# Analyzer setup (dashboard metrics)
# ============================================================

analyzer: Optional[DroneAudioAnalyzer] = None
if not USE_DUMMY_DATA:
    analyzer = DroneAudioAnalyzer()


# ============================================================
# HTTP: one-shot /predict
# Uses: detector ONNX + (optional) classifier + analyzer
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
        # --- 1) Preprocess audio ---
        x = preprocess_audio_bytes(raw_bytes)
        print(f"[PREDICT] preprocess ok in {time.time() - start:.3f}s, shape={x.shape}")

        # --- 2) Run detector (drone vs non-drone) ---
        det_probs = run_detector(x)
        print(f"[PREDICT] detector ok in {time.time() - start:.3f}s, probs={det_probs}")
    except Exception as e:
        import traceback
        print("[PREDICT] ERROR during detector ONNX processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Derive label + score from detector probabilities
    label = max(det_probs, key=det_probs.get)
    score = det_probs[label]
    is_drone = label == "drone" and score >= 0.5

    # --- 3) Optional: drone-type classifier (A–J) ---
    drone_type = None
    drone_type_score = None
    drone_type_probs: Optional[Dict[str, float]] = None

    if is_drone:
        try:
            cls_probs = run_classifier(x)
            drone_type_probs = cls_probs
            drone_type = max(cls_probs, key=cls_probs.get)
            drone_type_score = cls_probs[drone_type]
            print(f"[PREDICT] classifier ok in {time.time() - start:.3f}s, type={drone_type}, probs={cls_probs}")
        except Exception as e:
            import traceback
            print("[PREDICT] WARNING: classifier failed, continuing without drone type")
            traceback.print_exc()

    # --- 4) Compute metrics from analyzer (speed, distance, etc.) ---
    metrics_obj: Optional[DroneMetrics] = None
    try:
        file_format = file.filename.split(".")[-1] if file.filename else "wav"
        metrics_dict = analyzer.analyze_audio(raw_bytes, format=file_format)

        try:
            audio_io = io.BytesIO(raw_bytes)
            audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
            features = analyzer._extract_features(audio_array, sr)
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
        import traceback
        print("[PREDICT] WARNING: analyzer metrics failed:")
        traceback.print_exc()
        metrics_obj = None

    event = InferenceEvent(
        label=label,
        score=score,
        probs=det_probs,
        timestamp=int(time.time() * 1000),
        window_sec=None,
        metrics=metrics_obj if is_drone else None,
        drone_type=drone_type,
        drone_type_score=drone_type_score,
        drone_type_probs=drone_type_probs,
    )

    print(f"[PREDICT] total {time.time() - start:.3f}s")
    return event


# ============================================================
# WS: streaming /ws/audio
# (unchanged: still heuristic analyzer only)
# ============================================================

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    window_sec = 0.3  # ~300ms chunks

    if USE_DUMMY_DATA:
        print("[WS] client connected (dummy mode)")
        try:
            while True:
                await ws.receive_bytes()
                msg = build_dummy_event(window=window_sec)
                await ws.send_text(msg.json())
        except WebSocketDisconnect:
            print("[WS] client disconnected (dummy mode)")
            return

    print("[WS] client connected")
    stream_analyzer = DroneAudioAnalyzer()

    try:
        while True:
            chunk: bytes = await ws.receive_bytes()
            print(f"[WS] got chunk of {len(chunk)} bytes")

            metrics_dict = stream_analyzer.analyze_audio(chunk, format="webm")

            try:
                audio_io = io.BytesIO(chunk)
                audio_array, sr = librosa.load(audio_io, sr=44100, mono=True)
                features = stream_analyzer._extract_features(audio_array, sr)
                is_drone = stream_analyzer.filter_ambient_noise(features)
            except Exception:
                is_drone = True
                features = {}

            drone_confidence = metrics_dict.get("confidence", 0.5) if is_drone else 0.1
            probs = {
                "drone": float(drone_confidence),
                "non-drone": float(1.0 - drone_confidence),
            }

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

            msg = InferenceEvent(
                label="drone" if drone_confidence > 0.5 else "non-drone",
                score=float(drone_confidence),
                probs=probs,
                timestamp=int(time.time() * 1000),
                window_sec=window_sec,
                metrics=metrics_obj if is_drone else None,
                drone_type=None,
                drone_type_score=None,
                drone_type_probs=None,
            )

            await ws.send_text(msg.json())

    except WebSocketDisconnect:
        print("[WS] client disconnected")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
