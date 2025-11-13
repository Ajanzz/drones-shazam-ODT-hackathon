
# backend/main.py

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import io

import numpy as np
import onnxruntime as ort
import librosa
import av  # ffmpeg-based decoder

# ------------ response model ------------

class InferenceEvent(BaseModel):
    label: str
    score: float
    probs: dict[str, float]
    timestamp: int
    window_sec: float | None = None


app = FastAPI()

# ------------ CORS for Vite -------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ ONNX model load -----------

ONNX_MODEL_PATH = "model_detector.onnx"

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


CLASS_NAMES = ["non-drone", "drone"]

# ------------ audio / feature helpers ----

TARGET_SR = 16000
N_MELS = 64          # must match ONNX: ['batch', 1, 64, 'time']
MAX_SECONDS = 3      # length of window trained on (change if needed)


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


def run_model(input_tensor: np.ndarray) -> dict[str, float]:
    """
    Run ONNX model and return probs per class.
    """
    input_tensor = input_tensor.astype(np.float32)

    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})
    out = outputs[0]  # shape: (batch, num_classes)
    out = np.squeeze(out)  # (num_classes,)

    # if model already has softmax, skip this
    exp = np.exp(out - np.max(out))
    probs = exp / np.sum(exp)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def probs_to_event(probs: dict[str, float], window_sec: float | None) -> InferenceEvent:
    label = max(probs, key=probs.get)
    score = probs[label]
    return InferenceEvent(
        label=label,
        score=score,
        probs=probs,
        timestamp=int(time.time() * 1000),
        window_sec=window_sec,
    )

# ------------ HTTP: one-shot /predict ----

@app.post("/predict", response_model=InferenceEvent)
async def predict(file: UploadFile = File(...)):
    start = time.time()
    raw_bytes = await file.read()
    print(f"[PREDICT] file={file.filename}, bytes={len(raw_bytes)}")

    try:
        x = preprocess_audio_bytes(raw_bytes)
        print(f"[PREDICT] preprocess ok in {time.time() - start:.3f}s, shape={x.shape}")

        probs = run_model(x)
        print(f"[PREDICT] inference ok in {time.time() - start:.3f}s, probs={probs}")
    except Exception as e:
        import traceback
        print("[PREDICT] ERROR during processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    event = probs_to_event(probs, window_sec=None)
    print(f"[PREDICT] total {time.time() - start:.3f}s")
    return event

# ------------ WS: streaming /ws/audio ----

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    print("[WS] client connected")
    window_sec = 0.3

    try:
        while True:
            # just receive the audio chunk so the socket stays alive
            chunk: bytes = await ws.receive_bytes()
            print(f"[WS] got chunk of {len(chunk)} bytes")

            # TODO: later, plug in real model here
            # For now, send a neutral / dummy prediction so UI works
            msg = probs_to_event(
                {"drone": 0.5, "non-drone": 0.5},
                window_sec=window_sec,
            )

            await ws.send_text(msg.json())

    except WebSocketDisconnect:
        print("[WS] client disconnected")