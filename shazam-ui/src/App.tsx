import React, { useEffect, useMemo, useState } from "react";
import { ApiClient } from "./lib/api";
// import { AudioRecorder } from "./components/AudioRecorder";
import { ProbBar } from "./components/ProbBar";
import type { InferenceEvent } from "./types";
import DroneRadar from "./components/DroneRadar";
import type { TrackedDrone } from "./components/DroneRadar";


const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export default function App() {
  const api = useMemo(() => new ApiClient(BACKEND_URL), []);
  const [wsConnected, setWsConnected] = useState(false);
  const [current, setCurrent] = useState<InferenceEvent | null>(null);
  const [history, setHistory] = useState<InferenceEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // auto-connect WS on load
  useEffect(() => {
    const unsub = api.connectWS(
      (msg) => {
        setCurrent(msg);
        setHistory((h) => [msg, ...h].slice(0, 50));
      },
      () => setWsConnected(true),
      () => setWsConnected(false)
    );
    return () => unsub();
  }, [api]);

  async function onUpload(file: File) {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictFile(file);
      setCurrent(res);
      setHistory((h) => [res, ...h].slice(0, 50));
    } catch (e: any) {
      setError(e?.message ?? "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  // ---- derive tracked drones for radar from history ----
  const dronesForRadar: TrackedDrone[] = useMemo(() => {
    // take recent events that are drones with metrics
    const droneEvents = history.filter(
      (e) => e.label === "drone" && e.metrics
    );

    const maxDrones = 12;
    const limited = droneEvents.slice(0, maxDrones);

    return limited.map((e, idx) => {
      const m = e.metrics!;
      const distance =
        typeof m.distance === "number" && !isNaN(m.distance)
          ? m.distance
          : 50;

      // For now we don't have real bearing from backend, so we spread them around the circle.
      // Later you can add a bearing_deg field to metrics and plug it in here.
      const bearingDeg = (idx * 360) / Math.max(maxDrones, 1);

      // Normalize direction string
      const rawDir = (m.direction || "unknown").toLowerCase();
      let direction: "approaching" | "receding" | "stationary" | "unknown";
      if (rawDir.includes("approach")) direction = "approaching";
      else if (rawDir.includes("reced")) direction = "receding";
      else if (rawDir.includes("station")) direction = "stationary";
      else direction = "unknown";

      return {
        id: String(e.timestamp),
        distance,
        bearingDeg,
        direction,
        hasPayload: m.has_payload,
        payloadConfidence: m.payload_confidence,
      };
    });
  }, [history]);

  return (
    <div className="app-shell page">
      <header className="header">
        <div className="brand">
          <div className="logo">A4K</div>
          <div>
            <h1 className="title">Shazam for Drones</h1>
            <div className="subtitle">Lightweight on-device audio recognition</div>
          </div>
        </div>

        <div className={"pill " + (wsConnected ? "ok" : "bad")}>
          WS: {wsConnected ? "connected" : "disconnected"}
        </div>
      </header>

      <section className="grid">
        <div className="card">
          <h2 className="section-title">
            <span className="icon">🎙️</span> Live microphone
          </h2>
          <p className="muted">
            Streams audio chunks to <code>{BACKEND_URL}</code> via WebSocket
            (<code>/ws/audio</code>). Start once the backend is running.
          </p>
          {/* Uncomment when component gets added
          <AudioRecorder
            enabled={wsConnected}
            onChunk={(blob) => api.sendAudioChunk(blob)}
          /> */}
        </div>

        <div className="card">
          <h2 className="section-title">
            <span className="icon">📁</span> Upload a clip
          </h2>
          <p className="muted">Sends a file to POST /predict for one-shot classification.</p>

          <div className="file-drop">
            <input
              className="file-input"
              type="file"
              accept="audio/*"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
              }}
            />
            <div className="hint">Drop or choose an audio file</div>
          </div>

          {loading && <p className="muted">Analyzing…</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      <section className="card">
        <h2 className="section-title">
          <span className="icon">🔮</span> Current prediction
        </h2>
        {!current ? (
          <p className="muted">No inference yet.</p>
        ) : (
          <div>
            <div className="current-row">
              <div className="big-label">{current.label}</div>
              <div className="big-score">{(current.score * 100).toFixed(1)}%</div>
            </div>
            <div className="prob-grid">
              {Object.entries(current.probs).map(([k, v]) => (
                <ProbBar key={k} label={k} value={v} />
              ))}
            </div>
            <p className="muted small">
              ts: {new Date(current.timestamp).toLocaleTimeString()} • dur:{" "}
              {(current.window_sec ?? 0).toFixed(2)}s
            </p>
          </div>
        )}
      </section>

      {current?.metrics && (
        <section className="card">
          <h2 className="section-title">
            <span className="icon">🚁</span> Drone Characteristics
          </h2>
          <div className="metrics-grid">
            <div className="metric-item">
              <div className="metric-label">Speed</div>
              <div className="metric-value">
                {current.metrics.speed !== null && current.metrics.speed !== undefined
                  ? `${current.metrics.speed.toFixed(1)} m/s (${(
                      current.metrics.speed * 3.6
                    ).toFixed(1)} km/h)`
                  : "Calculating..."}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Distance</div>
              <div className="metric-value">
                {current.metrics.distance !== null && current.metrics.distance !== undefined
                  ? `${current.metrics.distance.toFixed(1)} m`
                  : "Calculating..."}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Direction</div>
              <div className="metric-value">
                {current.metrics.direction ? (
                  <span
                    className={`direction-badge direction-${current.metrics.direction}`}
                  >
                    {current.metrics.direction}
                  </span>
                ) : (
                  "Unknown"
                )}
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Flight State</div>
              <div className="metric-value">
                <span
                  className={`state-badge state-${current.metrics.flight_state}`}
                >
                  {current.metrics.flight_state}
                </span>
              </div>
            </div>
            <div className="metric-item">
              <div className="metric-label">Payload</div>
              <div className="metric-value">
                {current.metrics.has_payload ? (
                  <span className="payload-badge payload-yes">
                    Yes ({(current.metrics.payload_confidence * 100).toFixed(0)}%)
                  </span>
                ) : (
                  <span className="payload-badge payload-no">
                    No ({(current.metrics.payload_confidence * 100).toFixed(0)}%)
                  </span>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* New radar map section */}
      <section className="card">
        <h2 className="section-title">
          <span className="icon">🗺️</span> Drone Radar Map
        </h2>
        {dronesForRadar.length === 0 ? (
          <p className="muted">No drone tracks yet.</p>
        ) : (
          <DroneRadar drones={dronesForRadar} />
        )}
      </section>

      <section className="card">
        <h2 className="section-title">
          <span className="icon">🧠</span> Recent events
        </h2>
        {history.length === 0 ? (
          <p className="muted">Nothing yet.</p>
        ) : (
          <ul className="history">
            {history.map((e, i) => (
              <li key={i} className="history-item">
                <span className="tag">{e.label}</span>
                <span className="score">{(e.score * 100).toFixed(1)}%</span>
                <span className="muted small">
                  {new Date(e.timestamp).toLocaleTimeString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
