import React, { useEffect, useMemo, useState } from "react";
import { ApiClient } from "./lib/api";
// import { AudioRecorder } from "./components/AudioRecorder"; 
import { ProbBar } from "./components/ProbBar";
import type { InferenceEvent } from "./types";

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

  return (
    <div className="page">
      <header className="header">
        <h1>Shazam for Drones</h1>
        <div className={"pill " + (wsConnected ? "ok" : "bad")}>
          WS: {wsConnected ? "connected" : "disconnected"}
        </div>
      </header>

      <section className="grid">
        <div className="card">
          <h2>Live microphone</h2>
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
          <h2>Upload a clip</h2>
          <p className="muted">Sends a file to POST /predict for one-shot classification.</p>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
            }}
          />
          {loading && <p>Analyzing…</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      <section className="card">
        <h2>Current prediction</h2>
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

      <section className="card">
        <h2>Recent events</h2>
        {history.length === 0 ? (
          <p className="muted">Nothing yet.</p>
        ) : (
          <ul className="history">
            {history.map((e, i) => (
              <li key={i}>
                <span className="tag">{e.label}</span>
                <span>{(e.score * 100).toFixed(1)}%</span>
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
