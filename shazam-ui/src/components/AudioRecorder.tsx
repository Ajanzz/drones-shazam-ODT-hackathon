// src/components/AudioRecorder.tsx
import React, { useRef, useState } from "react";

export const AudioRecorder: React.FC<{
  enabled: boolean;
  onChunk: (blob: Blob) => void;
}> = ({ enabled, onChunk }) => {
  const [recording, setRecording] = useState(false);
  const recRef = useRef<MediaRecorder | null>(null);

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    rec.ondataavailable = (e) => e.data?.size && onChunk(e.data);
    rec.start(300); // emit ~300ms chunks
    recRef.current = rec;
    setRecording(true);
  }

  function stop() {
    recRef.current?.stop();
    recRef.current = null;
    setRecording(false);
  }

  return (
    <div>
      <button onClick={recording ? stop : start} disabled={!enabled}>
        {recording ? "Stop" : "Start"}
      </button>
      {!enabled && <span> Waiting for WS…</span>}
    </div>
  );
};
