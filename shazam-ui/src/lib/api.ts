import type { InferenceEvent } from "../types";


export class ApiClient {
base: string;
ws: WebSocket | null = null;


constructor(base: string) {
this.base = base.replace(/\/$/, "");
}


connectWS(
onMessage: (msg: InferenceEvent) => void,
onOpen?: () => void,
onClose?: (ev: CloseEvent) => void
) {
const url = this.base.replace(/^http/, "ws") + "/ws/audio";
const ws = new WebSocket(url);
this.ws = ws;


ws.onopen = () => onOpen?.();
ws.onclose = (e) => onClose?.(e);
ws.onerror = (e) => console.error("WS error", e);
ws.onmessage = (ev) => {
try {
const msg: InferenceEvent = JSON.parse(ev.data);
onMessage(msg);
} catch (e) {
console.warn("Non-JSON WS message", ev.data);
}
};


return () => {
try {
ws.close();
} catch {}
};
}


sendAudioChunk(blob: Blob) {
if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
blob.arrayBuffer().then((ab) => this.ws!.send(ab));
}


async predictFile(file: File): Promise<InferenceEvent> {
const form = new FormData();
form.append("file", file);
const res = await fetch(`${this.base}/predict`, { method: "POST", body: form });
if (!res.ok) throw new Error(await res.text());
return (await res.json()) as InferenceEvent;
}
}