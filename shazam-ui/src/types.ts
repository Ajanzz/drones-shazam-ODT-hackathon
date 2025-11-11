export type InferenceEvent = {
label: string; // e.g., "drone-dji-phantom" or "non-drone"
score: number; // top label confidence 0..1
probs: Record<string, number>; // full distribution for bars
timestamp: number; // ms since epoch
window_sec?: number; // analysis window length
};