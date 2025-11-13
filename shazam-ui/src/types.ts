export type DroneMetrics = {
  speed: number | null; // m/s
  distance: number | null; // meters
  direction: string | null; // "approaching", "receding", "stationary", "unknown"
  flight_state: string; // "hovering", "flying", "transitioning", "unknown"
  has_payload: boolean;
  payload_confidence: number;
};

export type InferenceEvent = {
  label: string; // e.g., "drone-dji-phantom" or "non-drone"
  score: number; // top label confidence 0..1
  probs: Record<string, number>; // full distribution for bars
  timestamp: number; // ms since epoch
  window_sec?: number; // analysis window length
  metrics?: DroneMetrics | null; // Drone characteristics (only present if drone detected)
};