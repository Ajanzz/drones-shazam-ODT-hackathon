// types.ts (or wherever InferenceEvent is defined)

export interface DroneMetrics {
  speed: number | null;
  distance: number | null;
  direction: string | null;
  flight_state: string;
  has_payload: boolean;
  payload_confidence: number;
}

export interface InferenceEvent {
  label: string;
  score: number;
  probs: Record<string, number>;
  timestamp: number;
  window_sec?: number | null;
  metrics?: DroneMetrics | null;

  // NEW: drone type classifier output
  drone_type?: string | null;
  drone_type_score?: number | null;
  drone_type_probs?: Record<string, number> | null;
}
