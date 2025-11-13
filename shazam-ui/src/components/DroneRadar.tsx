import React from "react";

export type Direction = "approaching" | "receding" | "stationary" | "unknown";

export interface TrackedDrone {
  id: string;
  distance: number;         // meters
  bearingDeg: number;       // 0° = up, 90° = right
  direction: Direction;
  hasPayload: boolean;
  payloadConfidence: number;
}

interface DroneRadarProps {
  drones: TrackedDrone[];
  /**
   * Optional max cap for the outer ring, in meters.
   * If not provided, we autoscale based on data.
   */
  maxRadiusMeters?: number;
}

const DroneRadar: React.FC<DroneRadarProps> = ({ drones, maxRadiusMeters }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  // ---------- adaptive distance scale ----------
  const distances = drones
    .map((d) => d.distance)
    .filter((d) => Number.isFinite(d) && d >= 0);

  const MIN_SCALE = 5;
  const rawDynamicMax =
    distances.length > 0 ? Math.max(...distances) * 1.1 : MIN_SCALE;

  let scaleMaxMeters: number;
  if (maxRadiusMeters !== undefined) {
    scaleMaxMeters = Math.max(
      Math.min(rawDynamicMax, maxRadiusMeters),
      MIN_SCALE
    );
  } else {
    scaleMaxMeters = Math.max(rawDynamicMax, MIN_SCALE);
  }

  const normalizeDistance = (d: number) => {
    if (!isFinite(d) || d <= 0) return 0.05;
    const clamped = Math.min(d, scaleMaxMeters);
    return clamped / scaleMaxMeters;
  };

  const directionColor = (direction: Direction) => {
    switch (direction) {
      case "approaching":
        return "#f87171"; // red
      case "receding":
        return "#60a5fa"; // blue
      case "stationary":
        return "#fde047"; // yellow
      default:
        return "#a855f7"; // purple = unknown
    }
  };

  const opacityFromDistance = (d: number) => {
    const norm = 1 - Math.min(d / scaleMaxMeters, 1);
    return 0.3 + norm * 0.7; // 0.3–1.0
  };

  // ---------- core radar renderer (no randomness) ----------
  const renderRadar = (size: number) => {
    const center = size / 2;
    const radius = size * 0.45;

    return (
      <svg width={size} height={size}>
        {/* Background */}
        <defs>
          <radialGradient id="radarBg" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#0f172a" />
            <stop offset="100%" stopColor="#020617" />
          </radialGradient>
        </defs>

        <rect width={size} height={size} fill="url(#radarBg)" rx={16} />

        {/* Rings */}
        {[0.25, 0.5, 0.75, 1].map((r, idx) => (
          <circle
            key={idx}
            cx={center}
            cy={center}
            r={radius * r}
            fill="none"
            stroke="#1f2937"
            strokeDasharray="4 4"
          />
        ))}

        {/* Crosshair */}
        <line
          x1={center}
          y1={center - radius}
          x2={center}
          y2={center + radius}
          stroke="#1f2937"
          strokeWidth={1}
        />
        <line
          x1={center - radius}
          y1={center}
          x2={center + radius}
          y2={center}
          stroke="#1f2937"
          strokeWidth={1}
        />

        {/* YOU */}
        <circle cx={center} cy={center} r={6} fill="#22c55e" />
        <text
          x={center}
          y={center - 12}
          textAnchor="middle"
          fontSize="10"
          fill="#e5e7eb"
        >
          YOU
        </text>

        {/* Drones */}
        {drones.map((drone) => {
          const norm = normalizeDistance(drone.distance);
          const r = norm * radius;

          // Use bearingDeg -> deterministic angle
          // 0° = up; SVG 0° is to the right, so rotate by -90°
          const bearing = Number.isFinite(drone.bearingDeg)
            ? drone.bearingDeg
            : 0; // fallback if missing
          const angleRad = ((bearing - 90) * Math.PI) / 180;
          const x = center + r * Math.cos(angleRad);
          const y = center + r * Math.sin(angleRad);

          const color = directionColor(drone.direction);
          const opacity = opacityFromDistance(drone.distance);
          const hasPayloadIcon =
            drone.hasPayload || drone.payloadConfidence > 0.7;

          if (hasPayloadIcon) {
            // Payload drone: colored glow + bomb, NO inner dot
            return (
              <g key={drone.id}>
                <circle
                  cx={x}
                  cy={y}
                  r={14}
                  fill={color}
                  fillOpacity={opacity}
                />
                <text
                  x={x}
                  y={y + 4}
                  fontSize="20"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  💣
                </text>
                <text
                  x={x}
                  y={y + 24}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#e5e7eb"
                >
                  {Math.round(drone.distance)}m
                </text>
              </g>
            );
          }

          // Normal drone: colored circle + dot
          return (
            <g key={drone.id}>
              <circle
                cx={x}
                cy={y}
                r={10}
                fill={color}
                fillOpacity={opacity}
                stroke="#f9fafb"
                strokeWidth={1}
              />
              <circle cx={x} cy={y} r={4} fill="#0b1120" />
              <text
                x={x}
                y={y + 18}
                textAnchor="middle"
                fontSize="9"
                fill="#e5e7eb"
              >
                {Math.round(drone.distance)}m
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // ---------- fullscreen overlay with smooth zoom ----------
  const fullscreenOverlay = isExpanded && (
    <div
      onClick={() => setIsExpanded(false)}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(15,23,42,0.55)",
        backdropFilter: "blur(18px)",
        WebkitBackdropFilter: "blur(18px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 999,
        overflow: "hidden",
        animation: "radarFadeIn 250ms ease-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          position: "relative",
          borderRadius: 24,
          background:
            "radial-gradient(circle at top, #0f172a 0%, #020617 65%)",
          padding: 24,
          boxShadow: "0 24px 80px rgba(0,0,0,0.9)",
          border: "1px solid rgba(56,189,248,0.6)",
          transformOrigin: "center",
          animation: "radarZoomIn 350ms cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      >
        {/* Close button */}
        <button
          onClick={() => setIsExpanded(false)}
          style={{
            position: "absolute",
            top: 18,
            right: 22,
            background: "rgba(15,23,42,0.8)",
            borderRadius: "999px",
            border: "1px solid rgba(148,163,184,0.7)",
            width: 32,
            height: 32,
            fontSize: 18,
            color: "#e5e7eb",
            cursor: "pointer",
          }}
        >
          ✕
        </button>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
          }}
        >
          {renderRadar(480)}
          <div
            style={{
              fontSize: 11,
              color: "#9ca3af",
              marginTop: 4,
              textAlign: "center",
            }}
          >
            Click outside or ✕ to close
          </div>
        </div>
      </div>

      {/* keyframes injected once while overlay is shown */}
      <style>
        {`
          @keyframes radarZoomIn {
            0% {
              transform: scale(0.25) translateY(40px);
              opacity: 0;
            }
            40% {
              transform: scale(0.55) translateY(8px);
              opacity: 0.6;
            }
            100% {
              transform: scale(1) translateY(0);
              opacity: 1;
            }
          }

          @keyframes radarFadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
          }
        `}
      </style>
    </div>
  );

  // ---------- legend panel with fixed spacing ----------
  const legend = (
    <div
      style={{
        width: 320,
        padding: 20,
        borderRadius: 20,
        background: "linear-gradient(145deg, #0f172a 0%, #0b1221 95%)",
        border: "1px solid rgba(56,189,248,0.35)",
        boxShadow: "0 8px 25px rgba(0,0,0,0.4)",
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 4,
        }}
      >
        <span
          style={{
            fontSize: 16,
            fontWeight: 600,
            color: "#e5e7eb",
          }}
        >
          Legend
        </span>

        <span
          style={{
            fontSize: 12,
            padding: "4px 10px",
            borderRadius: 999,
            color: "#93c5fd",
            background: "rgba(56,189,248,0.15)",
            border: "1px solid rgba(56,189,248,0.35)",
          }}
        >
          0–{Math.round(scaleMaxMeters)} m
        </span>
      </div>

      {/* Items */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Approaching */}
        <LegendItem
          color="#f87171"
          title="Approaching drone"
          description="Red circle (no payload)"
        />
        {/* Receding */}
        <LegendItem
          color="#60a5fa"
          title="Receding drone"
          description="Blue circle (no payload)"
        />
        {/* Stationary */}
        <LegendItem
          color="#fde047"
          title="Stationary drone"
          description="Yellow circle (no payload)"
        />
        {/* Payload */}
        <LegendBomb />
      </div>
    </div>
  );

  return (
    <>
      {fullscreenOverlay}

      <div
        style={{
          display: "flex",
          flexDirection: "row",
          flexWrap: "wrap",
          alignItems: "flex-start",
          justifyContent: "center",
          gap: 32,
          width: "100%",
        }}
      >
        {/* Radar (click to expand) */}
        <div
          style={{
            flexShrink: 0,
            cursor: "zoom-in",
            transition: "transform 180ms ease-out, box-shadow 180ms ease-out",
          }}
          onClick={() => setIsExpanded(true)}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLDivElement).style.transform = "scale(1.02)";
            (e.currentTarget as HTMLDivElement).style.boxShadow =
              "0 16px 40px rgba(15,23,42,0.9)";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLDivElement).style.transform = "scale(1)";
            (e.currentTarget as HTMLDivElement).style.boxShadow = "none";
          }}
        >
          {renderRadar(300)}
        </div>

        {/* Legend */}
        {legend}
      </div>
    </>
  );
};

// ---------- legend helpers ----------

interface LegendItemProps {
  color: string;
  title: string;
  description: string;
}

const LegendItem: React.FC<LegendItemProps> = ({
  color,
  title,
  description,
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 6px ${color}99`,
        }}
      />
      <span
        style={{
          color: "#e5e7eb",
          fontSize: 14,
          fontWeight: 500,
        }}
      >
        {title}
      </span>
    </div>
    <span
      style={{
        color: "#9ca3af",
        fontSize: 12,
        paddingLeft: 24,
      }}
    >
      {description}
    </span>
  </div>
);

const LegendBomb: React.FC = () => (
  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontSize: 20 }}>💣</span>
      <span
        style={{
          color: "#e5e7eb",
          fontSize: 14,
          fontWeight: 500,
        }}
      >
        Payload detected
      </span>
    </div>
    <span
      style={{
        color: "#9ca3af",
        fontSize: 12,
        paddingLeft: 28,
      }}
    >
      Bomb icon replaces circle when payload is suspected
    </span>
  </div>
);

export default DroneRadar;
