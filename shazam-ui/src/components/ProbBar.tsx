import React from "react";


type Props = { label: string; value: number };
export const ProbBar: React.FC<Props> = ({ label, value }) => {
const pct = Math.max(0, Math.min(1, value));
return (
<div className="prob">
<span className="prob-label">{label}</span>
<div className="prob-rail">
<div className="prob-fill" style={{ width: `${pct * 100}%` }} />
</div>
<span className="prob-val">{(pct * 100).toFixed(1)}%</span>
</div>
);
};