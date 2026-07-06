"use client";

import React from "react";
import { DriftMetrics } from "../lib/dataSource";

interface DriftMonitorPanelProps {
  drift: DriftMetrics;
}

export default function DriftMonitorPanel({ drift }: DriftMonitorPanelProps) {
  const isOnTrack = drift.status === "on track";
  
  // High-fidelity low-saturation status styling
  const statusBadgeStyle = isOnTrack
    ? "bg-gain/10 text-gain border border-gain/20"
    : "bg-loss/10 text-loss border border-loss/20";

  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium">
          Validation Drift
        </h2>
        <span className={`px-2 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider ${statusBadgeStyle}`}>
          {drift.status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 my-2">
        <div className="flex flex-col">
          <span className="text-[9px] text-muted uppercase tracking-wider">Win Rate (Live vs Bkt)</span>
          <div className="flex items-baseline space-x-1.5 mt-0.5">
            <span className="text-sm font-mono font-medium text-primary tabular-nums">
              {(drift.liveWinRate * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] font-mono text-muted tabular-nums">
              / {(drift.backtestWinRate * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        
        <div className="flex flex-col">
          <span className="text-[9px] text-muted uppercase tracking-wider">Sharpe (Live vs Bkt)</span>
          <div className="flex items-baseline space-x-1.5 mt-0.5">
            <span className="text-sm font-mono font-medium text-primary tabular-nums">
              {drift.liveSharpe.toFixed(2)}
            </span>
            <span className="text-[10px] font-mono text-muted tabular-nums">
              / {drift.backtestSharpe.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-2 pt-2 border-t border-hairline/50 text-[10px] font-mono text-muted leading-relaxed">
        {drift.reason}
      </div>
    </div>
  );
}
