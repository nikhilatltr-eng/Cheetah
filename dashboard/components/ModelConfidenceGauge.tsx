"use client";

import React, { useMemo } from "react";

interface ModelConfidenceGaugeProps {
  confidence: number;
  history: number[];
}

export default function ModelConfidenceGauge({ confidence, history }: ModelConfidenceGaugeProps) {
  // Map history to sparkline points
  const sparklinePoints = useMemo(() => {
    if (!history || history.length < 2) return "";
    const width = 72;
    const height = 24;
    const padding = 2;
    const usableHeight = height - padding * 2;
    
    const minVal = Math.min(...history);
    const maxVal = Math.max(...history);
    const range = maxVal - minVal;
    
    return history
      .map((val, idx) => {
        const x = (idx / (history.length - 1)) * width;
        const pct = range > 0 ? (val - minVal) / range : 0.5;
        const y = height - (pct * usableHeight + padding);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [history]);

  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium mb-3">
        Entry Confidence
      </h2>

      <div className="flex items-center space-x-6 py-1">
        {/* Hallmark Circular Badge */}
        <div className="relative h-16 w-16 flex-shrink-0 flex items-center justify-center border-2 border-brass rounded-full select-none">
          {/* Subtle outer stamp style border dash */}
          <div className="absolute inset-[2px] border border-dashed border-brass/50 rounded-full" />
          <div className="flex flex-col items-center justify-center text-center">
            <span className="text-[9px] uppercase tracking-widest text-brass font-bold leading-none mb-0.5">XAU</span>
            <span className="text-sm font-mono font-bold text-brass leading-none tabular-nums">
              {confidence.toFixed(1)}
            </span>
            <span className="text-[8px] uppercase text-brass/70 font-semibold leading-none mt-0.5">CONF</span>
          </div>
        </div>

        {/* Confidence Sparkline */}
        <div className="flex flex-col justify-center flex-grow">
          <span className="text-[10px] text-muted uppercase tracking-wider mb-1.5">Confidence Trend</span>
          {history && history.length >= 2 ? (
            <div className="flex items-center space-x-2">
              <svg width="72" height="24" className="overflow-visible">
                <polyline
                  fill="none"
                  stroke="#C9A66B"
                  strokeWidth="1.5"
                  points={sparklinePoints}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Last dot */}
                {history.length > 0 && (
                  <circle
                    cx="72"
                    cy={sparklinePoints.split(" ").pop()?.split(",")[1]}
                    r="2"
                    fill="#C9A66B"
                  />
                )}
              </svg>
              <span className="text-[10px] font-mono text-muted tabular-nums">
                {history[history.length - 1]?.toFixed(1)}%
              </span>
            </div>
          ) : (
            <span className="text-[10px] font-mono text-muted">No history</span>
          )}
        </div>
      </div>
    </div>
  );
}
