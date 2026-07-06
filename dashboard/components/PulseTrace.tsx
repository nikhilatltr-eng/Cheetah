"use client";

import React, { useMemo } from "react";

interface PulseTraceProps {
  data: number[];
  height?: number;
  isGain?: boolean;
}

export default function PulseTrace({ data, height = 64, isGain = true }: PulseTraceProps) {
  const points = useMemo(() => {
    if (!data || data.length === 0) return { path: "", lastX: 0, lastY: 0, width: 0 };
    
    // Width can be dynamic or statically 100% inside SVG viewBox
    const width = 800;
    const paddingY = 8;
    const usableHeight = height - paddingY * 2;
    
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal;
    
    const mapped = data.map((val, idx) => {
      const x = (idx / Math.max(1, data.length - 1)) * width;
      const pct = range > 0 ? (val - minVal) / range : 0.5;
      // Invert Y so higher value is at top
      const y = height - (pct * usableHeight + paddingY);
      return { x, y };
    });
    
    const path = mapped.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
    const lastPoint = mapped[mapped.length - 1] || { x: 0, y: 0 };
    
    return { path, lastX: lastPoint.x, lastY: lastPoint.y, width };
  }, [data, height]);

  if (!data || data.length === 0) {
    return (
      <div 
        style={{ height }} 
        className="w-full flex items-center justify-center border border-dashed border-hairline text-muted text-xs font-mono"
      >
        No trace data
      </div>
    );
  }

  // Choose stroke color classes with CSS transition support
  const strokeColorClass = isGain ? "stroke-gain" : "stroke-loss";
  const glowColorClass = isGain ? "fill-gain" : "fill-loss";

  return (
    <div className="w-full relative select-none">
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes pulse-radius {
          0% { r: 3px; opacity: 0.9; }
          100% { r: 14px; opacity: 0; }
        }
        .animate-pulse-radius {
          animation: pulse-radius 2.5s infinite cubic-bezier(0.25, 1, 0.5, 1);
        }
        @media (prefers-reduced-motion: reduce) {
          .animate-pulse-radius {
            animation: none;
            opacity: 0.3;
            r: 6px;
          }
        }
      `}} />
      <svg 
        viewBox={`0 0 ${points.width} ${height}`} 
        className="w-full overflow-visible"
        style={{ height }}
        preserveAspectRatio="none"
      >
        {/* Draw the thin trace line */}
        <path
          d={points.path}
          fill="none"
          strokeWidth="1.5"
          className={`transition-all duration-1000 ease-in-out ${strokeColorClass}`}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Glow outer pulse dot (disabled if prefers-reduced-motion is active via CSS keyframe override) */}
        <circle
          cx={points.lastX}
          cy={points.lastY}
          className={`animate-pulse-radius ${glowColorClass}`}
          pointerEvents="none"
        />
        
        {/* Core solid center dot */}
        <circle
          cx={points.lastX}
          cy={points.lastY}
          r="3"
          className={glowColorClass}
          pointerEvents="none"
        />
      </svg>
    </div>
  );
}
