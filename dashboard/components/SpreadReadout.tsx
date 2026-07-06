"use client";

import React from "react";
import { SpreadData } from "../lib/dataSource";

interface SpreadReadoutProps {
  spread: SpreadData;
}

export default function SpreadReadout({ spread }: SpreadReadoutProps) {
  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium">
          Spread Monitor
        </h2>
        {spread.isElevated && (
          <span className="px-1.5 py-0.5 text-[8px] font-mono font-bold bg-loss/15 text-loss border border-loss/20 uppercase tracking-widest animate-pulse">
            Elevated
          </span>
        )}
      </div>

      <div className="flex items-baseline space-x-2 my-2">
        <span className={`text-2xl font-mono font-bold tabular-nums ${spread.isElevated ? "text-loss" : "text-primary"}`}>
          {spread.current.toFixed(2)}
        </span>
        <span className="text-[10px] font-mono text-muted">points</span>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-2 pt-2 border-t border-hairline/50 text-[10px] font-mono text-muted">
        <div>
          <span>Mean: </span>
          <span className="text-primary tabular-nums">{spread.mean.toFixed(2)}</span>
        </div>
        <div>
          <span>p95 Limit: </span>
          <span className="text-primary tabular-nums">{spread.p95.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
