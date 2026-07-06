"use client";

import React, { useState, useMemo } from "react";
import PulseTrace from "./PulseTrace";

interface EquityCurvePanelProps {
  equityHistory: number[];
  currentEquity: number;
  sessionPnl: number;
  maxDrawdown: number;
  rollingSharpe: number;
}

type Timeframe = "24H" | "1W" | "1M" | "ALL";

export default function EquityCurvePanel({
  equityHistory,
  currentEquity,
  sessionPnl,
  maxDrawdown,
  rollingSharpe
}: EquityCurvePanelProps) {
  const [timeframe, setTimeframe] = useState<Timeframe>("1W");

  // Slice historical data based on selected timeframe
  const filteredData = useMemo(() => {
    if (!equityHistory || equityHistory.length === 0) return [];
    
    switch (timeframe) {
      case "24H":
        return equityHistory.slice(-24);
      case "1W":
        return equityHistory.slice(-60);
      case "1M":
        return equityHistory.slice(-120);
      case "ALL":
      default:
        return equityHistory;
    }
  }, [equityHistory, timeframe]);

  const isGain = sessionPnl >= 0;

  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium">
          Equity History
        </h2>
        <div className="flex space-x-1">
          {(["24H", "1W", "1M", "ALL"] as Timeframe[]).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`text-[10px] font-mono px-2 py-0.5 border ${
                timeframe === tf
                  ? "bg-hairline text-primary border-hairline"
                  : "bg-transparent text-muted border-transparent hover:border-hairline"
              } transition-colors cursor-pointer`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Mini PulseTrace inside card */}
      <div className="my-2 border-b border-hairline pb-4">
        <PulseTrace data={filteredData} height={96} isGain={isGain} />
      </div>

      {/* Stats ledger row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2">
        <div className="flex flex-col">
          <span className="text-[10px] text-muted uppercase tracking-wider">Current Equity</span>
          <span className="text-lg font-mono font-medium text-primary mt-0.5">
            ${currentEquity.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        
        <div className="flex flex-col">
          <span className="text-[10px] text-muted uppercase tracking-wider">Session P&L</span>
          <span className={`text-lg font-mono font-medium mt-0.5 ${isGain ? "text-gain" : "text-loss"}`}>
            {isGain ? "+" : ""}${sessionPnl.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        
        <div className="flex flex-col">
          <span className="text-[10px] text-muted uppercase tracking-wider">Max Drawdown</span>
          <span className="text-lg font-mono font-medium text-primary mt-0.5">
            {(maxDrawdown * 100).toFixed(2)}%
          </span>
        </div>
        
        <div className="flex flex-col">
          <span className="text-[10px] text-muted uppercase tracking-wider">Rolling Sharpe</span>
          <span className="text-lg font-mono font-medium text-primary mt-0.5">
            {rollingSharpe.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
