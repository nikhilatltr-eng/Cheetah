"use client";

import React from "react";

interface RegimeStateBadgeProps {
  regime: "trending" | "ranging" | "volatile-news";
}

export default function RegimeStateBadge({ regime }: RegimeStateBadgeProps) {
  // Determine dot color based on regime state
  const dotColorClass = 
    regime === "trending" 
      ? "bg-gain" 
      : regime === "ranging" 
        ? "bg-brass" 
        : "bg-loss";

  const label = 
    regime === "trending" 
      ? "Trending" 
      : regime === "ranging" 
        ? "Ranging" 
        : "Volatile News";

  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium mb-2">
        Market Regime
      </h2>
      
      <div className="flex items-center space-x-2 py-1">
        <span className={`h-2.5 w-2.5 rounded-full ${dotColorClass}`} />
        <span className="text-sm font-mono font-medium text-primary uppercase tracking-wider">
          {label}
        </span>
      </div>
      
      <div className="text-[10px] text-muted font-sans mt-2">
        HMM State Classifier Output
      </div>
    </div>
  );
}
