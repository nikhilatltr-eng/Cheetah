"use client";

import React from "react";
import { TelegramAlert } from "../lib/dataSource";

interface RecentAlertsFeedProps {
  alerts: TelegramAlert[];
}

export default function RecentAlertsFeed({ alerts }: RecentAlertsFeedProps) {
  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col h-full">
      <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium mb-3">
        Alert History
      </h2>

      <div className="overflow-y-auto flex-grow max-h-[160px] pr-1">
        {alerts.length === 0 ? (
          <div className="h-32 flex items-center justify-center text-muted font-mono text-xs">
            No recent alerts
          </div>
        ) : (
          <div className="space-y-2.5">
            {alerts.map((alert) => {
              const isBuy = alert.direction === "BUY";
              const isSell = alert.direction === "SELL";
              
              const directionBadgeStyle = isBuy
                ? "text-gain font-bold"
                : isSell
                  ? "text-loss font-bold"
                  : "text-muted font-normal";

              return (
                <div 
                  key={alert.id} 
                  className="flex items-start space-x-3 text-xs border-b border-hairline/30 pb-2 last:border-0 last:pb-0"
                >
                  <span className="font-mono text-muted text-[10px] tabular-nums mt-0.5 select-none">
                    [{alert.timestamp}]
                  </span>
                  
                  <div className="flex-grow flex flex-col">
                    <span className="text-primary font-sans leading-tight">
                      {alert.signal}
                    </span>
                    
                    <div className="flex items-center space-x-2 mt-1 text-[9px] font-mono text-muted">
                      <span>Dir: <span className={directionBadgeStyle}>{alert.direction}</span></span>
                      <span>•</span>
                      <span className="tabular-nums">Conf: {alert.confidence.toFixed(1)}%</span>
                      <span>•</span>
                      <span className="uppercase tracking-wider">Regime: {alert.regime}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
