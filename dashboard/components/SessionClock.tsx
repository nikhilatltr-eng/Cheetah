"use client";

import React from "react";
import { SessionInfo } from "../lib/dataSource";

interface SessionClockProps {
  sessions: SessionInfo;
}

export default function SessionClock({ sessions }: SessionClockProps) {
  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col justify-between h-full">
      <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium mb-3">
        Session Indicator
      </h2>

      <div className="flex flex-col space-y-2">
        <div className="flex items-center space-x-1.5 flex-wrap">
          {sessions.activeSessions.map((s) => (
            <span
              key={s}
              className="px-1.5 py-0.5 text-[9px] font-mono border border-hairline text-primary uppercase tracking-wider"
            >
              {s}
            </span>
          ))}
          {sessions.activeSessions.length === 0 && (
            <span className="px-1.5 py-0.5 text-[9px] font-mono border border-dashed border-hairline text-muted uppercase">
              Off-market
            </span>
          )}
        </div>
        
        <div className="flex flex-col pt-1">
          <span className="text-[9px] text-muted uppercase tracking-wider">Time to Transition</span>
          <span className="text-lg font-mono font-bold text-primary mt-0.5 tabular-nums">
            {sessions.timeToNextTransition}
          </span>
        </div>
      </div>

      <div className="text-[9px] font-sans text-muted mt-2">
        Validated sessions dictate trading rules
      </div>
    </div>
  );
}
