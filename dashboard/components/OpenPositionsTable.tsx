"use client";

import React from "react";
import { TradePosition } from "../lib/dataSource";

interface OpenPositionsTableProps {
  positions: TradePosition[];
}

export default function OpenPositionsTable({ positions }: OpenPositionsTableProps) {
  return (
    <div className="bg-panel border border-hairline p-4 flex flex-col h-full">
      <h2 className="text-xs uppercase tracking-wider text-muted font-sans font-medium mb-3">
        Open Positions
      </h2>

      <div className="overflow-x-auto flex-grow">
        {positions.length === 0 ? (
          <div className="h-32 flex items-center justify-center text-muted font-mono text-xs">
            No open positions
          </div>
        ) : (
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-hairline text-[10px] text-muted uppercase tracking-wider">
                <th className="pb-2 font-normal">Side</th>
                <th className="pb-2 font-normal text-right">Vol</th>
                <th className="pb-2 font-normal text-right">Entry</th>
                <th className="pb-2 font-normal text-right">Current</th>
                <th className="pb-2 font-normal text-right">Unrealized P&L</th>
                <th className="pb-2 font-normal text-right">Dur</th>
                <th className="pb-2 font-normal text-right">Mode</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos) => {
                const isGain = pos.unrealizedPnl >= 0;
                
                return (
                  <tr key={pos.id} className="border-b border-hairline/40 last:border-0 text-xs font-mono">
                    <td className="py-2.5">
                      <span
                        className={`inline-block px-1.5 py-0.5 text-[10px] font-bold ${
                          pos.side === "BUY"
                            ? "bg-gain/10 text-gain border border-gain/20"
                            : "bg-loss/10 text-loss border border-loss/20"
                        }`}
                      >
                        {pos.side}
                      </span>
                    </td>
                    
                    <td className="py-2.5 text-right text-primary tabular-nums">
                      {pos.volume.toFixed(2)}
                    </td>
                    
                    <td className="py-2.5 text-right text-primary tabular-nums">
                      {pos.entryPrice.toLocaleString("en-US", { minimumFractionDigits: 2 })}
                    </td>
                    
                    <td className="py-2.5 text-right text-primary tabular-nums">
                      {pos.currentPrice.toLocaleString("en-US", { minimumFractionDigits: 2 })}
                    </td>
                    
                    <td className={`py-2.5 text-right font-medium tabular-nums ${isGain ? "text-gain" : "text-loss"}`}>
                      {isGain ? "+" : ""}${pos.unrealizedPnl.toLocaleString("en-US", { minimumFractionDigits: 2 })}
                    </td>
                    
                    <td className="py-2.5 text-right text-muted tabular-nums">
                      {pos.durationMins}m
                    </td>
                    
                    <td className="py-2.5 text-right">
                      <span
                        className={`inline-block px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${
                          pos.mode === "scalp"
                            ? "border border-hairline text-muted font-normal"
                            : "border border-brass/40 text-brass font-medium"
                        }`}
                      >
                        {pos.mode}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
