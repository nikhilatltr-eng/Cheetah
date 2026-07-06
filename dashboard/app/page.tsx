"use client";

import React, { useEffect, useState } from "react";
import { fetchLiveDashboardData, DashboardData } from "../lib/dataSource";
import PulseTrace from "../components/PulseTrace";
import EquityCurvePanel from "../components/EquityCurvePanel";
import OpenPositionsTable from "../components/OpenPositionsTable";
import RegimeStateBadge from "../components/RegimeStateBadge";
import ModelConfidenceGauge from "../components/ModelConfidenceGauge";
import DriftMonitorPanel from "../components/DriftMonitorPanel";
import SessionClock from "../components/SessionClock";
import RecentAlertsFeed from "../components/RecentAlertsFeed";
import SpreadReadout from "../components/SpreadReadout";

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);

  // Poll live metrics every 1.5 seconds
  useEffect(() => {
    let active = true;

    async function update() {
      const freshData = await fetchLiveDashboardData();
      if (active) {
        setData({ ...freshData });
      }
    }

    update();
    const interval = setInterval(update, 1500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  if (!data) {
    return (
      <div className="min-h-screen bg-graphite flex items-center justify-center font-mono text-muted text-xs">
        Connecting to metrics store...
      </div>
    );
  }

  const isGain = data.sessionPnl >= 0;

  return (
    <main className="min-h-screen bg-graphite text-primary flex flex-col p-4 md:p-6 font-sans">
      {/* Dashboard Top Header Ledger */}
      <header className="border border-hairline bg-panel p-4 mb-4 flex flex-col space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold font-serif text-brass tracking-wide leading-none">
              CHEETAH MONITOR
            </h1>
            <p className="text-[10px] text-muted font-mono tracking-widest uppercase mt-1">
              Precision Purity & Market Pulse
            </p>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex flex-col">
              <span className="text-[9px] text-muted uppercase tracking-wider">Live Gold Spot</span>
              <span className="text-sm font-mono font-bold text-primary tabular-nums">
                ${data.liveGoldPrice[data.liveGoldPrice.length - 1]?.toFixed(2)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[9px] text-muted uppercase tracking-wider">Session Return</span>
              <span className={`text-sm font-mono font-bold tabular-nums ${isGain ? "text-gain" : "text-loss"}`}>
                {isGain ? "+" : ""}${data.sessionPnl.toLocaleString("en-US", { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[9px] text-muted uppercase tracking-wider">Bot Status</span>
              <span className="text-[10px] font-mono border border-gain/35 bg-gain/5 text-gain px-1.5 py-0.5 uppercase tracking-wider">
                Active
              </span>
            </div>
          </div>
        </div>

        {/* Signature PulseTrace Element */}
        <div className="pt-2 border-t border-hairline/50">
          <div className="flex justify-between items-center text-[9px] text-muted font-mono mb-1">
            <span>LIVE SESSION EQUITY TRACE</span>
            <span className="tabular-nums">EQ: ${data.currentEquity.toFixed(2)}</span>
          </div>
          <PulseTrace data={data.liveEquity} height={48} isGain={isGain} />
        </div>
      </header>

      {/* Main Grid Panels Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Left Columns (Span 2) */}
        <div className="md:col-span-2 flex flex-col space-y-4">
          {/* Equity History Detail */}
          <div className="flex-grow">
            <EquityCurvePanel
              equityHistory={data.liveEquity}
              currentEquity={data.currentEquity}
              sessionPnl={data.sessionPnl}
              maxDrawdown={data.maxDrawdown}
              rollingSharpe={data.rollingSharpe}
            />
          </div>

          {/* Open Positions Ledger */}
          <div className="flex-grow">
            <OpenPositionsTable positions={data.openPositions} />
          </div>
        </div>

        {/* Right Column (Span 1) - Status indicators, clocks, readouts */}
        <div className="flex flex-col space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <RegimeStateBadge regime={data.regimeState} />
            <SessionClock sessions={data.sessions} />
          </div>

          <ModelConfidenceGauge
            confidence={data.modelConfidence}
            history={data.confidenceHistory}
          />

          <DriftMonitorPanel drift={data.drift} />

          <SpreadReadout spread={data.spread} />

          <RecentAlertsFeed alerts={data.alerts} />
        </div>
      </div>

      <footer className="mt-6 border-t border-hairline/40 pt-4 flex flex-col md:flex-row items-center justify-between text-[10px] text-muted font-mono">
        <span>CHEETAH TRADING SYSTEM V2.5.0</span>
        <span className="mt-1 md:mt-0">LAST TICK SYNC: {new Date().toLocaleTimeString()}</span>
      </footer>
    </main>
  );
}
