export interface TradePosition {
  id: string;
  side: "BUY" | "SELL";
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  durationMins: number;
  mode: "scalp" | "trend";
  volume: number;
}

export interface TelegramAlert {
  id: string;
  timestamp: string;
  signal: string;
  direction: "BUY" | "SELL" | "HOLD";
  confidence: number;
  regime: "trending" | "ranging" | "volatile-news";
}

export interface DriftMetrics {
  status: "on track" | "drift detected";
  liveWinRate: number;
  backtestWinRate: number;
  liveSharpe: number;
  backtestSharpe: number;
  tradesCount: number;
  reason: string;
}

export interface SpreadData {
  current: number;
  mean: number;
  p95: number;
  isElevated: boolean;
}

export interface SessionInfo {
  currentSession: "London" | "New York" | "Asian" | "London + NY Overlap" | "Off-market";
  timeToNextTransition: string; // "01h 45m"
  activeSessions: string[];
}

export interface DashboardData {
  liveEquity: number[];
  liveGoldPrice: number[];
  sessionPnl: number;
  maxDrawdown: number;
  rollingSharpe: number;
  currentEquity: number;
  openPositions: TradePosition[];
  regimeState: "trending" | "ranging" | "volatile-news";
  modelConfidence: number;
  confidenceHistory: number[];
  drift: DriftMetrics;
  sessions: SessionInfo;
  alerts: TelegramAlert[];
  spread: SpreadData;
}

// Global state container for mock data updating
let mockState: DashboardData | null = null;
let tickCount = 0;

export function getMockDashboardData(): DashboardData {
  if (!mockState) {
    // Initialize state
    const initialEquity = 10450;
    const initialGold = 2345.50;
    
    mockState = {
      liveEquity: Array.from({ length: 60 }, (_, i) => initialEquity - 50 + Math.sin(i / 5) * 40 + i * 1.5),
      liveGoldPrice: Array.from({ length: 60 }, (_, i) => initialGold - 5 + Math.sin(i / 3) * 3 + i * 0.15),
      sessionPnl: 185.50,
      maxDrawdown: 0.0135,
      rollingSharpe: 2.45,
      currentEquity: initialEquity,
      openPositions: [
        {
          id: "pos_1",
          side: "BUY",
          entryPrice: 2341.20,
          currentPrice: 2345.50,
          unrealizedPnl: 43.00,
          durationMins: 14,
          mode: "trend",
          volume: 0.10
        },
        {
          id: "pos_2",
          side: "SELL",
          entryPrice: 2347.80,
          currentPrice: 2345.50,
          unrealizedPnl: 23.00,
          durationMins: 4,
          mode: "scalp",
          volume: 0.10
        }
      ],
      regimeState: "trending",
      modelConfidence: 72.4,
      confidenceHistory: [68.5, 71.0, 73.2, 70.8, 72.4],
      drift: {
        status: "on track",
        liveWinRate: 0.585,
        backtestWinRate: 0.520,
        liveSharpe: 2.15,
        backtestSharpe: 1.85,
        tradesCount: 142,
        reason: "System performing within acceptable statistical limits."
      },
      sessions: {
        currentSession: "London + NY Overlap",
        timeToNextTransition: "02h 46m",
        activeSessions: ["London", "New York"]
      },
      alerts: [
        {
          id: "alert_1",
          timestamp: "12:54:10",
          signal: "Strong short momentum deviation detected",
          direction: "SELL",
          confidence: 72.4,
          regime: "trending"
        },
        {
          id: "alert_2",
          timestamp: "12:45:00",
          signal: "Double top validation trigger at resistance",
          direction: "SELL",
          confidence: 68.5,
          regime: "ranging"
        },
        {
          id: "alert_3",
          timestamp: "12:30:15",
          signal: "EMA crossover short-term trend entry",
          direction: "BUY",
          confidence: 71.0,
          regime: "trending"
        }
      ],
      spread: {
        current: 0.11,
        mean: 0.12,
        p95: 0.18,
        isElevated: false
      }
    };
  }

  tickCount++;
  
  // Tick updates
  const lastEquity = mockState.liveEquity[mockState.liveEquity.length - 1];
  const lastGold = mockState.liveGoldPrice[mockState.liveGoldPrice.length - 1];
  
  // Random walks
  const equityChange = (Math.random() - 0.48) * 12; // slight positive bias
  const goldChange = (Math.random() - 0.50) * 0.8;
  
  const newEquity = lastEquity + equityChange;
  const newGold = lastGold + goldChange;
  
  // Append new and discard old
  mockState.liveEquity = [...mockState.liveEquity.slice(1), newEquity];
  mockState.liveGoldPrice = [...mockState.liveGoldPrice.slice(1), newGold];
  mockState.currentEquity = newEquity;
  
  // Update open positions values
  mockState.openPositions = mockState.openPositions.map(pos => {
    let pnl = pos.unrealizedPnl;
    if (pos.side === "BUY") {
      pnl = (newGold - pos.entryPrice) * pos.volume * 1000; // Gold sizing proxy
    } else {
      pnl = (pos.entryPrice - newGold) * pos.volume * 1000;
    }
    
    return {
      ...pos,
      currentPrice: Number(newGold.toFixed(2)),
      unrealizedPnl: Number(pnl.toFixed(2)),
      durationMins: pos.durationMins + (tickCount % 10 === 0 ? 1 : 0)
    };
  });
  
  // Update sessions countdown
  const now = new Date();
  const minsRemaining = (60 - now.getMinutes()) + (tickCount % 60);
  const hrsRemaining = (3 - now.getHours() % 4 + 4) % 4;
  mockState.sessions.timeToNextTransition = `${hrsRemaining.toString().padStart(2, '0')}h ${minsRemaining.toString().padStart(2, '0')}m`;

  // Update mock spreads
  const baseSpread = 0.12;
  const currentSpread = Number((baseSpread + (Math.random() - 0.45) * 0.05).toFixed(2));
  mockState.spread.current = currentSpread;
  mockState.spread.isElevated = currentSpread > mockState.spread.p95 - 0.02;
  
  // Randomly toggle regimes or change confidence occasionally
  if (tickCount % 20 === 0) {
    const regimes: ("trending" | "ranging" | "volatile-news")[] = ["trending", "ranging", "volatile-news"];
    mockState.regimeState = regimes[Math.floor(Math.random() * regimes.length)];
    
    const newConf = Number((60 + Math.random() * 25).toFixed(1));
    mockState.modelConfidence = newConf;
    mockState.confidenceHistory = [...mockState.confidenceHistory.slice(1), newConf];
    
    // Trigger mock alert
    const directions: ("BUY" | "SELL" | "HOLD")[] = ["BUY", "SELL", "HOLD"];
    const dir = directions[Math.floor(Math.random() * directions.length)];
    const timeStr = now.toTimeString().split(' ')[0];
    mockState.alerts = [
      {
        id: `alert_${tickCount}`,
        timestamp: timeStr,
        signal: dir === "HOLD" ? "Model neutral state hold filter" : `High confidence ${dir.toLowerCase()} momentum trigger`,
        direction: dir,
        confidence: newConf,
        regime: mockState.regimeState
      },
      ...mockState.alerts.slice(0, 4)
    ];
  }

  // Session PnL update
  mockState.sessionPnl = Number((mockState.sessionPnl + equityChange).toFixed(2));
  
  return mockState;
}

export async function fetchLiveDashboardData(useMock: boolean = false): Promise<DashboardData> {
  if (useMock) {
    return getMockDashboardData();
  }
  
  try {
    const res = await fetch("/dashboard_data.json");
    if (!res.ok) {
      throw new Error("HTTP error querying dashboard_data.json");
    }
    return await res.json();
  } catch (err) {
    return getMockDashboardData();
  }
}
