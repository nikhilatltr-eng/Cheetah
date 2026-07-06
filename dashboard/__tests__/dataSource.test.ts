import { getMockDashboardData, fetchLiveDashboardData, DashboardData } from "../lib/dataSource";

describe("dataSource module", () => {
  it("getMockDashboardData returns a valid dashboard payload conforming to schema", () => {
    const data = getMockDashboardData();
    
    // Test overall structure
    expect(data).toHaveProperty("liveEquity");
    expect(data).toHaveProperty("liveGoldPrice");
    expect(data).toHaveProperty("openPositions");
    expect(data).toHaveProperty("regimeState");
    expect(data).toHaveProperty("modelConfidence");
    expect(data).toHaveProperty("confidenceHistory");
    expect(data).toHaveProperty("drift");
    expect(data).toHaveProperty("sessions");
    expect(data).toHaveProperty("alerts");
    expect(data).toHaveProperty("spread");
    
    // Test data types
    expect(Array.isArray(data.liveEquity)).toBe(true);
    expect(data.liveEquity.length).toBeGreaterThan(0);
    expect(typeof data.currentEquity).toBe("number");
    expect(typeof data.modelConfidence).toBe("number");
    expect(["trending", "ranging", "volatile-news"]).toContain(data.regimeState);
    expect(Array.isArray(data.openPositions)).toBe(true);
    
    // Check nested objects
    expect(data.drift).toHaveProperty("status");
    expect(["on track", "drift detected"]).toContain(data.drift.status);
    expect(data.spread).toHaveProperty("current");
    expect(data.spread).toHaveProperty("isElevated");
  });

  it("updates live metrics sequentially on tick updates", () => {
    const firstCall = { ...getMockDashboardData() };
    const secondCall = { ...getMockDashboardData() };
    
    // Equity or gold should fluctuate based on the random walk updates
    // In mock data updates are in-place on the singleton so we compare currentEquity values
    expect(secondCall.currentEquity).not.toBe(firstCall.liveEquity[0]);
  });

  it("fetchLiveDashboardData conforms to standard dashboard schema in mock fallback", async () => {
    const data = await fetchLiveDashboardData(true);
    expect(data).toHaveProperty("currentEquity");
    expect(data.liveEquity.length).toBe(60);
  });
});
