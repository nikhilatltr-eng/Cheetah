import React from "react";
import { render, screen } from "@testing-library/react";
import PulseTrace from "../components/PulseTrace";

describe("PulseTrace Component", () => {
  it("renders empty state when no data points are provided", () => {
    const { container } = render(<PulseTrace data={[]} height={48} />);
    expect(screen.getByText("No trace data")).toBeInTheDocument();
  });

  it("renders correctly with 1 data point", () => {
    const { container } = render(<PulseTrace data={[100]} height={48} />);
    // Should not crash and should render an SVG element
    const svgElement = container.querySelector("svg");
    expect(svgElement).toBeInTheDocument();
    
    // Check that there is a path element representing the trace
    const pathElement = container.querySelector("path");
    expect(pathElement).toBeInTheDocument();
  });

  it("renders correctly with multiple data points", () => {
    const testData = [100, 105, 102, 110, 108];
    const { container } = render(<PulseTrace data={testData} height={48} />);
    
    const svgElement = container.querySelector("svg");
    expect(svgElement).toBeInTheDocument();
    
    const pathElement = container.querySelector("path");
    expect(pathElement).toBeInTheDocument();
    expect(pathElement?.getAttribute("d")).toContain("M");
  });

  it("includes rules for reduced-motion in media overrides styling", () => {
    const testData = [100, 105, 102, 110, 108];
    const { container } = render(<PulseTrace data={testData} height={48} />);
    
    // Check the stylesheet is injected
    const styleElement = container.querySelector("style");
    expect(styleElement).toBeInTheDocument();
    expect(styleElement?.innerHTML).toContain("prefers-reduced-motion: reduce");
    expect(styleElement?.innerHTML).toContain("animation: none");
  });
});
