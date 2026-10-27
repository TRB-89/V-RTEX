import React, { useMemo, useState } from 'react';
import { Download, Info, RotateCcw } from 'lucide-react';
import type { ProblemFormulation, SolverResult } from '../types/solver';

interface FeasibleRegionPlot2DProps {
  formulation: ProblemFormulation | null;
  activeResult: SolverResult | null;
}

interface Point2D {
  x: number;
  y: number;
}

export const FeasibleRegionPlot2D: React.FC<FeasibleRegionPlot2DProps> = ({
  formulation,
  activeResult,
}) => {
  const [zoomFactor, setZoomFactor] = useState<number>(1.0);

  // Check if formulation is 2-variable
  const n = formulation?.c?.length ?? 0;
  const isTwoVariable = n === 2;

  // Geometry calculations
  const {
    lines,
    vertices,
    optimalPoint,
    domain,
    objectiveContours,
    gradientVector,
  } = useMemo(() => {
    if (!formulation || !isTwoVariable) {
      return {
        isValid: false,
        lines: [],
        vertices: [],
        optimalPoint: null,
        domain: { xMin: 0, xMax: 10, yMin: 0, yMax: 10 },
        objectiveContours: [],
        gradientVector: null,
      };
    }

    const c = formulation.c;
    const A = formulation.A;
    const b = formulation.b;
    const senses = formulation.constraint_types || Array(A.length).fill('LEQ');

    // Build half-plane lines: [a1, a2, rhs, sense, name]
    interface ConstraintLine {
      id: string;
      name: string;
      a1: number;
      a2: number;
      rhs: number;
      sense: 'LEQ' | 'GEQ' | 'EQ';
    }

    const allLines: ConstraintLine[] = [
      { id: 'bound_x1', name: 'x1 ≥ 0', a1: -1, a2: 0, rhs: 0, sense: 'LEQ' },
      { id: 'bound_x2', name: 'x2 ≥ 0', a1: 0, a2: -1, rhs: 0, sense: 'LEQ' },
    ];

    A.forEach((row, idx) => {
      allLines.push({
        id: `c_${idx + 1}`,
        name: formulation.constraintNames?.[idx] || `c${idx + 1}`,
        a1: row[0] ?? 0,
        a2: row[1] ?? 0,
        rhs: b[idx] ?? 0,
        sense: senses[idx] as 'LEQ' | 'GEQ' | 'EQ',
      });
    });

    // Compute pairwise intersection points
    const candidatePoints: Point2D[] = [];
    const tol = 1e-5;

    for (let i = 0; i < allLines.length; i++) {
      for (let j = i + 1; j < allLines.length; j++) {
        const L1 = allLines[i];
        const L2 = allLines[j];
        const det = L1.a1 * L2.a2 - L1.a2 * L2.a1;
        if (Math.abs(det) > 1e-9) {
          const px = (L1.rhs * L2.a2 - L1.a2 * L2.rhs) / det;
          const py = (L1.a1 * L2.rhs - L1.rhs * L2.a1) / det;
          if (isFinite(px) && isFinite(py)) {
            candidatePoints.push({ x: px, y: py });
          }
        }
      }
    }

    // Filter points that satisfy all constraints
    const feasiblePoints = candidatePoints.filter((pt) => {
      // Must be non-negative
      if (pt.x < -tol || pt.y < -tol) return false;
      // Must satisfy every constraint
      for (const line of allLines) {
        const val = line.a1 * pt.x + line.a2 * pt.y;
        if (line.sense === 'LEQ' && val > line.rhs + tol) return false;
        if (line.sense === 'GEQ' && val < line.rhs - tol) return false;
        if (line.sense === 'EQ' && Math.abs(val - line.rhs) > tol) return false;
      }
      return true;
    });

    // Remove duplicates
    const uniqueFeasible: Point2D[] = [];
    feasiblePoints.forEach((pt) => {
      const exists = uniqueFeasible.some(
        (u) => Math.abs(u.x - pt.x) < 1e-4 && Math.abs(u.y - pt.y) < 1e-4
      );
      if (!exists) uniqueFeasible.push(pt);
    });

    // Sort feasible polygon vertices angularly around centroid
    let sortedVertices: Point2D[] = [];
    if (uniqueFeasible.length >= 3) {
      const cx = uniqueFeasible.reduce((sum, p) => sum + p.x, 0) / uniqueFeasible.length;
      const cy = uniqueFeasible.reduce((sum, p) => sum + p.y, 0) / uniqueFeasible.length;
      sortedVertices = [...uniqueFeasible].sort((p1, p2) => {
        const angle1 = Math.atan2(p1.y - cy, p1.x - cx);
        const angle2 = Math.atan2(p2.y - cy, p2.x - cx);
        return angle1 - angle2;
      });
    } else {
      sortedVertices = uniqueFeasible;
    }

    // Determine domain bounds
    let maxX = 5;
    let maxY = 5;
    sortedVertices.forEach((p) => {
      if (p.x > maxX && p.x < 1e6) maxX = p.x;
      if (p.y > maxY && p.y < 1e6) maxY = p.y;
    });

    // Check optimal point
    let optPt: Point2D | null = null;
    if (activeResult?.variable_values && activeResult.variable_values.length >= 2) {
      optPt = {
        x: activeResult.variable_values[0],
        y: activeResult.variable_values[1],
      };
      if (optPt.x > maxX) maxX = optPt.x;
      if (optPt.y > maxY) maxY = optPt.y;
    }

    const domainXMax = Math.max(maxX * 1.35 * zoomFactor, 6);
    const domainYMax = Math.max(maxY * 1.35 * zoomFactor, 6);

    // Compute objective contours: c1 * x1 + c2 * x2 = k
    const optObj = activeResult?.objective_value ?? 0;
    const contours: { k: number; p1: Point2D; p2: Point2D }[] = [];
    const kValues = [optObj * 0.4, optObj * 0.7, optObj, optObj * 1.25].filter(
      (k) => !isNaN(k) && isFinite(k)
    );

    kValues.forEach((kVal) => {
      if (Math.abs(c[0]) > 1e-7 && Math.abs(c[1]) > 1e-7) {
        // Line x2 = (k - c1*x1) / c2
        const p1 = { x: 0, y: kVal / c[1] };
        const p2 = { x: kVal / c[0], y: 0 };
        contours.push({ k: kVal, p1, p2 });
      }
    });

    // Gradient vector direction [c1, c2]
    const gradMag = Math.hypot(c[0], c[1]);
    const gradVec = gradMag > 0 ? { dx: (c[0] / gradMag) * 1.5, dy: (c[1] / gradMag) * 1.5 } : null;

    return {
      isValid: true,
      lines: allLines.filter((l) => !l.id.startsWith('bound_')),
      vertices: sortedVertices,
      optimalPoint: optPt,
      domain: { xMin: 0, xMax: domainXMax, yMin: 0, yMax: domainYMax },
      objectiveContours: contours,
      gradientVector: gradVec,
    };
  }, [formulation, isTwoVariable, activeResult, zoomFactor]);

  if (!isTwoVariable) {
    return (
      <div className="p-8 bg-obsidian border border-graphite-border rounded-sm text-center space-y-3 font-mono text-xs">
        <Info className="w-8 h-8 text-violet-muted mx-auto" />
        <span className="font-semibold text-chrome text-sm block">
          2D Geometric Visualization Unavailable
        </span>
        <p className="text-violet-muted max-w-md mx-auto leading-relaxed">
          The current formulation contains {n} decision variables. 2D linear constraint boundaries and feasible polyhedra are geometrically defined only for two-variable models ($n = 2$). Arbitrary projections of high-dimensional polyhedra distort true constraint geometry.
        </p>
      </div>
    );
  }

  // Coordinate transformation: Cartesian (x, y) -> SVG (svgX, svgY)
  const svgWidth = 620;
  const svgHeight = 420;
  const padLeft = 55;
  const padBottom = 45;
  const padRight = 30;
  const padTop = 30;

  const plotW = svgWidth - padLeft - padRight;
  const plotH = svgHeight - padTop - padBottom;

  const toSvgX = (x: number) => padLeft + (x / domain.xMax) * plotW;
  const toSvgY = (y: number) => padTop + plotH - (y / domain.yMax) * plotH;

  // Build SVG polygon points
  const polygonPointsStr = vertices
    .map((pt) => `${toSvgX(pt.x).toFixed(1)},${toSvgY(pt.y).toFixed(1)}`)
    .join(' ');

  // Export SVG handler
  const handleExportSvg = () => {
    const svgElem = document.getElementById('vortex-feasible-svg');
    if (!svgElem) return;
    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svgElem);
    const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vortex_feasible_region_2d_${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      
      {/* Title & Action Bar */}
      <div className="flex flex-wrap items-center justify-between border-b border-graphite-border pb-2 gap-2">
        <div>
          <span className="font-semibold text-chrome text-xs uppercase tracking-wider block">
            2D Feasible Polyhedron &amp; Linear Objective Geometry
          </span>
          <span className="text-[10px] text-violet-muted">
            Constraint half-spaces, feasible region convex hull, objective contours, and optimal vertex x*.
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setZoomFactor((z) => Math.max(z * 0.8, 0.5))}
            className="px-2 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px]"
            title="Zoom In"
          >
            +
          </button>
          <button
            onClick={() => setZoomFactor((z) => Math.min(z * 1.25, 3.0))}
            className="px-2 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px]"
            title="Zoom Out"
          >
            -
          </button>
          <button
            onClick={() => setZoomFactor(1.0)}
            className="px-2 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px] flex items-center gap-1"
            title="Reset Scale"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
          <button
            onClick={handleExportSvg}
            className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px] flex items-center gap-1"
            title="Export Vector Graphic (SVG)"
          >
            <Download className="w-3 h-3 text-electric" />
            <span>SVG</span>
          </button>
        </div>
      </div>

      {/* SVG Canvas Container */}
      <div className="bg-obsidian rounded-sm border border-graphite-border overflow-hidden flex justify-center items-center py-2">
        <svg
          id="vortex-feasible-svg"
          width={svgWidth}
          height={svgHeight}
          className="select-none"
        >
          {/* Background Grid Lines */}
          {[0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
            const xVal = domain.xMax * frac;
            const yVal = domain.yMax * frac;
            return (
              <g key={idx}>
                <line
                  x1={toSvgX(xVal)}
                  y1={padTop}
                  x2={toSvgX(xVal)}
                  y2={padTop + plotH}
                  stroke="#1B1C27"
                  strokeWidth="1"
                />
                <line
                  x1={padLeft}
                  y1={toSvgY(yVal)}
                  x2={padLeft + plotW}
                  y2={toSvgY(yVal)}
                  stroke="#1B1C27"
                  strokeWidth="1"
                />
                {/* Tick Labels */}
                <text
                  x={toSvgX(xVal)}
                  y={padTop + plotH + 18}
                  fill="#77768F"
                  fontSize="9"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  {xVal.toFixed(1)}
                </text>
                <text
                  x={padLeft - 8}
                  y={toSvgY(yVal) + 3}
                  fill="#77768F"
                  fontSize="9"
                  fontFamily="monospace"
                  textAnchor="end"
                >
                  {yVal.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Axes */}
          <line
            x1={padLeft}
            y1={padTop + plotH}
            x2={padLeft + plotW}
            y2={padTop + plotH}
            stroke="#77768F"
            strokeWidth="1.5"
          />
          <line
            x1={padLeft}
            y1={padTop}
            x2={padLeft}
            y2={padTop + plotH}
            stroke="#77768F"
            strokeWidth="1.5"
          />

          {/* Axis Labels */}
          <text
            x={padLeft + plotW - 5}
            y={padTop + plotH + 32}
            fill="#E7E8EF"
            fontSize="11"
            fontFamily="monospace"
            fontWeight="bold"
            textAnchor="end"
          >
            {formulation?.variableNames?.[0] || 'x₁'} →
          </text>
          <text
            x={padLeft}
            y={padTop - 12}
            fill="#E7E8EF"
            fontSize="11"
            fontFamily="monospace"
            fontWeight="bold"
            textAnchor="start"
          >
            ↑ {formulation?.variableNames?.[1] || 'x₂'}
          </text>

          {/* Feasible Region Shaded Polygon */}
          {vertices.length >= 3 && (
            <polygon
              points={polygonPointsStr}
              fill="rgba(37, 20, 237, 0.22)"
              stroke="#2514ED"
              strokeWidth="2"
            />
          )}

          {/* Linear Constraint Boundary Lines */}
          {lines.map((line) => {
            let x1 = 0;
            let y1 = 0;
            let x2 = domain.xMax;
            let y2 = 0;

            if (Math.abs(line.a2) > 1e-7) {
              y1 = (line.rhs - line.a1 * x1) / line.a2;
              y2 = (line.rhs - line.a1 * x2) / line.a2;
            } else if (Math.abs(line.a1) > 1e-7) {
              x1 = line.rhs / line.a1;
              x2 = x1;
              y1 = 0;
              y2 = domain.yMax;
            }

            return (
              <g key={line.id}>
                <line
                  x1={toSvgX(x1)}
                  y1={toSvgY(y1)}
                  x2={toSvgX(x2)}
                  y2={toSvgY(y2)}
                  stroke="#5A5872"
                  strokeWidth="1.5"
                  strokeDasharray="4 2"
                />
              </g>
            );
          })}

          {/* Objective Contours (dashed) */}
          {objectiveContours.map((cnt, idx) => (
            <line
              key={idx}
              x1={toSvgX(cnt.p1.x)}
              y1={toSvgY(cnt.p1.y)}
              x2={toSvgX(cnt.p2.x)}
              y2={toSvgY(cnt.p2.y)}
              stroke="#9B99AF"
              strokeWidth="1"
              strokeDasharray="3 3"
              opacity="0.6"
            />
          ))}

          {/* Objective Gradient Vector */}
          {gradientVector && (
            <line
              x1={toSvgX(1.5)}
              y1={toSvgY(1.5)}
              x2={toSvgX(1.5 + gradientVector.dx)}
              y2={toSvgY(1.5 + gradientVector.dy)}
              stroke="#2514ED"
              strokeWidth="2"
              markerEnd="url(#arrow)"
            />
          )}

          {/* Optimal Solution Vertex Marker */}
          {optimalPoint && (
            <g>
              <circle
                cx={toSvgX(optimalPoint.x)}
                cy={toSvgY(optimalPoint.y)}
                r="6"
                fill="#2514ED"
                stroke="#E7E8EF"
                strokeWidth="2"
              />
              <circle
                cx={toSvgX(optimalPoint.x)}
                cy={toSvgY(optimalPoint.y)}
                r="12"
                fill="none"
                stroke="#2514ED"
                strokeWidth="1"
                opacity="0.5"
              />
              {/* Callout Box */}
              <rect
                x={toSvgX(optimalPoint.x) + 10}
                y={toSvgY(optimalPoint.y) - 26}
                width="140"
                height="22"
                fill="#0C0D13"
                stroke="#2514ED"
                strokeWidth="1"
                rx="2"
              />
              <text
                x={toSvgX(optimalPoint.x) + 16}
                y={toSvgY(optimalPoint.y) - 11}
                fill="#E7E8EF"
                fontSize="10"
                fontFamily="monospace"
                fontWeight="bold"
              >
                x* = ({optimalPoint.x.toFixed(2)}, {optimalPoint.y.toFixed(2)})
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Legend & Feasible Region Summary */}
      <div className="flex flex-wrap items-center justify-between text-[11px] text-violet-muted pt-1 gap-2">
        <div className="flex items-center space-x-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-electric/25 border border-electric inline-block" />
            Feasible Region Polyhedron ({vertices.length} vertices)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 border-t border-dashed border-violet-muted inline-block" />
            Constraint Boundaries (Ax = b)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-electric border border-chrome inline-block" />
            Optimal Vertex x*
          </span>
        </div>
        {activeResult?.objective_value !== undefined && (
          <span className="text-chrome font-semibold">
            Optimal Z* = {Number(activeResult.objective_value).toFixed(4)}
          </span>
        )}
      </div>

    </div>
  );
};
