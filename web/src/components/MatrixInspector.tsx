import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { 
  Grid, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Download, 
  Table, 
  Layers, 
  FileSpreadsheet
} from 'lucide-react';
import type { SolverResult, ProblemFormulation } from '../types/solver';

interface MatrixInspectorProps {
  activeResult?: SolverResult | null;
  formulation?: ProblemFormulation | null;
}

export const MatrixInspector: React.FC<MatrixInspectorProps> = ({ 
  activeResult, 
  formulation 
}) => {
  // Matrix selection: 'A' (Constraints) or 'Q' (Quadratic Hessian)
  const [selectedMatrixType, setSelectedMatrixType] = useState<'A' | 'Q'>('A');
  const [viewMode, setViewMode] = useState<'SPY' | 'DENSE' | 'COMPRESSED'>('SPY');
  const [compressedFormat, setCompressedFormat] = useState<'CSR' | 'CSC'>('CSR');

  // Canvas zoom & pan state
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Cell hover inspection state
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    val: number;
    rowName: string;
    colName: string;
  } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Active matrix data derivation
  const { currentMatrix, matrixLabel, rowNames, colNames } = useMemo(() => {
    if (selectedMatrixType === 'Q' && formulation?.Q && formulation.Q.length > 0) {
      const q = formulation.Q;
      const n = q.length;
      const rNames = formulation.variableNames || Array.from({ length: n }, (_, i) => `x${i + 1}`);
      return {
        currentMatrix: q,
        matrixLabel: `Hessian Matrix Q (${n} × ${n})`,
        rowNames: rNames,
        colNames: rNames,
      };
    }

    // Default to constraint matrix A
    if (formulation?.A && formulation.A.length > 0) {
      const a = formulation.A;
      const m = a.length;
      const n = a[0]?.length || 0;
      const rNames = formulation.constraintNames || Array.from({ length: m }, (_, i) => `c${i + 1}`);
      const cNames = formulation.variableNames || Array.from({ length: n }, (_, j) => `x${j + 1}`);
      return {
        currentMatrix: a,
        matrixLabel: `Constraint Matrix A (${m} × ${n})`,
        rowNames: rNames,
        colNames: cNames,
      };
    }

    // Fallback reference matrix if no problem formulation is loaded
    const fallback = [
      [1.0, 0.0, 0.0, 2.5],
      [0.0, 2.0, -1.4, 0.0],
      [3.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 5.0, 1.0],
      [0.0, -3.0, 0.0, 4.0],
    ];
    return {
      currentMatrix: fallback,
      matrixLabel: 'Reference Benchmark Matrix (5 × 4)',
      rowNames: ['c1', 'c2', 'c3', 'c4', 'c5'],
      colNames: ['x1', 'x2', 'x3', 'x4'],
    };
  }, [formulation, selectedMatrixType]);

  const m = currentMatrix.length;
  const n = currentMatrix[0]?.length || 0;
  const totalElements = m * n;

  // Compute canonical CSR and CSC representations
  const { nnz, density, sparsity, csr, csc } = useMemo(() => {
    let nonZeroCount = 0;

    // CSR structures
    const csrValues: number[] = [];
    const csrColIndices: number[] = [];
    const csrRowPointers: number[] = [0];

    for (let i = 0; i < m; ++i) {
      for (let j = 0; j < n; ++j) {
        const val = currentMatrix[i]?.[j] ?? 0;
        if (Math.abs(val) > 1e-12) {
          nonZeroCount++;
          csrValues.push(val);
          csrColIndices.push(j);
        }
      }
      csrRowPointers.push(csrValues.length);
    }

    // CSC structures
    const cscValues: number[] = [];
    const cscRowIndices: number[] = [];
    const cscColPointers: number[] = [0];

    for (let j = 0; j < n; ++j) {
      for (let i = 0; i < m; ++i) {
        const val = currentMatrix[i]?.[j] ?? 0;
        if (Math.abs(val) > 1e-12) {
          cscValues.push(val);
          cscRowIndices.push(i);
        }
      }
      cscColPointers.push(cscValues.length);
    }

    const d = totalElements > 0 ? nonZeroCount / totalElements : 0;
    const s = 1.0 - d;

    return {
      nnz: nonZeroCount,
      density: d,
      sparsity: s,
      csr: {
        values: csrValues,
        colIndices: csrColIndices,
        rowPointers: csrRowPointers,
      },
      csc: {
        values: cscValues,
        rowIndices: cscRowIndices,
        colPointers: cscColPointers,
      },
    };
  }, [currentMatrix, m, n, totalElements]);

  // Render Matrix Spy Plot onto HTML5 Canvas
  const drawSpyPlot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#0C0D13'; // Obsidian
    ctx.fillRect(0, 0, width, height);

    if (m === 0 || n === 0) return;

    // Calculate cell dimensions with zoom & pan
    const padding = 40;
    const availW = width - padding * 2;
    const availH = height - padding * 2;
    const baseCellW = availW / n;
    const baseCellH = availH / m;

    const cellW = baseCellW * zoomLevel;
    const cellH = baseCellH * zoomLevel;

    ctx.save();
    ctx.translate(padding + panOffset.x, padding + panOffset.y);

    // Grid lines for small matrices
    if (cellW > 6 && cellH > 6) {
      ctx.strokeStyle = '#1F202B'; // Graphite line
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let j = 0; j <= n; j++) {
        ctx.moveTo(j * cellW, 0);
        ctx.lineTo(j * cellW, m * cellH);
      }
      for (let i = 0; i <= m; i++) {
        ctx.moveTo(0, i * cellH);
        ctx.lineTo(n * cellW, i * cellH);
      }
      ctx.stroke();
    }

    // Outer boundary
    ctx.strokeStyle = '#38384A';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0, 0, n * cellW, m * cellH);

    // Highlight hovered row & column
    if (hoveredCell) {
      ctx.fillStyle = 'rgba(37, 20, 237, 0.08)'; // Electric tint
      ctx.fillRect(0, hoveredCell.row * cellH, n * cellW, cellH);
      ctx.fillRect(hoveredCell.col * cellW, 0, cellW, m * cellH);
    }

    // Draw non-zero entries
    const radius = Math.max(1.5, Math.min(cellW, cellH) * 0.38);

    for (let i = 0; i < m; ++i) {
      for (let j = 0; j < n; ++j) {
        const val = currentMatrix[i]?.[j] ?? 0;
        if (Math.abs(val) > 1e-12) {
          const cx = j * cellW + cellW / 2;
          const cy = i * cellH + cellH / 2;

          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);

          // Color by sign: Electric blue for positive, warm amber for negative
          if (val > 0) {
            ctx.fillStyle = '#2514ED'; // Electric Blue
          } else {
            ctx.fillStyle = '#F59E0B'; // Amber
          }
          ctx.fill();

          // If zoomed in sufficiently, draw numerical value
          if (cellW >= 42 && cellH >= 24) {
            ctx.fillStyle = '#E7E8EF';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(val.toFixed(2), cx, cy);
          }
        }
      }
    }

    ctx.restore();
  }, [currentMatrix, m, n, zoomLevel, panOffset, hoveredCell]);

  useEffect(() => {
    drawSpyPlot();
  }, [drawSpyPlot]);

  // Handle canvas mouse move for interactive cell inspection
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isDragging) {
      setPanOffset({
        x: panOffset.x + (e.clientX - dragStart.x),
        y: panOffset.y + (e.clientY - dragStart.y),
      });
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const padding = 40;
    const availW = canvas.width - padding * 2;
    const availH = canvas.height - padding * 2;
    const cellW = (availW / n) * zoomLevel;
    const cellH = (availH / m) * zoomLevel;

    const relX = mouseX - (padding + panOffset.x);
    const relY = mouseY - (padding + panOffset.y);

    if (relX >= 0 && relX < n * cellW && relY >= 0 && relY < m * cellH) {
      const col = Math.floor(relX / cellW);
      const row = Math.floor(relY / cellH);
      if (row >= 0 && row < m && col >= 0 && col < n) {
        const val = currentMatrix[row]?.[col] ?? 0;
        setHoveredCell({
          row,
          col,
          val,
          rowName: rowNames[row] || `Row ${row + 1}`,
          colName: colNames[col] || `Col ${col + 1}`,
        });
        return;
      }
    }
    setHoveredCell(null);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleResetView = () => {
    setZoomLevel(1.0);
    setPanOffset({ x: 0, y: 0 });
    setHoveredCell(null);
  };

  // Export Matrix Data as CSV
  const handleExportCsv = () => {
    const headers = ['Row_Index', 'Row_Name', ...colNames];
    const rows = currentMatrix.map((row, rIdx) => [
      (rIdx + 1).toString(),
      rowNames[rIdx] || `r${rIdx + 1}`,
      ...row.map((val) => val.toString()),
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `vortex_matrix_${selectedMatrixType.toLowerCase()}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export Canvas as PNG
  const handleExportPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `vortex_spy_plot_${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 font-mono text-xs">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-graphite-border gap-4">
        <div>
          <div className="text-xs text-electric tracking-editorial uppercase flex items-center gap-2">
            <span>// MODULE A — MATRIX INSPECTOR &amp; SPARSITY AUDIT</span>
            {activeResult && (
              <span className="px-2 py-0.5 rounded text-[10px] bg-electric/15 text-electric border border-electric/30">
                ACTIVE SOLVE: {activeResult.solver_type} ({activeResult.status})
              </span>
            )}
          </div>
          <h1 className="text-2xl sm:text-3xl font-display font-bold text-chrome mt-0.5">
            Sparsity &amp; Compressed Matrix Analyzer
          </h1>
          <p className="text-violet-muted text-xs sm:text-sm mt-0.5 font-sans">
            Inspect real coefficient matrices from active optimization models. Canvas-based zoomable spy plots and canonical CSR/CSC representations.
          </p>
        </div>

        {/* Matrix Type Selector: A vs Q */}
        <div className="flex items-center space-x-2">
          {formulation?.Q && formulation.Q.length > 0 && (
            <div className="flex bg-obsidian border border-graphite-border rounded-sm p-0.5">
              <button
                onClick={() => setSelectedMatrixType('A')}
                className={`px-3 py-1 text-xs rounded-sm transition-colors ${
                  selectedMatrixType === 'A'
                    ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                    : 'text-violet-muted hover:text-chrome'
                }`}
              >
                Constraint Matrix A
              </button>
              <button
                onClick={() => setSelectedMatrixType('Q')}
                className={`px-3 py-1 text-xs rounded-sm transition-colors ${
                  selectedMatrixType === 'Q'
                    ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                    : 'text-violet-muted hover:text-chrome'
                }`}
              >
                Hessian Matrix Q
              </button>
            </div>
          )}

          {/* View Mode Toggle: Spy Plot vs Dense Table vs Compressed */}
          <div className="flex bg-obsidian border border-graphite-border rounded-sm p-0.5">
            <button
              onClick={() => setViewMode('SPY')}
              className={`px-3 py-1 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                viewMode === 'SPY'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Grid className="w-3.5 h-3.5 text-electric" />
              <span>Spy Plot</span>
            </button>
            <button
              onClick={() => setViewMode('DENSE')}
              className={`px-3 py-1 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                viewMode === 'DENSE'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Table className="w-3.5 h-3.5 text-chrome" />
              <span>Dense Grid</span>
            </button>
            <button
              onClick={() => setViewMode('COMPRESSED')}
              className={`px-3 py-1 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                viewMode === 'COMPRESSED'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-chrome" />
              <span>CSR / CSC</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Telemetry Metrics Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        
        <div className="p-3.5 bg-obsidian-surface border border-graphite-border rounded-sm">
          <span className="text-[10px] text-violet-muted block uppercase">Dimension</span>
          <span className="font-semibold text-chrome text-sm mt-0.5 block">
            {m} × {n}
          </span>
          <span className="text-[10px] text-violet-dim">Total: {totalElements} elements</span>
        </div>

        <div className="p-3.5 bg-obsidian-surface border border-graphite-border rounded-sm">
          <span className="text-[10px] text-violet-muted block uppercase">Nonzero Count (nnz)</span>
          <span className="font-semibold text-electric text-sm mt-0.5 block">
            {nnz} nonzeros
          </span>
          <span className="text-[10px] text-violet-dim">Zero tolerance: 10⁻¹²</span>
        </div>

        <div className="p-3.5 bg-obsidian-surface border border-graphite-border rounded-sm">
          <span className="text-[10px] text-violet-muted block uppercase">Density</span>
          <span className="font-semibold text-chrome text-sm mt-0.5 block">
            {(density * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-violet-dim">Sparsity: {(sparsity * 100).toFixed(1)}%</span>
        </div>

        <div className="p-3.5 bg-obsidian-surface border border-graphite-border rounded-sm">
          <span className="text-[10px] text-violet-muted block uppercase">Memory Footprint</span>
          <span className="font-semibold text-chrome text-sm mt-0.5 block">
            {(nnz * 16 / 1024).toFixed(2)} KB (CSR)
          </span>
          <span className="text-[10px] text-violet-dim">Dense: {(totalElements * 8 / 1024).toFixed(2)} KB</span>
        </div>

      </div>

      {/* Main View Area */}
      {viewMode === 'SPY' && (
        <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3">
          
          {/* Controls Bar */}
          <div className="flex flex-wrap items-center justify-between border-b border-graphite-border pb-2 gap-2">
            <div className="flex items-center space-x-2">
              <span className="text-chrome font-semibold">{matrixLabel}</span>
              <span className="text-[10px] text-violet-muted px-2 py-0.5 bg-obsidian border border-graphite-border rounded">
                Drag to Pan • Click +/- to Zoom
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1 bg-obsidian border border-graphite-border rounded-sm p-0.5">
                <button
                  onClick={() => setZoomLevel((z) => Math.min(z * 1.3, 10.0))}
                  title="Zoom In"
                  className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setZoomLevel((z) => Math.max(z / 1.3, 0.2))}
                  title="Zoom Out"
                  className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={handleResetView}
                  title="Reset View"
                  className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>

              <button
                onClick={handleExportPng}
                title="Export Spy Plot as PNG"
                className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm flex items-center gap-1 transition-colors"
              >
                <Download className="w-3 h-3 text-electric" />
                <span>PNG</span>
              </button>
              <button
                onClick={handleExportCsv}
                title="Export Matrix as CSV"
                className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm flex items-center gap-1 transition-colors"
              >
                <FileSpreadsheet className="w-3 h-3 text-chrome" />
                <span>CSV</span>
              </button>
            </div>
          </div>

          {/* Interactive Canvas Spy Plot */}
          <div className="relative overflow-hidden bg-obsidian rounded-sm border border-graphite-border flex justify-center items-center">
            <canvas
              ref={canvasRef}
              width={800}
              height={420}
              onMouseMove={handleCanvasMouseMove}
              onMouseDown={handleMouseDown}
              onMouseUp={handleMouseUp}
              onMouseLeave={() => {
                setIsDragging(false);
                setHoveredCell(null);
              }}
              className="cursor-crosshair w-full max-w-[800px] h-[420px]"
            />

            {/* Cell Tooltip Overlay */}
            {hoveredCell && (
              <div className="absolute bottom-3 left-3 bg-obsidian/95 border border-electric/60 px-3 py-1.5 rounded text-[11px] shadow-lg pointer-events-none flex items-center gap-3">
                <span className="text-violet-muted">
                  Row: <strong className="text-chrome">{hoveredCell.rowName} [{hoveredCell.row + 1}]</strong>
                </span>
                <span className="text-violet-muted">
                  Col: <strong className="text-chrome">{hoveredCell.colName} [{hoveredCell.col + 1}]</strong>
                </span>
                <span className="text-violet-muted">
                  Value: <strong className={hoveredCell.val !== 0 ? (hoveredCell.val > 0 ? 'text-electric' : 'text-amber-400') : 'text-violet-dim'}>
                    {hoveredCell.val.toFixed(4)}
                  </strong>
                </span>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="flex items-center justify-between text-[10px] text-violet-muted pt-1">
            <div className="flex items-center space-x-4">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-electric inline-block" />
                Positive Nonzero (+Aᵢⱼ)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" />
                Negative Nonzero (−Aᵢⱼ)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm border border-graphite-border inline-block" />
                Zero Element
              </span>
            </div>
            <span>Zoom: {(zoomLevel * 100).toFixed(0)}%</span>
          </div>

        </div>
      )}

      {/* Dense Value Inspector View */}
      {viewMode === 'DENSE' && (
        <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3">
          <div className="flex items-center justify-between border-b border-graphite-border pb-2">
            <span className="font-semibold text-chrome">{matrixLabel} — Dense Numerical Grid</span>
            <button
              onClick={handleExportCsv}
              className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm flex items-center gap-1 text-[11px]"
            >
              <FileSpreadsheet className="w-3 h-3 text-chrome" />
              <span>Export CSV</span>
            </button>
          </div>

          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-obsidian border-b border-graphite-border text-[10px] text-violet-muted uppercase">
                  <th className="py-1.5 px-3 w-16 text-center">Row \ Col</th>
                  {colNames.map((cName, j) => (
                    <th key={j} className="py-1.5 px-3 text-center min-w-[75px]">
                      {cName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-graphite-border">
                {currentMatrix.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-graphite/20">
                    <td className="py-1.5 px-3 text-center font-semibold text-violet-muted bg-obsidian/60">
                      {rowNames[rIdx] || `r${rIdx + 1}`}
                    </td>
                    {row.map((val, cIdx) => {
                      const isNonZero = Math.abs(val) > 1e-12;
                      return (
                        <td
                          key={cIdx}
                          className={`py-1.5 px-3 text-center font-mono ${
                            isNonZero
                              ? val > 0
                                ? 'text-chrome font-semibold bg-electric/5'
                                : 'text-amber-300 font-semibold bg-amber-950/20'
                              : 'text-violet-dim/50'
                          }`}
                        >
                          {val.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Canonical CSR / CSC Representation View */}
      {viewMode === 'COMPRESSED' && (
        <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4">
          
          <div className="flex items-center justify-between border-b border-graphite-border pb-3">
            <div>
              <span className="font-semibold text-chrome text-xs uppercase tracking-wider block">
                Compressed Sparse Storage Vectors ({compressedFormat})
              </span>
              <span className="text-[10px] text-violet-muted">
                Zero-overhead contiguous array representation for sparse matrix-vector multiplication (SpMV).
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[10px] bg-electric/10 text-electric border border-electric/30">
                // Canonical Derived Sparse Storage
              </span>
              <div className="flex bg-obsidian border border-graphite-border rounded-sm p-0.5">
                <button
                  onClick={() => setCompressedFormat('CSR')}
                  className={`px-2.5 py-1 text-xs rounded-sm transition-colors ${
                    compressedFormat === 'CSR'
                      ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                      : 'text-violet-muted hover:text-chrome'
                  }`}
                >
                  CSR Format
                </button>
                <button
                  onClick={() => setCompressedFormat('CSC')}
                  className={`px-2.5 py-1 text-xs rounded-sm transition-colors ${
                    compressedFormat === 'CSC'
                      ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                      : 'text-violet-muted hover:text-chrome'
                  }`}
                >
                  CSC Format
                </button>
              </div>
            </div>
          </div>

          {compressedFormat === 'CSR' ? (
            <div className="space-y-3 font-mono text-xs">
              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  1. Non-Zero Values Array (`values` [FP64], length = {csr.values.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csr.values.map((v) => v.toFixed(3)).join(', ')}]
                </div>
              </div>

              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  2. Column Indices Array (`col_indices` [int32], length = {csr.colIndices.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csr.colIndices.join(', ')}]
                </div>
              </div>

              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  3. Row Pointers Array (`row_offsets` [int32], length = {csr.rowPointers.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csr.rowPointers.join(', ')}]
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3 font-mono text-xs">
              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  1. Non-Zero Values Array (`values` [FP64], length = {csc.values.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csc.values.map((v) => v.toFixed(3)).join(', ')}]
                </div>
              </div>

              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  2. Row Indices Array (`row_indices` [int32], length = {csc.rowIndices.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csc.rowIndices.join(', ')}]
                </div>
              </div>

              <div>
                <span className="text-[10px] text-violet-muted block uppercase mb-1">
                  3. Column Pointers Array (`col_offsets` [int32], length = {csc.colPointers.length})
                </span>
                <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome overflow-x-auto max-h-24">
                  [{csc.colPointers.join(', ')}]
                </div>
              </div>
            </div>
          )}

          <div className="p-3 bg-obsidian/40 border border-graphite-border/70 rounded-sm text-[11px] text-violet-muted">
            <span className="text-chrome font-semibold">// Linear Algebra Storage Note: </span>
            VØRTEX CUDA backend invokes cuSPARSE `spmv` kernel with CSR format for sparse GPU matrix-vector products.
          </div>

        </div>
      )}

    </div>
  );
};
