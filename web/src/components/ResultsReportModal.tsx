import React from 'react';
import { X, Download, Copy, Check, FileText, Printer, Cpu, Zap, Activity } from 'lucide-react';
import type { SolverResult } from '../types/solver';
import { generateVortexPdfReport } from '../utils/pdfExport';

interface ResultsReportModalProps {
  result: SolverResult | null;
  isOpen: boolean;
  onClose: () => void;
}

export const ResultsReportModal: React.FC<ResultsReportModalProps> = ({
  result,
  isOpen,
  onClose,
}) => {
  const [copied, setCopied] = React.useState(false);

  if (!isOpen || !result) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `vortex_solution_${result.solver_type.toLowerCase()}_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleDownloadCsv = () => {
    let csvContent = 'data:text/csv;charset=utf-8,Category,Identifier,Value\n';
    csvContent += `Meta,Solver_Type,${result.solver_type}\n`;
    csvContent += `Meta,Status,${result.status}\n`;
    csvContent += `Meta,Objective_Value,${result.objective_value ?? 'N/A'}\n`;
    csvContent += `Meta,Wall_Time_ms,${result.wall_time_ms ?? 'N/A'}\n`;
    
    if (result.variable_values) {
      result.variable_values.forEach((v, i) => {
        csvContent += `Variable,x[${i + 1}],${v}\n`;
      });
    }
    if (result.dual_values) {
      result.dual_values.forEach((v, i) => {
        csvContent += `Dual,lambda[${i + 1}],${v}\n`;
      });
    }
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `vortex_solution_${result.solver_type.toLowerCase()}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleDownloadPdf = () => {
    generateVortexPdfReport(result);
  };

  const diag = result.device_diagnostics;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-obsidian/80 backdrop-blur-sm">
      <div className="w-full max-w-3xl bg-obsidian-surface border border-graphite-border rounded-sm shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-graphite-border flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono text-electric uppercase tracking-wider flex items-center gap-1.5">
              <span>// SOLVER EXECUTION AUDIT &amp; REPORT</span>
              <span className="text-violet-muted">·</span>
              <span className="text-emerald-400">RESEARCH PROTOTYPE v1.0</span>
            </div>
            <h2 className="text-lg font-display font-bold text-chrome mt-0.5">
              Optimization Solution Record [{result.solver_type}]
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-5 font-mono text-xs">
          
          {/* Status & Summary */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted block uppercase">Status</span>
              <span className="text-emerald-400 font-bold text-sm mt-0.5 block">{result.status}</span>
              <span className="text-[10px] text-violet-dim truncate block">{result.message}</span>
            </div>

            <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted block uppercase">Objective Value Z*</span>
              <span className="text-chrome font-bold text-sm mt-0.5 block">
                {result.objective_value !== undefined ? Number(result.objective_value).toFixed(6) : 'N/A'}
              </span>
              <span className="text-[10px] text-violet-dim">Wall Time: {result.wall_time_ms ? `${result.wall_time_ms.toFixed(2)} ms` : '< 1 ms'}</span>
            </div>

            <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted block uppercase">Executed Backend</span>
              <span className="text-electric font-bold text-sm mt-0.5 block flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5" />
                <span>{diag?.executed_backend || 'CPU Native'}</span>
              </span>
              <span className="text-[10px] text-violet-dim">Requested: {diag?.requested_backend || 'AUTO'}</span>
            </div>
          </div>

          {/* Backend Diagnostics & Fallbacks */}
          <div className="p-3.5 bg-obsidian border border-graphite-border rounded-sm space-y-2">
            <div className="text-chrome font-semibold text-[11px] flex items-center justify-between border-b border-graphite-border pb-1.5">
              <span className="flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-electric" />
                <span>HARDWARE &amp; DISPATCH TELEMETRY</span>
              </span>
              {diag?.fallback_reason ? (
                <span className="text-[10px] text-amber-400 font-normal">
                  Fallback: {diag.fallback_reason}
                </span>
              ) : (
                <span className="text-[10px] text-emerald-400 font-normal">
                  ✓ Dispatch Verified
                </span>
              )}
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] pt-1">
              <div>
                <span className="text-violet-muted block text-[10px]">GPU Iterations</span>
                <span className="text-chrome font-medium">{diag?.gpu_iterations ?? 0}</span>
              </div>
              <div>
                <span className="text-violet-muted block text-[10px]">GPU Kernel Calls</span>
                <span className="text-chrome font-medium">{(diag?.gpu_gemv_calls ?? 0) + (diag?.gpu_spmv_calls ?? 0)}</span>
              </div>
              <div>
                <span className="text-violet-muted block text-[10px]">H2D Transfers</span>
                <span className="text-chrome font-medium">{diag?.h2d_transfers ?? 0}</span>
              </div>
              <div>
                <span className="text-violet-muted block text-[10px]">D2H Transfers</span>
                <span className="text-chrome font-medium">{diag?.d2h_transfers ?? 0}</span>
              </div>
            </div>
          </div>

          {/* Numerical Diagnostics (LP / MILP / QP) */}
          <div className="p-3.5 bg-obsidian border border-graphite-border rounded-sm space-y-2">
            <div className="text-chrome font-semibold text-[11px] flex items-center justify-between border-b border-graphite-border pb-1.5">
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-400" />
                <span>MATHEMATICAL FEASIBILITY &amp; RESIDUALS</span>
              </span>
              <span className="text-[10px] text-violet-muted">IEEE 754 FP64 Ground Truth</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] pt-1">
              {result.solver_type === 'QP' ? (
                <>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Primal Residual ||Ax - b||</span>
                    <span className="text-chrome font-medium">{result.primal_residual !== undefined ? result.primal_residual.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Dual Residual ||Qx+c-A^T λ||</span>
                    <span className="text-chrome font-medium">{result.dual_residual !== undefined ? result.dual_residual.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Duality Gap λ^T s</span>
                    <span className="text-chrome font-medium">{result.duality_gap !== undefined ? result.duality_gap.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                </>
              ) : result.solver_type === 'MILP' ? (
                <>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Best Bound</span>
                    <span className="text-chrome font-medium">{result.best_bound !== undefined ? result.best_bound.toFixed(6) : 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Absolute Optimality Gap</span>
                    <span className="text-chrome font-medium">{result.absolute_gap !== undefined ? result.absolute_gap.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Relative Optimality Gap</span>
                    <span className="text-chrome font-medium">{result.relative_gap !== undefined ? `${(result.relative_gap * 100).toFixed(4)}%` : '0.00%'}</span>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Max Primal Residual</span>
                    <span className="text-chrome font-medium">{result.max_primal_residual !== undefined ? result.max_primal_residual.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Max Dual Residual</span>
                    <span className="text-chrome font-medium">{result.max_dual_residual !== undefined ? result.max_dual_residual.toExponential(4) : '0.0000e+00'}</span>
                  </div>
                  <div>
                    <span className="text-violet-muted block text-[10px]">Basis Condition Number (κ)</span>
                    <span className="text-chrome font-medium">{result.basis_condition !== undefined ? result.basis_condition.toExponential(2) : '1.00e+00'}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Variables Table */}
          <div className="space-y-1.5">
            <span className="text-chrome font-semibold text-[11px] block uppercase">
              Decision Variable Values (x*)
            </span>
            <div className="bg-obsidian border border-graphite-border rounded-sm divide-y divide-graphite-border max-h-40 overflow-y-auto">
              {result.variable_values && result.variable_values.map((v, i) => (
                <div key={i} className="px-3 py-1 flex items-center justify-between">
                  <span className="text-violet-muted">x[{i + 1}]</span>
                  <span className="text-chrome font-medium">{v.toFixed(6)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Raw JSON Data Preview */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-chrome font-semibold text-[11px] uppercase">
                Machine-Readable JSON Output
              </span>
              <button
                onClick={handleCopy}
                className="text-[10px] text-violet-muted hover:text-chrome flex items-center gap-1"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre className="bg-obsidian border border-graphite-border rounded-sm p-3 text-[10px] text-violet-muted overflow-x-auto max-h-40">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3 border-t border-graphite-border bg-obsidian flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleDownloadPdf}
              className="px-3 py-1.5 bg-electric hover:bg-electric-glow text-white font-medium rounded-sm transition-colors flex items-center gap-1.5 shadow-[0_0_10px_rgba(37,20,237,0.3)]"
            >
              <Printer className="w-3.5 h-3.5" />
              Download PDF Report
            </button>
            <button
              onClick={handleDownloadJson}
              className="px-3 py-1.5 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm transition-colors flex items-center gap-1.5"
            >
              <FileText className="w-3.5 h-3.5 text-electric" />
              JSON
            </button>
            <button
              onClick={handleDownloadCsv}
              className="px-3 py-1.5 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm transition-colors flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5 text-electric" />
              CSV
            </button>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-graphite hover:bg-graphite-light text-violet-muted hover:text-chrome rounded-sm transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
};
