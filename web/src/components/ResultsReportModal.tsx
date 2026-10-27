import React from 'react';
import { X, Download, Copy, Check, FileText } from 'lucide-react';
import type { SolverResult } from '../types/solver';

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
    let csvContent = 'data:text/csv;charset=utf-8,Variable,Value\n';
    if (result.variable_values) {
      result.variable_values.forEach((v, i) => {
        csvContent += `x[${i + 1}],${v}\n`;
      });
    }
    if (result.dual_values) {
      result.dual_values.forEach((v, i) => {
        csvContent += `lambda[${i + 1}],${v}\n`;
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-obsidian/80 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-obsidian-surface border border-graphite-border rounded-sm shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-graphite-border flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono text-electric uppercase tracking-wider">
              // SOLVER EXECUTION AUDIT &amp; REPORT
            </div>
            <h2 className="text-lg font-display font-bold text-chrome">
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
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted block uppercase">Status</span>
              <span className="text-emerald-400 font-bold text-sm mt-0.5 block">{result.status}</span>
              <span className="text-[10px] text-violet-dim">{result.message}</span>
            </div>

            <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted block uppercase">Objective Value Z*</span>
              <span className="text-chrome font-bold text-sm mt-0.5 block">
                {result.objective_value !== undefined ? Number(result.objective_value).toFixed(6) : 'N/A'}
              </span>
              <span className="text-[10px] text-violet-dim">Wall Time: {result.wall_time_ms ? `${result.wall_time_ms.toFixed(2)} ms` : '< 1 ms'}</span>
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
            <pre className="bg-obsidian border border-graphite-border rounded-sm p-3 text-[10px] text-violet-muted overflow-x-auto max-h-48">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3 border-t border-graphite-border bg-obsidian flex items-center justify-between font-mono text-xs">
          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadJson}
              className="px-3 py-1.5 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm transition-colors flex items-center gap-1.5"
            >
              <FileText className="w-3.5 h-3.5 text-electric" />
              Download JSON
            </button>
            <button
              onClick={handleDownloadCsv}
              className="px-3 py-1.5 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm transition-colors flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5 text-electric" />
              Download CSV
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
