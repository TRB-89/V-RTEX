import React, { useState } from 'react';
import { 
  Check, 
  Copy, 
  Download, 
  HelpCircle, 
  CheckCircle2, 
  XCircle, 
  AlertCircle,
  Clock,
  Cpu,
  Layers,
  Activity,
  FileSpreadsheet
} from 'lucide-react';
import type { SolverResult } from '../types/solver';
import type { VariableDef, ConstraintDef } from './MathematicalPreview';

interface ResultsInspectorProps {
  result: SolverResult | null;
  variables: VariableDef[];
  constraints: ConstraintDef[];
  isSolving: boolean;
  onInspectMatrix?: () => void;
  onViewConvergence?: () => void;
}

export const ResultsInspector: React.FC<ResultsInspectorProps> = ({
  result,
  variables,
  constraints,
  isSolving,
  onInspectMatrix,
  onViewConvergence,
}) => {
  const [copied, setCopied] = useState<boolean>(false);

  // Copy full result JSON to clipboard
  const handleCopyJson = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Download result JSON
  const handleDownloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vortex_solution_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export CSV of solution variables & constraint slacks
  const handleDownloadCsv = () => {
    if (!result || !result.variable_values) return;
    const rows = [
      ['Index', 'Variable_Name', 'Type', 'Value', 'Lower_Bound', 'Upper_Bound'],
      ...result.variable_values.map((val, idx) => {
        const v = variables[idx];
        return [
          (idx + 1).toString(),
          v?.name ?? `x${idx + 1}`,
          v?.type ?? 'CONTINUOUS',
          val.toString(),
          v?.lower.toString() ?? '0',
          (v?.upper ?? 1e20) > 1e19 ? 'inf' : (v?.upper?.toString() ?? 'inf')
        ];
      })
    ];
    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `vortex_solution_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Compute constraint feasibility checks: A_i * x vs b_i
  const constraintChecks = React.useMemo(() => {
    if (!result || !result.variable_values || result.variable_values.length === 0) return [];
    const x = result.variable_values;
    return constraints.map((c) => {
      let lhs = 0;
      for (let j = 0; j < x.length && j < c.coefficients.length; j++) {
        lhs += c.coefficients[j] * x[j];
      }
      const tol = 1e-5;
      let isFeasible = true;
      let slack = 0;

      if (c.sense === 'LEQ') {
        slack = c.rhs - lhs;
        isFeasible = lhs <= c.rhs + tol;
      } else if (c.sense === 'GEQ') {
        slack = lhs - c.rhs;
        isFeasible = lhs >= c.rhs - tol;
      } else {
        slack = Math.abs(lhs - c.rhs);
        isFeasible = Math.abs(lhs - c.rhs) <= tol;
      }

      return {
        id: c.id,
        name: c.name,
        sense: c.sense,
        rhs: c.rhs,
        computedLhs: lhs,
        slack,
        isFeasible,
      };
    });
  }, [result, constraints]);

  // Compute integrality checks for integer/binary variables
  const integralityChecks = React.useMemo(() => {
    if (!result || !result.variable_values) return [];
    const x = result.variable_values;
    return variables
      .map((v, idx) => {
        if (v.type === 'CONTINUOUS') return null;
        const val = x[idx] ?? 0;
        const nearestInt = Math.round(val);
        const residual = Math.abs(val - nearestInt);
        const isSatisfied = residual <= 1e-5;
        return {
          id: v.id,
          name: v.name,
          type: v.type,
          value: val,
          nearestInt,
          residual,
          isSatisfied,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null);
  }, [result, variables]);

  return (
    <div className="p-5 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4 font-mono text-xs">
      
      {/* Console Header */}
      <div className="flex items-center justify-between border-b border-graphite-border pb-3">
        <div className="font-semibold text-chrome flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${isSolving ? 'bg-amber-400 animate-ping' : result ? 'bg-electric' : 'bg-violet-muted'}`} />
          <span className="tracking-wider uppercase text-[11px]">SOLVER RESULTS &amp; TELEMETRY</span>
        </div>

        {result && (
          <div className="flex items-center space-x-1">
            <button
              onClick={handleCopyJson}
              title="Copy JSON Result"
              className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleDownloadCsv}
              title="Export CSV Solution"
              className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-chrome" />
            </button>
            <button
              onClick={handleDownloadJson}
              title="Download JSON Solution"
              className="p-1 hover:bg-graphite text-violet-muted hover:text-chrome rounded-sm transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-chrome" />
            </button>
          </div>
        )}
      </div>

      {isSolving ? (
        <div className="py-20 text-center space-y-3">
          <div className="w-8 h-8 border-2 border-electric border-t-transparent rounded-full animate-spin mx-auto" />
          <div className="text-chrome font-semibold text-xs tracking-wider">
            NATIVE ENGINE EXECUTING...
          </div>
          <p className="text-[11px] text-violet-muted max-w-xs mx-auto">
            Dispatching problem formulation to native C++20/CUDA binary.
          </p>
        </div>
      ) : result ? (
        <div className="space-y-4">
          
          {/* Status & Objective Banner */}
          <div className="p-4 bg-obsidian border border-graphite-border rounded-sm flex items-center justify-between">
            <div>
              <div className="text-[10px] text-violet-muted uppercase tracking-wider">
                Optimal Objective Z*
              </div>
              <div className="text-2xl font-bold text-chrome mt-0.5 tracking-tight">
                {result.objective_value !== undefined && result.objective_value !== null
                  ? Number(result.objective_value).toFixed(6)
                  : <span className="text-violet-muted text-lg italic">Not reported</span>}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-violet-muted uppercase tracking-wider">
                Termination Status
              </div>
              <div className={`text-xs font-semibold px-2.5 py-1 rounded-sm mt-0.5 inline-block ${
                result.status === 'OPTIMAL'
                  ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800'
                  : result.status === 'INFEASIBLE'
                  ? 'bg-rose-950/60 text-rose-300 border border-rose-800'
                  : result.status === 'UNBOUNDED'
                  ? 'bg-amber-950/60 text-amber-300 border border-amber-800'
                  : result.status === 'CLIENT_ABORTED'
                  ? 'bg-amber-950/60 text-amber-200 border border-amber-800'
                  : 'bg-red-950/60 text-red-300 border border-red-800'
              }`}>
                {result.status}
              </div>
            </div>
          </div>

          {/* Computational Metrics Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            
            {/* Wall time */}
            <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm flex items-start gap-2">
              <Clock className="w-3.5 h-3.5 text-violet-muted mt-0.5 shrink-0" />
              <div>
                <span className="text-[10px] text-violet-muted block uppercase">Solve Runtime</span>
                <span className="font-semibold text-chrome">
                  {result.wall_time_ms !== undefined && result.wall_time_ms !== null
                    ? `${result.wall_time_ms.toFixed(3)} ms`
                    : 'Not reported'}
                </span>
              </div>
            </div>

            {/* Iterations / Nodes */}
            <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm flex items-start gap-2">
              <Layers className="w-3.5 h-3.5 text-violet-muted mt-0.5 shrink-0" />
              <div>
                <span className="text-[10px] text-violet-muted block uppercase">
                  {result.nodes_explored !== undefined ? 'B&B Nodes / Iter' : 'Iterations'}
                </span>
                <span className="font-semibold text-chrome">
                  {result.iterations !== undefined
                    ? `${result.iterations} iters`
                    : result.nodes_explored !== undefined
                    ? `${result.nodes_explored} nodes`
                    : 'Not reported'}
                </span>
              </div>
            </div>

            {/* Executed Backend */}
            <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm flex items-start gap-2">
              <Cpu className="w-3.5 h-3.5 text-electric mt-0.5 shrink-0" />
              <div>
                <span className="text-[10px] text-violet-muted block uppercase">Executed Backend</span>
                <span className="font-semibold text-electric">
                  {result.device_diagnostics?.executed_backend ?? 'CPU'}
                </span>
              </div>
            </div>

            {/* Residuals / Convergence */}
            <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm flex items-start gap-2">
              <Activity className="w-3.5 h-3.5 text-violet-muted mt-0.5 shrink-0" />
              <div>
                <span className="text-[10px] text-violet-muted block uppercase">Residual Diagnostic</span>
                <span className="font-semibold text-chrome">
                  {result.max_primal_residual !== undefined
                    ? `Primal: ${result.max_primal_residual.toExponential(2)}`
                    : result.primal_residual !== undefined
                    ? `Primal: ${result.primal_residual.toExponential(2)}`
                    : 'Not reported'}
                </span>
              </div>
            </div>

          </div>

          {/* Backend Diagnostics & GPU Fallback Reason */}
          {result.device_diagnostics?.fallback_reason && (
            <div className="p-2.5 bg-amber-950/30 border border-amber-900/60 rounded-sm text-[11px] text-amber-200">
              <span className="font-semibold text-amber-400 block mb-0.5 flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" />
                Backend Fallback Reason:
              </span>
              {result.device_diagnostics.fallback_reason}
            </div>
          )}

          {/* Decision Variables Table */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-chrome uppercase tracking-wider">
                Primal Solution Vector x* ({result.variable_values?.length ?? 0} variables)
              </span>
            </div>
            <div className="bg-obsidian border border-graphite-border rounded-sm divide-y divide-graphite-border max-h-48 overflow-y-auto">
              {result.variable_values && result.variable_values.length > 0 ? (
                result.variable_values.map((val, idx) => {
                  const varDef = variables[idx];
                  return (
                    <div key={idx} className="px-3 py-1.5 flex items-center justify-between text-xs hover:bg-graphite/20">
                      <div className="flex items-center gap-2">
                        <span className="text-violet-muted font-semibold">
                          {varDef?.name ?? `x${idx + 1}`}
                        </span>
                        {varDef && (
                          <span className="text-[10px] px-1 py-0.2 bg-graphite rounded text-violet-muted">
                            {varDef.type}
                          </span>
                        )}
                      </div>
                      <span className="text-chrome font-bold">
                        {val.toFixed(6)}
                      </span>
                    </div>
                  );
                })
              ) : (
                <div className="p-3 text-center text-violet-muted text-[11px]">
                  No primal values returned for this status.
                </div>
              )}
            </div>
          </div>

          {/* MILP Integrality Verification Check */}
          {integralityChecks.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-chrome uppercase tracking-wider">
                  Integrality Feasibility Check ({integralityChecks.length} discrete vars)
                </span>
              </div>
              <div className="bg-obsidian border border-graphite-border rounded-sm divide-y divide-graphite-border max-h-36 overflow-y-auto">
                {integralityChecks.map((ic) => (
                  <div key={ic.id} className="px-3 py-1.5 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5">
                      {ic.isSatisfied ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      )}
                      <span className="text-chrome font-medium">{ic.name}</span>
                      <span className="text-[10px] text-violet-muted">({ic.type})</span>
                    </div>
                    <div className="text-right">
                      <span className="text-chrome font-semibold mr-2">{ic.value.toFixed(4)}</span>
                      <span className={`text-[10px] ${ic.isSatisfied ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {ic.isSatisfied ? 'Exact Integer' : `Violated (Δ ${ic.residual.toExponential(2)})`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Constraint Feasibility Verification Check */}
          {constraintChecks.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-chrome uppercase tracking-wider">
                  Constraint Feasibility Verification ({constraintChecks.length} rows)
                </span>
              </div>
              <div className="bg-obsidian border border-graphite-border rounded-sm divide-y divide-graphite-border max-h-40 overflow-y-auto">
                {constraintChecks.map((cc) => (
                  <div key={cc.id} className="px-3 py-1.5 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5">
                      {cc.isFeasible ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      )}
                      <span className="text-chrome font-medium">{cc.name}</span>
                      <span className="text-[10px] text-violet-muted">
                        ({cc.computedLhs.toFixed(2)} {cc.sense === 'LEQ' ? '≤' : cc.sense === 'GEQ' ? '≥' : '='} {cc.rhs})
                      </span>
                    </div>
                    <div className="text-right">
                      <span className={`text-[10px] font-semibold ${cc.isFeasible ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {cc.isFeasible ? `Feasible (Slack: ${cc.slack.toFixed(2)})` : `VIOLATED (Slack: ${cc.slack.toFixed(2)})`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dual Multipliers / Shadow Prices */}
          {(result.dual_values || result.lambda) && (
            <div className="space-y-1.5">
              <div className="text-[11px] font-semibold text-chrome uppercase tracking-wider">
                Dual Multipliers / Shadow Prices
              </div>
              <div className="bg-obsidian border border-graphite-border rounded-sm divide-y divide-graphite-border max-h-32 overflow-y-auto">
                {(result.dual_values || result.lambda || []).map((val, idx) => (
                  <div key={idx} className="px-3 py-1 flex items-center justify-between text-xs">
                    <span className="text-violet-muted font-medium">λ[{idx + 1}]</span>
                    <span className="text-chrome font-semibold">{val.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quick Inspection Buttons */}
          <div className="flex items-center space-x-2 pt-2 border-t border-graphite-border">
            {onInspectMatrix && (
              <button
                onClick={onInspectMatrix}
                className="flex-1 py-1.5 bg-graphite hover:bg-graphite-light text-chrome text-xs rounded-sm transition-colors border border-graphite-border"
              >
                Inspect Matrix Structure
              </button>
            )}
            {onViewConvergence && (
              <button
                onClick={onViewConvergence}
                className="flex-1 py-1.5 bg-graphite hover:bg-graphite-light text-chrome text-xs rounded-sm transition-colors border border-graphite-border"
              >
                Convergence Trace
              </button>
            )}
          </div>

        </div>
      ) : (
        <div className="py-16 text-center space-y-2">
          <HelpCircle className="w-8 h-8 text-violet-dim mx-auto" />
          <div className="text-xs text-violet-muted">
            No active solution loaded in memory.
          </div>
          <p className="text-[11px] text-violet-dim max-w-xs mx-auto">
            Configure decision variables and constraints, then click 'RUN SOLVER' to invoke the native C++ engine.
          </p>
        </div>
      )}

    </div>
  );
};
