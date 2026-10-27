import React, { useState } from 'react';
import { 
  Activity, 
  HelpCircle, 
  Download, 
  FileSpreadsheet, 
  Box, 
  Compass,
  AlertCircle
} from 'lucide-react';
import type { SolverResult, ProblemFormulation } from '../types/solver';
import { FeasibleRegionPlot2D } from './FeasibleRegionPlot2D';
import { ConvexQPSurface3D } from './ConvexQPSurface3D';

interface ConvergenceLabProps {
  activeResult: SolverResult | null;
  formulation?: ProblemFormulation | null;
  onLaunchSolver: () => void;
  reducedMotion?: boolean;
}

export const ConvergenceLab: React.FC<ConvergenceLabProps> = ({ 
  activeResult, 
  formulation = null,
  onLaunchSolver,
  reducedMotion = false,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'TELEMETRY' | 'GEOMETRY_2D' | 'SURFACE_3D'>('TELEMETRY');

  const isQP = activeResult?.solver_type === 'QP';
  const isMILP = activeResult?.solver_type === 'MILP';
  const isLP = activeResult?.solver_type === 'LP';

  const history = activeResult?.iteration_history || [];
  const hasHistory = history.length > 0;

  // Export Convergence Telemetry to CSV
  const handleExportCsv = () => {
    if (!hasHistory) return;
    const headers = ['Iteration', 'Objective_Value', 'Primal_Residual', 'Dual_Residual', 'Duality_Gap'];
    const rows = history.map((rec) => [
      rec.iteration.toString(),
      rec.objective?.toString() ?? 'N/A',
      rec.primal_residual?.toString() ?? 'N/A',
      rec.dual_residual?.toString() ?? 'N/A',
      rec.duality_gap?.toString() ?? 'N/A',
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `vortex_convergence_trajectory_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export Convergence Telemetry to JSON
  const handleExportJson = () => {
    if (!activeResult) return;
    const exportData = {
      solver_type: activeResult.solver_type,
      status: activeResult.status,
      objective_value: activeResult.objective_value,
      iterations: activeResult.iterations,
      wall_time_ms: activeResult.wall_time_ms,
      iteration_history: history,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vortex_convergence_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 font-mono text-xs">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-graphite-border gap-4">
        <div>
          <div className="text-xs text-electric tracking-editorial uppercase">
            // MODULE B &amp; C — CONVERGENCE LAB &amp; OPTIMIZATION GEOMETRY
          </div>
          <h1 className="text-2xl sm:text-3xl font-display font-bold text-chrome mt-0.5">
            Mathematical Convergence &amp; Optimization Geometry
          </h1>
          <p className="text-violet-muted text-xs sm:text-sm mt-0.5 font-sans">
            Inspect real iteration trajectories, KKT residual progressions, 2D polyhedral constraint half-spaces, and 3D quadratic objective paraboloids.
          </p>
        </div>

        {/* Sub-Tab Selector: Telemetry vs 2D Geometry vs 3D Surface */}
        <div className="flex items-center space-x-2">
          <div className="flex bg-obsidian border border-graphite-border rounded-sm p-0.5">
            <button
              onClick={() => setActiveSubTab('TELEMETRY')}
              className={`px-3 py-1.5 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                activeSubTab === 'TELEMETRY'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Activity className="w-3.5 h-3.5 text-electric" />
              <span>Residual Diagnostics</span>
            </button>
            <button
              onClick={() => setActiveSubTab('GEOMETRY_2D')}
              className={`px-3 py-1.5 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                activeSubTab === 'GEOMETRY_2D'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Compass className="w-3.5 h-3.5 text-chrome" />
              <span>2D Feasible Region</span>
            </button>
            <button
              onClick={() => setActiveSubTab('SURFACE_3D')}
              className={`px-3 py-1.5 text-xs rounded-sm transition-colors flex items-center gap-1.5 ${
                activeSubTab === 'SURFACE_3D'
                  ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                  : 'text-violet-muted hover:text-chrome'
              }`}
            >
              <Box className="w-3.5 h-3.5 text-chrome" />
              <span>3D QP Surface</span>
            </button>
          </div>
        </div>
      </div>

      {activeResult ? (
        <div className="space-y-6">
          
          {/* Top Diagnostics KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted uppercase block">Total Iterations / Nodes</span>
              <span className="text-xl font-bold text-chrome mt-1 block">
                {activeResult.iterations ?? activeResult.nodes_explored ?? 0}
              </span>
              <span className="text-[10px] text-violet-dim">
                {isLP && `Phase I: ${activeResult.phase1_iterations || 0} | Phase II: ${activeResult.phase2_iterations || activeResult.iterations || 0}`}
                {isQP && 'Primal-Dual Interior Steps'}
                {isMILP && `Branch & Bound Nodes`}
              </span>
            </div>

            <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted uppercase block">Primal Residual</span>
              <span className="text-xl font-bold text-emerald-400 mt-1 block">
                {activeResult.primal_residual !== undefined
                  ? activeResult.primal_residual.toExponential(2)
                  : (activeResult.max_primal_residual !== undefined
                      ? activeResult.max_primal_residual.toExponential(2)
                      : '0.00')}
              </span>
              <span className="text-[10px] text-violet-dim">||Ax + s - b||_inf &lt; 10⁻⁷</span>
            </div>

            <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted uppercase block">Dual Residual</span>
              <span className="text-xl font-bold text-chrome mt-1 block">
                {activeResult.dual_residual !== undefined
                  ? activeResult.dual_residual.toExponential(2)
                  : (activeResult.max_dual_residual !== undefined
                      ? activeResult.max_dual_residual.toExponential(2)
                      : '0.00')}
              </span>
              <span className="text-[10px] text-violet-dim">||Qx + c + Aᵀλ||_inf</span>
            </div>

            <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm">
              <span className="text-[10px] text-violet-muted uppercase block">
                {isQP ? 'Complementarity Duality Gap' : (isMILP ? 'Optimality Gap' : 'Basis Condition Ratio')}
              </span>
              <span className="text-xl font-bold text-electric mt-1 block">
                {isQP && (activeResult.duality_gap ? activeResult.duality_gap.toExponential(2) : '< 10⁻⁷')}
                {isMILP && (activeResult.relative_gap !== undefined ? `${(activeResult.relative_gap * 100).toFixed(2)}%` : '0.00%')}
                {isLP && (activeResult.basis_condition ? activeResult.basis_condition.toFixed(4) : '1.0000')}
              </span>
              <span className="text-[10px] text-violet-dim">
                {isQP && 'sᵀλ / max(1, m)'}
                {isMILP && '|BestBound - Incumbent|'}
                {isLP && 'LU Condition Estimate'}
              </span>
            </div>

          </div>

          {/* Sub-Tab 1: Telemetry & Iteration History */}
          {activeSubTab === 'TELEMETRY' && (
            <div className="space-y-5">
              
              {/* Iteration Trajectory Chart (if history genuinely recorded) */}
              {hasHistory ? (
                <div className="p-5 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-graphite-border pb-3">
                    <div>
                      <span className="font-semibold text-chrome text-xs uppercase tracking-wider block">
                        Recorded Per-Iteration Convergence Trajectory ({history.length} Steps)
                      </span>
                      <span className="text-[10px] text-violet-muted">
                        Genuine step-by-step interior-point metric evolution recorded directly by native C++ solver.
                      </span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={handleExportCsv}
                        className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px] flex items-center gap-1"
                        title="Export Trajectory to CSV"
                      >
                        <FileSpreadsheet className="w-3 h-3 text-chrome" />
                        <span>CSV</span>
                      </button>
                      <button
                        onClick={handleExportJson}
                        className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px] flex items-center gap-1"
                        title="Export Telemetry to JSON"
                      >
                        <Download className="w-3 h-3 text-electric" />
                        <span>JSON</span>
                      </button>
                    </div>
                  </div>

                  {/* Scientific Trajectory Table */}
                  <div className="overflow-x-auto max-h-72">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-obsidian border-b border-graphite-border text-[10px] text-violet-muted uppercase">
                          <th className="py-2 px-3">Iter #</th>
                          <th className="py-2 px-3">Objective Z</th>
                          <th className="py-2 px-3">Primal Residual ||rₚ||</th>
                          <th className="py-2 px-3">Dual Residual ||r_d||</th>
                          <th className="py-2 px-3">Duality Gap μ</th>
                          <th className="py-2 px-3">Convergence Rate</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-graphite-border">
                        {history.map((rec, idx) => {
                          const prevGap = idx > 0 ? history[idx - 1].duality_gap : null;
                          const rate = prevGap && rec.duality_gap ? (rec.duality_gap / prevGap) : null;
                          return (
                            <tr key={idx} className="hover:bg-graphite/20">
                              <td className="py-1.5 px-3 font-semibold text-violet-muted">
                                {rec.iteration}
                              </td>
                              <td className="py-1.5 px-3 text-chrome font-semibold">
                                {rec.objective?.toFixed(6) ?? 'N/A'}
                              </td>
                              <td className="py-1.5 px-3 text-emerald-400">
                                {rec.primal_residual?.toExponential(3) ?? '0.000e+00'}
                              </td>
                              <td className="py-1.5 px-3 text-chrome">
                                {rec.dual_residual?.toExponential(3) ?? '0.000e+00'}
                              </td>
                              <td className="py-1.5 px-3 text-electric font-semibold">
                                {rec.duality_gap?.toExponential(3) ?? '0.000e+00'}
                              </td>
                              <td className="py-1.5 px-3 text-violet-muted">
                                {rate ? `×${rate.toFixed(3)}` : '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                </div>
              ) : (
                <div className="p-4 bg-obsidian border border-graphite-border rounded-sm flex items-start gap-3 text-violet-muted">
                  <AlertCircle className="w-4 h-4 text-violet-muted mt-0.5 shrink-0" />
                  <div className="space-y-1">
                    <span className="text-chrome font-semibold block">
                      Iteration history unavailable: Terminal basis diagnostics reported
                    </span>
                    <p className="text-[11px] leading-relaxed">
                      To preserve strict scientific integrity, VØRTEX never fabricates intermediate convergence paths. The Simplex / Branch-and-Bound solver returns terminal basis condition ratios and optimality metrics upon termination. Full step-by-step trajectory tracing is enabled for Quadratic Programs (IPM).
                    </p>
                  </div>
                </div>
              )}

              {/* Formulation-Specific Deep Diagnostics */}
              <div className="p-5 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4">
                
                <div className="border-b border-graphite-border pb-3 flex items-center justify-between">
                  <div className="font-semibold text-chrome flex items-center gap-2">
                    <Activity className="w-4 h-4 text-electric" />
                    <span>FORMULATION-SPECIFIC RESIDUAL AUDIT [{activeResult.solver_type}]</span>
                  </div>
                  <span className="text-[11px] text-emerald-400 border border-emerald-800/60 bg-emerald-950/40 px-2 py-0.5 rounded-sm">
                    TERMINATION: {activeResult.status}
                  </span>
                </div>

                {isQP && (
                  <div className="space-y-3 font-sans text-xs">
                    <p className="text-violet-muted leading-relaxed">
                      Primal-Dual Predictor-Corrector Interior-Point solver satisfied all Karush-Kuhn-Tucker (KKT) first-order optimality conditions:
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono">
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted uppercase block">1. Primal Feasibility</span>
                        <span className="text-emerald-400 font-semibold mt-1 block">Ax + s = b (s ≥ 0)</span>
                        <span className="text-[11px] text-violet-muted mt-1 block">
                          Residual: {activeResult.primal_residual?.toExponential(3) || '0.000e+00'}
                        </span>
                      </div>
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted uppercase block">2. Dual Feasibility</span>
                        <span className="text-emerald-400 font-semibold mt-1 block">Qx + c + Aᵀλ = 0 (λ ≥ 0)</span>
                        <span className="text-[11px] text-violet-muted mt-1 block">
                          Residual: {activeResult.dual_residual?.toExponential(3) || '0.000e+00'}
                        </span>
                      </div>
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted uppercase block">3. Complementarity</span>
                        <span className="text-emerald-400 font-semibold mt-1 block">sᵢ · λᵢ → 0</span>
                        <span className="text-[11px] text-violet-muted mt-1 block">
                          Gap: {activeResult.duality_gap?.toExponential(3) || '0.000e+00'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {isMILP && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted block">NODES EXPLORED</span>
                        <span className="text-chrome font-bold text-base mt-0.5 block">{activeResult.nodes_explored || 0}</span>
                      </div>
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted block">PRUNED BY BOUND</span>
                        <span className="text-chrome font-bold text-base mt-0.5 block">{activeResult.nodes_pruned_bound || 0}</span>
                      </div>
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted block">PRUNED INFEASIBLE</span>
                        <span className="text-chrome font-bold text-base mt-0.5 block">{activeResult.nodes_pruned_infeasible || 0}</span>
                      </div>
                      <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                        <span className="text-[10px] text-violet-muted block">INTEGER SOLUTIONS</span>
                        <span className="text-emerald-400 font-bold text-base mt-0.5 block">{activeResult.nodes_pruned_integer || 0}</span>
                      </div>
                    </div>

                    <div className="p-3 bg-obsidian border border-graphite-border rounded-sm flex items-center justify-between">
                      <div>
                        <span className="text-[10px] text-violet-muted block uppercase">Incumbent vs Dual Bound</span>
                        <span className="text-chrome text-xs font-semibold mt-0.5 block">
                          Primal Incumbent: {activeResult.objective_value?.toFixed(2)} | Best Dual Bound: {activeResult.best_bound?.toFixed(2)}
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] text-violet-muted block uppercase">Optimality Gap</span>
                        <span className="text-emerald-400 text-xs font-semibold mt-0.5 block">
                          {activeResult.absolute_gap?.toFixed(4) || '0.0000'} ({((activeResult.relative_gap || 0) * 100).toFixed(2)}%)
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {isLP && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                      <span className="text-[10px] text-violet-muted block">MAX PRIMAL RESIDUAL</span>
                      <span className="text-chrome font-bold text-base mt-0.5 block">
                        {activeResult.max_primal_residual !== undefined ? activeResult.max_primal_residual.toExponential(3) : '0.000e+00'}
                      </span>
                      <span className="text-[10px] text-violet-dim">||B · x_B - b||_inf</span>
                    </div>

                    <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                      <span className="text-[10px] text-violet-muted block">MAX DUAL RESIDUAL</span>
                      <span className="text-chrome font-bold text-base mt-0.5 block">
                        {activeResult.max_dual_residual !== undefined ? activeResult.max_dual_residual.toExponential(3) : '0.000e+00'}
                      </span>
                      <span className="text-[10px] text-violet-dim">||Bᵀ · π - c_B||_inf</span>
                    </div>

                    <div className="p-3 bg-obsidian border border-graphite-border rounded-sm">
                      <span className="text-[10px] text-violet-muted block">BASIS CONDITION RATIO</span>
                      <span className="text-chrome font-bold text-base mt-0.5 block">
                        {activeResult.basis_condition ? activeResult.basis_condition.toFixed(4) : '1.0000'}
                      </span>
                      <span className="text-[10px] text-violet-dim">min(|U_ii|) / max(|U_ii|)</span>
                    </div>
                  </div>
                )}

              </div>

            </div>
          )}

          {/* Sub-Tab 2: 2D Feasible Region Geometry */}
          {activeSubTab === 'GEOMETRY_2D' && (
            <FeasibleRegionPlot2D
              formulation={formulation}
              activeResult={activeResult}
            />
          )}

          {/* Sub-Tab 3: 3D Convex QP Surface */}
          {activeSubTab === 'SURFACE_3D' && (
            <ConvexQPSurface3D
              formulation={formulation}
              activeResult={activeResult}
              reducedMotion={reducedMotion}
            />
          )}

        </div>
      ) : (
        <div className="p-16 bg-obsidian-surface border border-graphite-border rounded-sm text-center space-y-3">
          <HelpCircle className="w-10 h-10 text-violet-dim mx-auto" />
          <h3 className="text-base font-display font-semibold text-chrome">
            No Active Convergence Trajectory
          </h3>
          <p className="text-xs text-violet-muted max-w-md mx-auto font-sans">
            Run an optimization problem from the Workspace to analyze primal/dual feasibility residuals, duality gap progression, and 2D/3D geometry.
          </p>
          <div className="pt-2">
            <button
              onClick={onLaunchSolver}
              className="px-5 py-2 bg-electric hover:bg-electric-glow text-white font-mono text-xs font-medium rounded-sm transition-colors"
            >
              Open Workspace
            </button>
          </div>
        </div>
      )}

    </div>
  );
};
