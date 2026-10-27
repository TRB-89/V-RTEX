import React, { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Play, 
  RotateCcw, 
  XCircle, 
  Download,
  Upload,
  StopCircle,
  FileCode2,
  Info
} from 'lucide-react';
import type { 
  SolverType, 
  ObjectiveSense, 
  SolverBackend, 
  SolverResult, 
  ProblemPreset,
  ProblemFormulation
} from '../types/solver';
import { SolverApiClient, type ApiCapabilitiesResponse } from '../api/client';
import { VariableEditor } from './VariableEditor';
import { ConstraintEditor } from './ConstraintEditor';
import { QMatrixEditor } from './QMatrixEditor';
import { MathematicalPreview, type VariableDef, type ConstraintDef } from './MathematicalPreview';
import { ResultsInspector } from './ResultsInspector';

interface WorkspaceProps {
  presets?: ProblemPreset[];
  activeResult: SolverResult | null;
  setActiveResult: (res: SolverResult | null) => void;
  onInspectMatrix?: () => void;
  onViewConvergence?: () => void;
  onFormulationChange?: (formulation: ProblemFormulation) => void;
}

// Built-in verified reference problems with known mathematical solutions
const DEFAULT_PRESETS: ProblemPreset[] = [
  {
    id: 'sih26119-lp',
    name: 'SIH26119 Simplex Benchmark',
    description: 'Reference 2-variable LP: Max 3x1 + 5x2 s.t. x1 <= 4, 2x2 <= 12, 3x1 + 2x2 <= 18. Known Z* = 36.000',
    solver_type: 'LP',
    sense: 'MAXIMIZE',
    backend: 'AUTO',
    c: [3.0, 5.0],
    A: [
      [1.0, 0.0],
      [0.0, 2.0],
      [3.0, 2.0],
    ],
    b: [4.0, 12.0, 18.0],
    constraint_types: ['LEQ', 'LEQ', 'LEQ'],
    variable_types: ['CONTINUOUS', 'CONTINUOUS'],
    bounds: [
      { lower: 0.0, upper: 1e20 },
      { lower: 0.0, upper: 1e20 },
    ],
  },
  {
    id: 'two-phase-lp',
    name: 'Two-Phase Mixed LP',
    description: 'LP requiring Phase 1 artificials: Max 2x1 + 3x2 s.t. x1 + x2 >= 2, x1 <= 3, x2 <= 4. Known Z* = 18.000',
    solver_type: 'LP',
    sense: 'MAXIMIZE',
    backend: 'AUTO',
    c: [2.0, 3.0],
    A: [
      [1.0, 1.0],
      [1.0, 0.0],
      [0.0, 1.0],
    ],
    b: [2.0, 3.0, 4.0],
    constraint_types: ['GEQ', 'LEQ', 'LEQ'],
    variable_types: ['CONTINUOUS', 'CONTINUOUS'],
    bounds: [
      { lower: 0.0, upper: 1e20 },
      { lower: 0.0, upper: 1e20 },
    ],
  },
  {
    id: 'knapsack-milp',
    name: '0-1 Knapsack Problem',
    description: 'Binary Integer Program: Max 10x1 + 40x2 + 30x3 + 50x4 s.t. 5x1 + 4x2 + 6x3 + 3x4 <= 10. Known Z* = 90.000',
    solver_type: 'MILP',
    sense: 'MAXIMIZE',
    backend: 'CPU',
    c: [10.0, 40.0, 30.0, 50.0],
    A: [
      [5.0, 4.0, 6.0, 3.0],
    ],
    b: [10.0],
    constraint_types: ['LEQ'],
    variable_types: ['BINARY', 'BINARY', 'BINARY', 'BINARY'],
    bounds: [
      { lower: 0.0, upper: 1.0 },
      { lower: 0.0, upper: 1.0 },
      { lower: 0.0, upper: 1.0 },
      { lower: 0.0, upper: 1.0 },
    ],
  },
  {
    id: 'portfolio-qp',
    name: 'Constrained Portfolio QP',
    description: 'Quadratic Program: Min 0.5(2x1^2 + 2x2^2) - 2x1 - 5x2 s.t. x1 + x2 <= 2. Known Z* = -6.125',
    solver_type: 'QP',
    sense: 'MINIMIZE',
    backend: 'AUTO',
    c: [-2.0, -5.0],
    A: [
      [1.0, 1.0],
    ],
    b: [2.0],
    constraint_types: ['LEQ'],
    variable_types: ['CONTINUOUS', 'CONTINUOUS'],
    bounds: [
      { lower: 0.0, upper: 1e20 },
      { lower: 0.0, upper: 1e20 },
    ],
    Q: [
      [2.0, 0.0],
      [0.0, 2.0],
    ],
  },
];

export const OptimizationWorkspace: React.FC<WorkspaceProps> = ({
  presets = DEFAULT_PRESETS,
  activeResult,
  setActiveResult,
  onInspectMatrix,
  onViewConvergence,
  onFormulationChange,
}) => {
  // Solver formulation metadata
  const [solverType, setSolverType] = useState<SolverType>('LP');
  const [sense, setSense] = useState<ObjectiveSense>('MAXIMIZE');
  const [backend, setBackend] = useState<SolverBackend>('AUTO');

  // Dynamic capabilities state from backend
  const [capabilities, setCapabilities] = useState<ApiCapabilitiesResponse | null>(null);
  const [unsupportedNotice, setUnsupportedNotice] = useState<string | null>(null);

  // Formulation variables & constraints
  const [variables, setVariables] = useState<VariableDef[]>([
    { id: 'v1', name: 'x1', c: 3.0, type: 'CONTINUOUS', lower: 0.0, upper: 1e20 },
    { id: 'v2', name: 'x2', c: 5.0, type: 'CONTINUOUS', lower: 0.0, upper: 1e20 },
  ]);

  const [constraints, setConstraints] = useState<ConstraintDef[]>([
    { id: 'c1', name: 'c1', coefficients: [1.0, 0.0], sense: 'LEQ', rhs: 4.0 },
    { id: 'c2', name: 'c2', coefficients: [0.0, 2.0], sense: 'LEQ', rhs: 12.0 },
    { id: 'c3', name: 'c3', coefficients: [3.0, 2.0], sense: 'LEQ', rhs: 18.0 },
  ]);

  // QP Hessian Q (size n x n)
  const [qMatrix, setQMatrix] = useState<number[][]>([
    [2.0, 0.0],
    [0.0, 2.0],
  ]);

  // Solver execution state
  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('sih26119-lp');
  const abortControllerRef = useRef<AbortController | null>(null);

  // File import ref
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Fetch capabilities on mount
  useEffect(() => {
    let isMounted = true;
    SolverApiClient.getCapabilities()
      .then((caps) => {
        if (isMounted) setCapabilities(caps);
      })
      .catch((err) => {
        console.warn('Could not query /api/v1/capabilities:', err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  // Sync Q dimension whenever variable count changes
  useEffect(() => {
    const n = variables.length;
    setQMatrix((prev) => {
      const updated: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row: number[] = [];
        for (let j = 0; j < n; j++) {
          row.push(prev[i]?.[j] ?? (i === j ? 1.0 : 0.0));
        }
        updated.push(row);
      }
      return updated;
    });

    // Also sync constraint coefficient lengths
    setConstraints((prevConstraints) =>
      prevConstraints.map((c) => {
        const coeffs = [...c.coefficients];
        while (coeffs.length < n) coeffs.push(0.0);
        return { ...c, coefficients: coeffs.slice(0, n) };
      })
    );
  }, [variables.length]);

  // Propagate active formulation to parent for Matrix Inspector & Convergence Lab
  useEffect(() => {
    if (onFormulationChange) {
      onFormulationChange({
        solverType,
        sense,
        backend,
        c: variables.map((v) => v.c),
        A: constraints.map((c) => c.coefficients.slice(0, variables.length)),
        b: constraints.map((c) => c.rhs),
        constraint_types: constraints.map((c) => c.sense),
        variable_types: variables.map((v) => v.type),
        bounds: variables.map((v) => ({ lower: v.lower, upper: v.upper })),
        Q: solverType === 'QP' ? qMatrix : undefined,
        variableNames: variables.map((v) => v.name),
        constraintNames: constraints.map((c) => c.name),
      });
    }
  }, [solverType, sense, backend, variables, constraints, qMatrix, onFormulationChange]);

  // Real-time validation checks
  const validationErrors = useMemo(() => {
    const errors: string[] = [];

    if (variables.length === 0) {
      errors.push('Problem must have at least one decision variable.');
    }

    if (constraints.length === 0) {
      errors.push('Problem must have at least one constraint.');
    }

    // Check variable bounds and NaN
    variables.forEach((v, idx) => {
      if (isNaN(v.c) || !isFinite(v.c)) {
        errors.push(`Variable ${v.name || `x${idx + 1}`} has invalid cost coefficient.`);
      }
      if (isNaN(v.lower) || !isFinite(v.lower)) {
        errors.push(`Variable ${v.name || `x${idx + 1}`} has invalid lower bound.`);
      }
      if (isNaN(v.upper)) {
        errors.push(`Variable ${v.name || `x${idx + 1}`} has invalid upper bound.`);
      }
      if (v.lower > v.upper) {
        errors.push(`Variable ${v.name || `x${idx + 1}`} has lower bound > upper bound (${v.lower} > ${v.upper}).`);
      }
    });

    // Check constraints
    constraints.forEach((c, idx) => {
      if (isNaN(c.rhs) || !isFinite(c.rhs)) {
        errors.push(`Constraint ${c.name || `c${idx + 1}`} has invalid right-hand side (RHS).`);
      }
      c.coefficients.slice(0, variables.length).forEach((coeff, vIdx) => {
        if (isNaN(coeff) || !isFinite(coeff)) {
          errors.push(`Constraint ${c.name || `c${idx + 1}`} has invalid coefficient for variable ${variables[vIdx]?.name}.`);
        }
      });
    });

    // QP specific checks
    if (solverType === 'QP') {
      const n = variables.length;
      if (qMatrix.length !== n) {
        errors.push(`Q matrix row count (${qMatrix.length}) does not match variable dimension (${n}).`);
      }
      for (let i = 0; i < n && i < qMatrix.length; i++) {
        if (qMatrix[i].length !== n) {
          errors.push(`Q matrix row ${i + 1} column count (${qMatrix[i].length}) does not match variable dimension (${n}).`);
          break;
        }
        for (let j = 0; j < n; j++) {
          const val = qMatrix[i][j];
          if (isNaN(val) || !isFinite(val)) {
            errors.push(`Q matrix entry (${i + 1}, ${j + 1}) is NaN or non-finite.`);
          }
          const diff = Math.abs(val - (qMatrix[j]?.[i] ?? 0));
          if (diff > 1e-7) {
            errors.push(`Q matrix is asymmetric at (${i + 1}, ${j + 1}): Q_ij = ${val}, Q_ji = ${qMatrix[j]?.[i]}. Click 'Symmetrize' to fix.`);
            break;
          }
        }
      }
    }

    return errors;
  }, [variables, constraints, solverType, qMatrix]);

  // Load preset problem
  const handleLoadPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    const p = presets.find((item) => item.id === presetId);
    if (!p) return;

    setSolverType(p.solver_type);
    setSense(p.sense);
    setBackend(p.backend);

    // Map variables
    const vars: VariableDef[] = p.c.map((cost, idx) => ({
      id: `preset_v_${idx + 1}`,
      name: `x${idx + 1}`,
      c: cost,
      type: p.variable_types?.[idx] ?? 'CONTINUOUS',
      lower: p.bounds?.[idx]?.lower ?? 0.0,
      upper: p.bounds?.[idx]?.upper ?? 1e20,
    }));
    setVariables(vars);

    // Map constraints
    const constrs: ConstraintDef[] = p.A.map((row, idx) => ({
      id: `preset_c_${idx + 1}`,
      name: `c${idx + 1}`,
      coefficients: [...row],
      sense: p.constraint_types?.[idx] ?? 'LEQ',
      rhs: p.b[idx] ?? 0.0,
    }));
    setConstraints(constrs);

    // Map Q if QP
    if (p.Q && p.Q.length > 0) {
      setQMatrix(p.Q);
    }

    setExecutionError(null);
    setActiveResult(null);
  };

  // Run Solver Execution
  const handleSolve = async () => {
    if (validationErrors.length > 0) {
      setExecutionError(validationErrors[0]);
      return;
    }

    setIsSolving(true);
    setExecutionError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const c = variables.map((v) => v.c);
      const b = constraints.map((item) => item.rhs);
      const A = constraints.map((item) => item.coefficients.slice(0, variables.length));
      const constraint_types = constraints.map((item) => item.sense);

      const payload: Record<string, any> = {
        solver_type: solverType,
        sense,
        backend,
        c,
        A,
        b,
        constraint_types,
        options: {},
      };

      if (solverType === 'MILP') {
        payload.variable_types = variables.map((v) => v.type);
        payload.bounds = variables.map((v) => ({ lower: v.lower, upper: v.upper }));
        payload.options = { node_limit: 10000, time_limit_sec: 60.0 };
      }

      if (solverType === 'QP') {
        payload.Q = qMatrix;
        payload.options = {
          primal_tolerance: 1e-7,
          dual_tolerance: 1e-7,
          duality_gap_tolerance: 1e-7,
          max_iterations: 100,
        };
      }

      const res = await SolverApiClient.solve(payload, controller.signal);
      setActiveResult(res);
    } catch (err: any) {
      if (err.name === 'AbortError' || err.message?.includes('aborted')) {
        setActiveResult({
          success: false,
          solver_type: solverType,
          status: 'CLIENT_ABORTED',
          message: 'Solve request was cancelled by the user in the browser.',
        });
      } else {
        setExecutionError(err.message || 'An unexpected solver error occurred.');
      }
    } finally {
      setIsSolving(false);
      abortControllerRef.current = null;
    }
  };

  // Cancel Solver Execution
  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  // Reset workspace to blank default
  const handleReset = () => {
    setVariables([
      { id: 'v1', name: 'x1', c: 1.0, type: 'CONTINUOUS', lower: 0.0, upper: 1e20 },
      { id: 'v2', name: 'x2', c: 1.0, type: 'CONTINUOUS', lower: 0.0, upper: 1e20 },
    ]);
    setConstraints([
      { id: 'c1', name: 'c1', coefficients: [1.0, 1.0], sense: 'LEQ', rhs: 10.0 },
    ]);
    setQMatrix([
      [2.0, 0.0],
      [0.0, 2.0],
    ]);
    setSelectedPresetId('');
    setExecutionError(null);
    setActiveResult(null);
  };

  // Versioned Problem JSON Export
  const handleExportProblemJson = () => {
    const problemData = {
      schema_version: 'vortex_problem_v1',
      solver_type: solverType,
      sense,
      backend,
      variables,
      constraints,
      qMatrix: solverType === 'QP' ? qMatrix : undefined,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(problemData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vortex_problem_${solverType.toLowerCase()}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Versioned Problem JSON Import
  const handleImportProblemJson = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsed = JSON.parse(content);
        if (parsed.schema_version !== 'vortex_problem_v1') {
          throw new Error('Invalid problem file format: missing "vortex_problem_v1" schema version.');
        }

        if (parsed.solver_type) setSolverType(parsed.solver_type);
        if (parsed.sense) setSense(parsed.sense);
        if (parsed.backend) setBackend(parsed.backend);
        if (Array.isArray(parsed.variables)) setVariables(parsed.variables);
        if (Array.isArray(parsed.constraints)) setConstraints(parsed.constraints);
        if (Array.isArray(parsed.qMatrix)) setQMatrix(parsed.qMatrix);

        setExecutionError(null);
        setSelectedPresetId('');
      } catch (err: any) {
        setExecutionError(`Failed to import problem: ${err.message}`);
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Disclaimer for MPS / LP format import
  const handleUnsupportedFormatClick = (formatName: string) => {
    setUnsupportedNotice(
      `Format [${formatName}] is not compiled into the current native engine (supported_file_formats: {JSON: true, MPS: false, LP_FILE: false}). Import problem formulations via standard VØRTEX JSON format.`
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-graphite-border gap-4">
        <div>
          <div className="text-xs font-mono text-electric tracking-editorial uppercase flex items-center gap-2">
            <span>// PHASE 9C INTERACTIVE WORKSPACE</span>
            {capabilities && (
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-graphite text-chrome border border-graphite-border">
                ENGINE: {capabilities.engine} {capabilities.version}
              </span>
            )}
          </div>
          <h1 className="text-2xl sm:text-3xl font-display font-bold text-chrome mt-0.5">
            Optimization Formulation Console
          </h1>
          <p className="text-violet-muted text-xs sm:text-sm mt-0.5">
            Interactive research environment connected directly to the native C++20 / CUDA physical optimization solver.
          </p>
        </div>

        {/* Preset Selector & File Import/Export Controls */}
        <div className="flex flex-wrap items-center gap-2">
          
          {/* Preset dropdown */}
          <div className="flex items-center space-x-1.5">
            <span className="text-xs font-mono text-violet-muted whitespace-nowrap">Preset:</span>
            <select
              value={selectedPresetId}
              onChange={(e) => handleLoadPreset(e.target.value)}
              className="bg-obsidian-surface border border-graphite-border text-chrome text-xs font-mono rounded-sm px-2.5 py-1.5 focus:border-electric focus:outline-none"
            >
              <option value="">-- Custom Problem --</option>
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} [{p.solver_type}]
                </option>
              ))}
            </select>
          </div>

          {/* Hidden File Input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImportProblemJson}
            accept=".json"
            className="hidden"
          />

          {/* Import JSON */}
          <button
            onClick={() => fileInputRef.current?.click()}
            title="Import problem from vortex_problem_v1 JSON"
            className="px-2.5 py-1.5 bg-obsidian-surface hover:bg-graphite text-chrome border border-graphite-border rounded-sm text-xs font-mono flex items-center gap-1 transition-colors"
          >
            <Upload className="w-3.5 h-3.5 text-violet-muted" />
            <span>Import</span>
          </button>

          {/* Export JSON */}
          <button
            onClick={handleExportProblemJson}
            title="Export problem definition as versioned JSON"
            className="px-2.5 py-1.5 bg-obsidian-surface hover:bg-graphite text-chrome border border-graphite-border rounded-sm text-xs font-mono flex items-center gap-1 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-violet-muted" />
            <span>Export</span>
          </button>

          {/* Disabled MPS / LP parser buttons with honest explanation */}
          <button
            onClick={() => handleUnsupportedFormatClick('MPS')}
            title="MPS parser capability check"
            className="px-2 py-1.5 bg-obsidian-surface/50 text-violet-dim border border-graphite-border/50 rounded-sm text-xs font-mono flex items-center gap-1 cursor-help"
          >
            <FileCode2 className="w-3 h-3 text-violet-dim" />
            <span className="line-through">MPS</span>
          </button>

          <button
            onClick={() => handleUnsupportedFormatClick('CPLEX LP')}
            title="LP parser capability check"
            className="px-2 py-1.5 bg-obsidian-surface/50 text-violet-dim border border-graphite-border/50 rounded-sm text-xs font-mono flex items-center gap-1 cursor-help"
          >
            <FileCode2 className="w-3 h-3 text-violet-dim" />
            <span className="line-through">.LP</span>
          </button>

        </div>
      </div>

      {/* Unsupported Notice Banner */}
      {unsupportedNotice && (
        <div className="p-3 bg-obsidian border border-electric/40 rounded-sm text-chrome text-xs font-mono flex items-start justify-between gap-2 shadow-[0_0_15px_rgba(37,20,237,0.15)]">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-electric mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-electric uppercase tracking-wider block">Engine Capability Note</span>
              <p className="text-chrome/90 mt-0.5">{unsupportedNotice}</p>
            </div>
          </div>
          <button
            onClick={() => setUnsupportedNotice(null)}
            className="text-violet-muted hover:text-chrome px-1 text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Two-Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Problem Builder & Preview (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          
          {/* Controls Bar: Solver Type, Sense, Backend */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-4 bg-obsidian-surface border border-graphite-border rounded-sm font-mono">
            
            {/* Solver Type Selection */}
            <div>
              <label className="block text-[11px] text-violet-muted uppercase mb-1">
                Mathematical Model
              </label>
              <div className="flex rounded-sm bg-obsidian border border-graphite-border p-0.5 text-xs">
                {(['LP', 'MILP', 'QP'] as SolverType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => {
                      setSolverType(type);
                      setExecutionError(null);
                    }}
                    className={`flex-1 py-1 rounded-sm transition-colors ${
                      solverType === type
                        ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                        : 'text-violet-muted hover:text-chrome'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Objective Sense */}
            <div>
              <label className="block text-[11px] text-violet-muted uppercase mb-1">
                Objective Sense
              </label>
              <div className="flex rounded-sm bg-obsidian border border-graphite-border p-0.5 text-xs">
                {(['MAXIMIZE', 'MINIMIZE'] as ObjectiveSense[]).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSense(s)}
                    className={`flex-1 py-1 rounded-sm transition-colors ${
                      sense === s
                        ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                        : 'text-violet-muted hover:text-chrome'
                    }`}
                  >
                    {s === 'MAXIMIZE' ? 'MAXIMIZE' : 'MINIMIZE'}
                  </button>
                ))}
              </div>
            </div>

            {/* Compute Backend */}
            <div>
              <label className="block text-[11px] text-violet-muted uppercase mb-1">
                Execution Backend
              </label>
              <div className="flex rounded-sm bg-obsidian border border-graphite-border p-0.5 text-xs">
                {(['AUTO', 'GPU', 'CPU'] as SolverBackend[]).map((b) => (
                  <button
                    key={b}
                    onClick={() => setBackend(b)}
                    className={`flex-1 py-1 rounded-sm transition-colors ${
                      backend === b
                        ? 'bg-graphite text-chrome font-semibold border border-graphite-muted'
                        : 'text-violet-muted hover:text-chrome'
                    }`}
                  >
                    {b}
                  </button>
                ))}
              </div>
            </div>

          </div>

          {/* Module 2: Variable Editor */}
          <VariableEditor
            variables={variables}
            setVariables={setVariables}
            isMilp={solverType === 'MILP'}
          />

          {/* Module 2: Constraint Editor */}
          <ConstraintEditor
            constraints={constraints}
            setConstraints={setConstraints}
            variables={variables}
          />

          {/* Module 2: Q Matrix Editor (Only when QP selected) */}
          {solverType === 'QP' && (
            <QMatrixEditor
              variables={variables}
              qMatrix={qMatrix}
              setQMatrix={setQMatrix}
            />
          )}

          {/* Module 3: Live Mathematical Preview */}
          <MathematicalPreview
            solverType={solverType}
            sense={sense}
            variables={variables}
            constraints={constraints}
            QMatrix={qMatrix}
            validationErrors={validationErrors}
          />

          {/* Execution Error Banner */}
          {executionError && (
            <div className="p-3 bg-red-950/40 border border-red-800/80 rounded-sm text-red-200 text-xs font-mono flex items-start gap-2">
              <XCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
              <div>
                <div className="font-semibold">Execution Precheck / Solver Error</div>
                <div className="mt-0.5 text-red-300">{executionError}</div>
              </div>
            </div>
          )}

          {/* Action Bar: Run / Cancel / Reset */}
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-3">
              {isSolving ? (
                <button
                  onClick={handleCancel}
                  className="px-6 py-2.5 bg-rose-700 hover:bg-rose-600 active:bg-rose-800 text-white font-mono text-xs font-semibold tracking-wider rounded-sm transition-all flex items-center gap-2 shadow-[0_0_15px_rgba(225,29,72,0.4)]"
                >
                  <StopCircle className="w-4 h-4" />
                  <span>CANCEL SOLVER (ABORT)</span>
                </button>
              ) : (
                <button
                  onClick={handleSolve}
                  disabled={validationErrors.length > 0}
                  className="px-6 py-2.5 bg-electric hover:bg-electric-glow active:bg-electric-deep disabled:opacity-40 disabled:cursor-not-allowed text-white font-mono text-xs font-semibold tracking-wider rounded-sm transition-all duration-150 flex items-center gap-2 shadow-[0_0_20px_rgba(37,20,237,0.35)]"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>RUN NATIVE SOLVER</span>
                </button>
              )}

              {validationErrors.length > 0 && (
                <span className="text-[11px] font-mono text-amber-400 flex items-center gap-1">
                  <span>Fix {validationErrors.length} validation {validationErrors.length === 1 ? 'error' : 'errors'}</span>
                </span>
              )}
            </div>

            <button
              onClick={handleReset}
              disabled={isSolving}
              className="px-4 py-2 bg-obsidian-surface hover:bg-graphite text-violet-muted hover:text-chrome font-mono text-xs rounded-sm transition-colors border border-graphite-border flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>

        </div>

        {/* Right Column: Results & Telemetry Inspector (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <ResultsInspector
            result={activeResult}
            variables={variables}
            constraints={constraints}
            isSolving={isSolving}
            onInspectMatrix={onInspectMatrix}
            onViewConvergence={onViewConvergence}
          />
        </div>

      </div>

    </div>
  );
};
