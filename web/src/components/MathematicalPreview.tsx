import React from 'react';
import type { SolverType, ObjectiveSense } from '../types/solver';

export interface VariableDef {
  id: string;
  name: string;
  c: number;
  type: 'CONTINUOUS' | 'INTEGER' | 'BINARY';
  lower: number;
  upper: number;
}

export interface ConstraintDef {
  id: string;
  name: string;
  coefficients: number[];
  sense: 'LEQ' | 'GEQ' | 'EQ';
  rhs: number;
}

interface MathematicalPreviewProps {
  solverType: SolverType;
  sense: ObjectiveSense;
  variables: VariableDef[];
  constraints: ConstraintDef[];
  QMatrix: number[][];
  validationErrors: string[];
}

export const MathematicalPreview: React.FC<MathematicalPreviewProps> = ({
  solverType,
  sense,
  variables,
  constraints,
  QMatrix,
  validationErrors,
}) => {
  // Format linear objective term
  const formatLinearObjective = () => {
    if (variables.length === 0) return '0';
    const terms = variables.map((v, i) => {
      const coeff = v.c;
      const sign = i > 0 ? (coeff >= 0 ? '+ ' : '- ') : (coeff < 0 ? '-' : '');
      const absCoeff = Math.abs(coeff);
      const coeffStr = absCoeff === 1 ? '' : `${absCoeff.toFixed(2)}`;
      return `${sign}${coeffStr}${v.name}`;
    });
    return terms.join(' ');
  };

  // Format quadratic objective term for QP: 1/2 x^T Q x
  const formatQuadraticObjective = () => {
    if (solverType !== 'QP' || QMatrix.length === 0) return null;
    const terms: string[] = [];
    for (let i = 0; i < variables.length && i < QMatrix.length; i++) {
      for (let j = 0; j < variables.length && j < QMatrix[i].length; j++) {
        const qVal = QMatrix[i][j];
        if (Math.abs(qVal) > 1e-12) {
          const var1 = variables[i].name;
          const var2 = variables[j].name;
          const term = i === j ? `${qVal.toFixed(2)}${var1}²` : `${qVal.toFixed(2)}${var1}${var2}`;
          terms.push(term);
        }
      }
    }
    if (terms.length === 0) return null;
    return `½ [ ${terms.join(' + ')} ]`;
  };

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      
      {/* Title & Mathematical Model Header */}
      <div className="flex items-center justify-between border-b border-graphite-border pb-2">
        <span className="text-electric font-semibold tracking-wider text-[11px] uppercase flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-electric" />
          Mathematical Formulation Preview
        </span>
        <span className="text-[10px] text-violet-muted border border-graphite-border px-1.5 py-0.5 rounded-sm">
          {solverType} · {sense}
        </span>
      </div>

      {/* Objective Function Formulation */}
      <div className="space-y-1">
        <span className="text-[10px] text-violet-muted uppercase block">Objective Function:</span>
        <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm text-chrome font-medium break-all">
          <span className="text-electric font-bold">{sense === 'MAXIMIZE' ? 'Maximize' : 'Minimize'}</span>{' '}
          Z ={' '}
          {solverType === 'QP' && formatQuadraticObjective() && (
            <>
              <span className="text-violet-dim">{formatQuadraticObjective()}</span>
              {variables.some((v) => Math.abs(v.c) > 1e-12) && ' + '}
            </>
          )}
          <span>{formatLinearObjective()}</span>
        </div>
      </div>

      {/* Constraints Formulation */}
      <div className="space-y-1">
        <span className="text-[10px] text-violet-muted uppercase block">Subject to Constraints:</span>
        <div className="p-2.5 bg-obsidian border border-graphite-border rounded-sm space-y-1.5 max-h-36 overflow-y-auto divide-y divide-graphite-border/50">
          {constraints.length > 0 ? (
            constraints.map((c) => {
              const terms = variables.map((v, vIdx) => {
                const coeff = c.coefficients[vIdx] ?? 0;
                if (Math.abs(coeff) < 1e-12) return null;
                const sign = termsCount > 0 ? (coeff >= 0 ? '+ ' : '- ') : (coeff < 0 ? '-' : '');
                const absCoeff = Math.abs(coeff);
                const coeffStr = absCoeff === 1 ? '' : `${absCoeff.toFixed(2)}`;
                termsCount++;
                return `${sign}${coeffStr}${v.name}`;
              });
              var termsCount = 0;
              const nonZeroTerms = terms.filter(Boolean);
              const lhs = nonZeroTerms.length > 0 ? nonZeroTerms.join(' ') : '0';
              const op = c.sense === 'LEQ' ? '≤' : c.sense === 'GEQ' ? '≥' : '=';

              return (
                <div key={c.id} className="pt-1 first:pt-0 flex items-center justify-between text-chrome">
                  <span className="text-violet-muted text-[10px] w-8">{c.name}:</span>
                  <span className="flex-1 px-2">{lhs} {op} {c.rhs.toFixed(2)}</span>
                </div>
              );
            })
          ) : (
            <div className="text-violet-dim text-[11px] py-1">No linear constraints defined (unconstrained).</div>
          )}
        </div>
      </div>

      {/* Variable Bounds and Integrality */}
      <div className="space-y-1">
        <span className="text-[10px] text-violet-muted uppercase block">Variable Bounds &amp; Classifications:</span>
        <div className="p-2 bg-obsidian border border-graphite-border rounded-sm text-[11px] text-violet-muted flex flex-wrap gap-2">
          {variables.map((v) => {
            const lb = v.lower <= -1e19 ? '-∞' : v.lower.toString();
            const ub = v.upper >= 1e19 ? '+∞' : v.upper.toString();
            const typeLabel = v.type === 'BINARY' ? '∈ {0, 1}' : v.type === 'INTEGER' ? '∈ ℤ' : '∈ ℝ';
            return (
              <span key={v.id} className="border border-graphite-border px-1.5 py-0.5 rounded-sm bg-obsidian-surface">
                {lb} ≤ <strong className="text-chrome">{v.name}</strong> ≤ {ub} <span className="text-violet-dim">({typeLabel})</span>
              </span>
            );
          })}
        </div>
      </div>

      {/* Real-time Validation Errors Highlight */}
      {validationErrors.length > 0 && (
        <div className="p-2.5 bg-red-950/40 border border-red-800 rounded-sm text-[11px] text-red-300 space-y-1">
          <span className="font-semibold block text-red-200 uppercase">// Formulation Inconsistencies:</span>
          {validationErrors.map((err, i) => (
            <div key={i} className="flex items-start gap-1.5">
              <span>•</span>
              <span>{err}</span>
            </div>
          ))}
        </div>
      )}

    </div>
  );
};
