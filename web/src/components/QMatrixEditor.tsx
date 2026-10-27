import React from 'react';
import { RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';
import type { VariableDef } from './MathematicalPreview';

interface QMatrixEditorProps {
  variables: VariableDef[];
  qMatrix: number[][];
  setQMatrix: React.Dispatch<React.SetStateAction<number[][]>>;
}

export const QMatrixEditor: React.FC<QMatrixEditorProps> = ({
  variables,
  qMatrix,
  setQMatrix,
}) => {
  const n = variables.length;

  // Check symmetry: |Q[i][j] - Q[j][i]| < 1e-7
  let isSymmetric = true;
  let maxAsymmetry = 0;
  for (let i = 0; i < n && i < qMatrix.length; i++) {
    for (let j = 0; j < n && j < (qMatrix[i]?.length ?? 0); j++) {
      const diff = Math.abs((qMatrix[i]?.[j] ?? 0) - (qMatrix[j]?.[i] ?? 0));
      if (diff > maxAsymmetry) maxAsymmetry = diff;
      if (diff > 1e-7) {
        isSymmetric = false;
      }
    }
  }

  // Auto-symmetrize: Q_ij = Q_ji = (Q_ij + Q_ji) / 2
  const handleAutoSymmetrize = () => {
    const updated = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        const val1 = qMatrix[i]?.[j] ?? 0;
        const val2 = qMatrix[j]?.[i] ?? 0;
        return (val1 + val2) / 2;
      })
    );
    setQMatrix(updated);
  };

  const handleCellChange = (i: number, j: number, val: number) => {
    const updated = Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => {
        if (r === i && c === j) return val;
        return qMatrix[r]?.[c] ?? (r === c ? 1.0 : 0.0);
      })
    );
    setQMatrix(updated);
  };

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-graphite-border pb-2">
        <div>
          <span className="font-semibold text-chrome text-[11px] uppercase tracking-wider block flex items-center gap-1.5">
            <span className="text-electric font-bold">Q</span>
            Hessian Matrix Editor ({n} × {n})
          </span>
          <span className="text-[10px] text-violet-muted">
            Defines the quadratic term ½ xᵀ Q x. Must be symmetric positive semi-definite (PSD).
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {isSymmetric ? (
            <span className="px-2 py-0.5 rounded-sm bg-emerald-950/50 border border-emerald-800 text-emerald-400 text-[10px] flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              Symmetric
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded-sm bg-amber-950/50 border border-amber-800 text-amber-400 text-[10px] flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              Asymmetric (Δ {maxAsymmetry.toFixed(2e-4 > maxAsymmetry ? 4 : 2)})
            </span>
          )}
          <button
            onClick={handleAutoSymmetrize}
            className="px-2 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm text-[10px] flex items-center gap-1 transition-colors"
            title="Enforce symmetry: Q = (Q + Qᵀ) / 2"
          >
            <RefreshCw className="w-3 h-3 text-electric" />
            <span>Symmetrize</span>
          </button>
        </div>
      </div>

      <div className="overflow-x-auto max-h-56">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="bg-obsidian border-b border-graphite-border text-[10px] text-violet-muted uppercase">
              <th className="py-1 px-2 w-10 text-center">i \ j</th>
              {variables.map((v) => (
                <th key={v.id} className="py-1 px-2 text-center min-w-[70px]">
                  {v.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-graphite-border">
            {variables.map((vRow, r) => (
              <tr key={vRow.id} className="hover:bg-graphite/20">
                <td className="py-1.5 px-2 text-center font-semibold text-violet-muted bg-obsidian/60">
                  {vRow.name}
                </td>
                {variables.map((_, c) => {
                  const val = qMatrix[r]?.[c] ?? (r === c ? 1.0 : 0.0);
                  const isDiagonal = r === c;
                  return (
                    <td key={c} className="py-1 px-1.5 text-center">
                      <input
                        type="number"
                        step="any"
                        value={isNaN(val) ? '' : val}
                        onChange={(e) => {
                          const parsed = parseFloat(e.target.value);
                          handleCellChange(r, c, isNaN(parsed) ? 0 : parsed);
                        }}
                        className={`w-full bg-obsidian border rounded-sm px-1.5 py-1 text-center font-mono text-xs focus:border-electric focus:outline-none transition-colors ${
                          isDiagonal
                            ? 'text-chrome font-semibold border-graphite-muted'
                            : 'text-chrome/90 border-graphite-border'
                        }`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="text-[10px] text-violet-dim bg-obsidian/40 border border-graphite-border/60 p-2 rounded-sm">
        <span className="text-violet-muted font-semibold">// Numerical Note: </span>
        Frontend checks dimension ({n}×{n}) and matrix symmetry. Rigorous PSD confirmation is performed natively by the VØRTEX C++ Cholesky factorization interior-point engine.
      </div>
    </div>
  );
};
