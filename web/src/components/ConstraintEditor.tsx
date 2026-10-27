import React from 'react';
import { Plus, Trash2 } from 'lucide-react';
import type { ConstraintDef, VariableDef } from './MathematicalPreview';

interface ConstraintEditorProps {
  constraints: ConstraintDef[];
  setConstraints: React.Dispatch<React.SetStateAction<ConstraintDef[]>>;
  variables: VariableDef[];
}

export const ConstraintEditor: React.FC<ConstraintEditorProps> = ({
  constraints,
  setConstraints,
  variables,
}) => {
  const handleAddConstraint = () => {
    const nextIdx = constraints.length + 1;
    const newConstraint: ConstraintDef = {
      id: `c_${Date.now()}_${Math.random()}`,
      name: `c${nextIdx}`,
      coefficients: new Array(variables.length).fill(0.0),
      sense: 'LEQ',
      rhs: 10.0,
    };
    setConstraints([...constraints, newConstraint]);
  };

  const handleRemoveConstraint = (idx: number) => {
    const updated = constraints.filter((_, i) => i !== idx);
    setConstraints(updated);
  };

  const handleUpdateCoeff = (cIdx: number, vIdx: number, val: number) => {
    const updated = [...constraints];
    const coeffs = [...updated[cIdx].coefficients];
    // Ensure array is sufficiently long
    while (coeffs.length < variables.length) coeffs.push(0.0);
    coeffs[vIdx] = val;
    updated[cIdx] = { ...updated[cIdx], coefficients: coeffs };
    setConstraints(updated);
  };

  const handleUpdateConstraint = (cIdx: number, patch: Partial<ConstraintDef>) => {
    const updated = [...constraints];
    updated[cIdx] = { ...updated[cIdx], ...patch };
    setConstraints(updated);
  };

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      
      <div className="flex items-center justify-between border-b border-graphite-border pb-2">
        <div>
          <span className="font-semibold text-chrome text-[11px] uppercase tracking-wider block">
            Linear Constraints ({constraints.length} Rows)
          </span>
          <span className="text-[10px] text-violet-muted">
            Configure constraint matrix A, inequality/equality relations, and RHS vector b.
          </span>
        </div>
        <button
          onClick={handleAddConstraint}
          className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm text-[11px] flex items-center gap-1 transition-colors"
        >
          <Plus className="w-3.5 h-3.5 text-electric" />
          <span>Add Constraint</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-obsidian border-b border-graphite-border text-[10px] text-violet-muted uppercase">
            <tr>
              <th className="py-2 px-2.5">Name</th>
              {variables.map((v) => (
                <th key={v.id} className="py-2 px-2.5">
                  Coeff ({v.name})
                </th>
              ))}
              <th className="py-2 px-2.5">Relation</th>
              <th className="py-2 px-2.5">RHS (b)</th>
              <th className="py-2 px-2 w-10 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-graphite-border">
            {constraints.map((c, cIdx) => (
              <tr key={c.id} className="hover:bg-graphite/30 transition-colors">
                
                {/* Constraint Name */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="text"
                    value={c.name}
                    onChange={(e) => handleUpdateConstraint(cIdx, { name: e.target.value })}
                    className="w-16 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome font-semibold focus:border-electric focus:outline-none"
                  />
                </td>

                {/* Coefficients for each variable */}
                {variables.map((v, vIdx) => {
                  const val = c.coefficients[vIdx] ?? 0.0;
                  return (
                    <td key={v.id} className="py-1.5 px-2.5">
                      <input
                        type="number"
                        step="any"
                        value={val}
                        onChange={(e) => handleUpdateCoeff(cIdx, vIdx, parseFloat(e.target.value) || 0.0)}
                        className="w-18 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome focus:border-electric focus:outline-none"
                      />
                    </td>
                  );
                })}

                {/* Constraint Sense (LEQ, GEQ, EQ) */}
                <td className="py-1.5 px-2.5">
                  <select
                    value={c.sense}
                    onChange={(e) => handleUpdateConstraint(cIdx, { sense: e.target.value as any })}
                    className="bg-obsidian border border-graphite-border text-chrome text-[11px] rounded-sm px-2 py-0.5 focus:border-electric focus:outline-none"
                  >
                    <option value="LEQ">≤ (LEQ)</option>
                    <option value="GEQ">≥ (GEQ)</option>
                    <option value="EQ">= (EQ)</option>
                  </select>
                </td>

                {/* Right-Hand Side b */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="number"
                    step="any"
                    value={c.rhs}
                    onChange={(e) => handleUpdateConstraint(cIdx, { rhs: parseFloat(e.target.value) || 0.0 })}
                    className="w-20 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome font-semibold focus:border-electric focus:outline-none"
                  />
                </td>

                {/* Action Delete */}
                <td className="py-1.5 px-2 text-center">
                  <button
                    onClick={() => handleRemoveConstraint(cIdx)}
                    className="p-1 text-violet-muted hover:text-red-400 rounded-sm transition-colors"
                    title="Remove constraint"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </td>

              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
};
