import React from 'react';
import { Plus, Trash2 } from 'lucide-react';
import type { VariableDef } from './MathematicalPreview';

interface VariableEditorProps {
  variables: VariableDef[];
  setVariables: React.Dispatch<React.SetStateAction<VariableDef[]>>;
  isMilp: boolean;
}

export const VariableEditor: React.FC<VariableEditorProps> = ({
  variables,
  setVariables,
  isMilp,
}) => {
  const handleAddVariable = () => {
    const nextIdx = variables.length + 1;
    const newVar: VariableDef = {
      id: `var_${Date.now()}_${Math.random()}`,
      name: `x${nextIdx}`,
      c: 1.0,
      type: 'CONTINUOUS',
      lower: 0.0,
      upper: 1e20,
    };
    setVariables([...variables, newVar]);
  };

  const handleRemoveVariable = (idx: number) => {
    if (variables.length <= 1) return; // Keep at least one variable
    const updated = variables.filter((_, i) => i !== idx);
    setVariables(updated);
  };

  const handleUpdate = (idx: number, patch: Partial<VariableDef>) => {
    const updated = [...variables];
    const item = { ...updated[idx], ...patch };
    
    // Binary automatically clamps to [0, 1]
    if (item.type === 'BINARY') {
      item.lower = 0.0;
      item.upper = 1.0;
    }
    
    updated[idx] = item;
    setVariables(updated);
  };

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      
      <div className="flex items-center justify-between border-b border-graphite-border pb-2">
        <div>
          <span className="font-semibold text-chrome text-[11px] uppercase tracking-wider block">
            Decision Variables ({variables.length})
          </span>
          <span className="text-[10px] text-violet-muted">
            Define decision vector dimension, objective costs, bounds, and integer restrictions.
          </span>
        </div>
        <button
          onClick={handleAddVariable}
          className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded-sm text-[11px] flex items-center gap-1 transition-colors"
        >
          <Plus className="w-3.5 h-3.5 text-electric" />
          <span>Add Variable</span>
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-obsidian border-b border-graphite-border text-[10px] text-violet-muted uppercase">
            <tr>
              <th className="py-2 px-2.5">Name</th>
              <th className="py-2 px-2.5">Cost (cⱼ)</th>
              {isMilp && <th className="py-2 px-2.5">Type</th>}
              <th className="py-2 px-2.5">Lower (lⱼ)</th>
              <th className="py-2 px-2.5">Upper (uⱼ)</th>
              <th className="py-2 px-2 w-10 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-graphite-border">
            {variables.map((v, idx) => (
              <tr key={v.id} className="hover:bg-graphite/30 transition-colors">
                
                {/* Variable Name */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="text"
                    value={v.name}
                    onChange={(e) => handleUpdate(idx, { name: e.target.value })}
                    className="w-16 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome font-semibold focus:border-electric focus:outline-none"
                  />
                </td>

                {/* Objective Cost c */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="number"
                    step="any"
                    value={v.c}
                    onChange={(e) => handleUpdate(idx, { c: parseFloat(e.target.value) || 0.0 })}
                    className="w-20 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome focus:border-electric focus:outline-none"
                  />
                </td>

                {/* Variable Type (MILP only) */}
                {isMilp && (
                  <td className="py-1.5 px-2.5">
                    <select
                      value={v.type}
                      onChange={(e) => handleUpdate(idx, { type: e.target.value as any })}
                      className="bg-obsidian border border-graphite-border text-chrome text-[11px] rounded-sm px-1.5 py-0.5 focus:border-electric focus:outline-none"
                    >
                      <option value="CONTINUOUS">Continuous (ℝ)</option>
                      <option value="INTEGER">Integer (ℤ)</option>
                      <option value="BINARY">Binary ({'{0, 1}'})</option>
                    </select>
                  </td>
                )}

                {/* Lower Bound */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="number"
                    step="any"
                    disabled={v.type === 'BINARY'}
                    value={v.lower <= -1e19 ? '' : v.lower}
                    placeholder="-inf"
                    onChange={(e) => {
                      const val = e.target.value === '' ? -1e20 : parseFloat(e.target.value);
                      handleUpdate(idx, { lower: isNaN(val) ? 0.0 : val });
                    }}
                    className="w-18 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome focus:border-electric focus:outline-none disabled:opacity-50"
                  />
                </td>

                {/* Upper Bound */}
                <td className="py-1.5 px-2.5">
                  <input
                    type="number"
                    step="any"
                    disabled={v.type === 'BINARY'}
                    value={v.upper >= 1e19 ? '' : v.upper}
                    placeholder="+inf"
                    onChange={(e) => {
                      const val = e.target.value === '' ? 1e20 : parseFloat(e.target.value);
                      handleUpdate(idx, { upper: isNaN(val) ? 1e20 : val });
                    }}
                    className="w-18 bg-obsidian border border-graphite-border rounded-sm px-1.5 py-0.5 text-chrome focus:border-electric focus:outline-none disabled:opacity-50"
                  />
                </td>

                {/* Action Delete */}
                <td className="py-1.5 px-2 text-center">
                  <button
                    onClick={() => handleRemoveVariable(idx)}
                    disabled={variables.length <= 1}
                    className="p-1 text-violet-muted hover:text-red-400 disabled:opacity-30 rounded-sm transition-colors"
                    title="Remove variable"
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
