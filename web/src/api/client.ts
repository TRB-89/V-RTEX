import type { 
  SolverResult, 
  ProblemPreset, 
  Phase8EBenchmarkDataset,
  BenchmarkRecord 
} from '../types/solver';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export interface ApiCapabilitiesResponse {
  engine: string;
  version: string;
  cxx_standard: string;
  precision: string;
  solvers: {
    LP: {
      algorithms: string[];
      senses: string[];
      constraints: string[];
      backends: string[];
      status_codes: string[];
    };
    MILP: {
      algorithms: string[];
      variable_types: string[];
      branching_strategies: string[];
      node_selection: string[];
      backends: string[];
      status_codes: string[];
    };
    QP: {
      algorithms: string[];
      problem_classes: string[];
      constraints: string[];
      backends: string[];
      status_codes: string[];
    };
  };
  supported_file_formats: {
    JSON: boolean;
    MPS: boolean;
    LP_FILE: boolean;
  };
  cuda: {
    available: boolean;
    device_name: string | null;
    compute_capability: string | null;
    multiprocessors: number | null;
    total_vram_bytes: number | null;
    kernels: string[];
  };
  limits: {
    max_variables: number;
    max_constraints: number;
    max_elements: number;
  };
}

export interface ApiHealthResponse {
  status: string;
  engine: string;
  version: string;
  executable_exists: boolean;
  executable_path: string | null;
}

export class SolverApiClient {
  static async getHealth(): Promise<ApiHealthResponse> {
    const res = await fetch(`${API_BASE}/api/v1/health`);
    if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
    return res.json();
  }

  static async getCapabilities(): Promise<ApiCapabilitiesResponse> {
    const res = await fetch(`${API_BASE}/api/v1/capabilities`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Failed to query capabilities');
    }
    return res.json();
  }

  static async solve(payload: Record<string, any>, signal?: AbortSignal): Promise<SolverResult> {
    const res = await fetch(`${API_BASE}/api/v1/solve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Solver error: HTTP ${res.status}`);
    }

    return res.json();
  }

  static async getPresets(): Promise<ProblemPreset[]> {
    const res = await fetch(`${API_BASE}/api/v1/presets`);
    if (!res.ok) throw new Error('Failed to load presets');
    return res.json();
  }

  static async getBenchmarks(): Promise<Phase8EBenchmarkDataset> {
    const res = await fetch(`${API_BASE}/api/v1/benchmarks`);
    if (!res.ok) throw new Error('Failed to load Phase 8E benchmarks');
    return res.json();
  }

  static async getBenchmarkById(id: string): Promise<{ system_info: any; benchmark: BenchmarkRecord }> {
    const res = await fetch(`${API_BASE}/api/v1/benchmarks/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(`Failed to load benchmark record '${id}'`);
    return res.json();
  }

  static getArtifactUrl(filename: string): string {
    return `${API_BASE}/api/v1/benchmarks/artifacts/${encodeURIComponent(filename)}`;
  }
}
