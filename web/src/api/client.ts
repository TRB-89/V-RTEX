import type { 
  SolverResult, 
  ProblemPreset, 
  Phase8EBenchmarkDataset,
  BenchmarkRecord 
} from '../types/solver';

/**
 * Normalizes and cleans the base URL from environment configuration.
 * Defaults to http://localhost:8000 for local development.
 * Supports VITE_API_BASE_URL (standard) and VITE_API_URL (legacy).
 * Strips trailing slashes to ensure consistent URL composition.
 */
export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL;
  if (!envUrl || typeof envUrl !== 'string') {
    return 'http://localhost:8000';
  }
  const trimmed = envUrl.trim().replace(/\/+$/, '');
  return trimmed || 'http://localhost:8000';
}

export const API_BASE = getApiBaseUrl();

/**
 * Safely constructs an absolute endpoint URL without duplicate /api or /api/v1 prefixes.
 */
export function buildApiUrl(path: string): string {
  const base = getApiBaseUrl();
  const cleanPath = path.startsWith('/') ? path : `/${path}`;

  // If base already ends with /api/v1 and path starts with /api/v1
  if (base.endsWith('/api/v1') && cleanPath.startsWith('/api/v1')) {
    return `${base}${cleanPath.slice(7)}`;
  }
  // If base already ends with /api and path starts with /api/
  if (base.endsWith('/api') && cleanPath.startsWith('/api/')) {
    return `${base}${cleanPath.slice(4)}`;
  }

  return `${base}${cleanPath}`;
}

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
  executable_runnable?: boolean;
  executable_path: string | null;
  platform?: string;
}

export class ApiError extends Error {
  statusCode?: number;
  isColdStartTimeout?: boolean;

  constructor(message: string, statusCode?: number, isColdStartTimeout = false) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.isColdStartTimeout = isColdStartTimeout;
  }
}

/**
 * Fetch with configurable timeout accounting for Render Free initial cold starts.
 */
async function requestWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = 60000
): Promise<Response> {
  const { signal: callerSignal, ...init } = options;
  const controller = new AbortController();

  let isTimedOut = false;
  const timeoutId = setTimeout(() => {
    isTimedOut = true;
    controller.abort();
  }, timeoutMs);

  if (callerSignal) {
    callerSignal.addEventListener('abort', () => {
      clearTimeout(timeoutId);
      controller.abort();
    });
  }

  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (isTimedOut) {
      throw new ApiError(
        `Request to ${url} timed out after ${Math.round(timeoutMs / 1000)}s. Render Free services take 30–50s to cold start from sleep.`,
        504,
        true
      );
    }
    if (err.name === 'AbortError') {
      throw err;
    }
    throw new ApiError(
      `Network failure connecting to ${url}: ${err.message || 'Server unreachable'}. Verify your connection or backend status.`,
      0
    );
  }
}

export class SolverApiClient {
  /**
   * Health check query against GET /api/v1/health
   */
  static async getHealth(signal?: AbortSignal, timeoutMs = 45000): Promise<ApiHealthResponse> {
    const url = buildApiUrl('/api/v1/health');
    const res = await requestWithTimeout(url, { signal }, timeoutMs);
    if (!res.ok) {
      throw new ApiError(`Health check failed: HTTP ${res.status}`, res.status);
    }
    return res.json();
  }

  /**
   * Hardware & solver capability query against GET /api/v1/capabilities
   */
  static async getCapabilities(signal?: AbortSignal, timeoutMs = 45000): Promise<ApiCapabilitiesResponse> {
    const url = buildApiUrl('/api/v1/capabilities');
    const res = await requestWithTimeout(url, { signal }, timeoutMs);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(err.detail || 'Failed to query capabilities', res.status);
    }
    return res.json();
  }

  /**
   * Submits an LP, MILP, or QP formulation to POST /api/v1/solve
   */
  static async solve(
    payload: Record<string, any>, 
    signal?: AbortSignal,
    timeoutMs = 60000
  ): Promise<SolverResult> {
    const url = buildApiUrl('/api/v1/solve');
    const res = await requestWithTimeout(
      url,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
        signal,
      },
      timeoutMs
    );

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(err.detail || `Solver execution failed: HTTP ${res.status}`, res.status);
    }

    return res.json();
  }

  /**
   * Fetches problem presets from GET /api/v1/presets
   */
  static async getPresets(signal?: AbortSignal, timeoutMs = 30000): Promise<ProblemPreset[]> {
    const url = buildApiUrl('/api/v1/presets');
    const res = await requestWithTimeout(url, { signal }, timeoutMs);
    if (!res.ok) {
      throw new ApiError('Failed to load presets', res.status);
    }
    return res.json();
  }

  /**
   * Fetches Phase 8E benchmark results from GET /api/v1/benchmarks
   */
  static async getBenchmarks(signal?: AbortSignal, timeoutMs = 30000): Promise<Phase8EBenchmarkDataset> {
    const url = buildApiUrl('/api/v1/benchmarks');
    const res = await requestWithTimeout(url, { signal }, timeoutMs);
    if (!res.ok) {
      throw new ApiError('Failed to load Phase 8E benchmarks', res.status);
    }
    return res.json();
  }

  /**
   * Fetches single benchmark record from GET /api/v1/benchmarks/{id}
   */
  static async getBenchmarkById(
    id: string, 
    signal?: AbortSignal,
    timeoutMs = 30000
  ): Promise<{ system_info: any; benchmark: BenchmarkRecord }> {
    const url = buildApiUrl(`/api/v1/benchmarks/${encodeURIComponent(id)}`);
    const res = await requestWithTimeout(url, { signal }, timeoutMs);
    if (!res.ok) {
      throw new ApiError(`Failed to load benchmark record '${id}'`, res.status);
    }
    return res.json();
  }

  /**
   * Returns absolute URL for static benchmark download artifacts
   */
  static getArtifactUrl(filename: string): string {
    return buildApiUrl(`/api/v1/benchmarks/artifacts/${encodeURIComponent(filename)}`);
  }
}
