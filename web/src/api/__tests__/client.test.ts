import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
  getApiBaseUrl, 
  buildApiUrl, 
  SolverApiClient, 
  ApiError 
} from '../client';

describe('VØRTEX API Client', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe('buildApiUrl', () => {
    it('should build clean path without duplicate /api or /api/v1', () => {
      const urlHealth = buildApiUrl('/api/v1/health');
      expect(urlHealth).toMatch(/\/api\/v1\/health$/);
      expect(urlHealth).not.toMatch(/\/api\/v1\/api\/v1/);
      expect(urlHealth).not.toMatch(/\/api\/api/);

      const urlSolve = buildApiUrl('api/v1/solve');
      expect(urlSolve).toMatch(/\/api\/v1\/solve$/);
      expect(urlSolve).not.toMatch(/\/api\/v1\/api\/v1/);

      const urlCapabilities = buildApiUrl('/api/v1/capabilities');
      expect(urlCapabilities).toMatch(/\/api\/v1\/capabilities$/);
    });

    it('should default to localhost:8000 when environment variable is unset', () => {
      const base = getApiBaseUrl();
      expect(base).toBe('http://localhost:8000');
    });
  });

  describe('SolverApiClient.getHealth', () => {
    it('should return health response model on 200 OK', async () => {
      const mockHealth = {
        status: 'online',
        engine: 'VØRTEX C++20 / CUDA Optimization Engine',
        version: '3.0.0',
        executable_exists: true,
        executable_runnable: true,
        executable_path: '/app/build/vortex_solver_runner',
        platform: 'linux',
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockHealth,
      } as Response);

      const res = await SolverApiClient.getHealth();
      expect(res.status).toBe('online');
      expect(res.executable_runnable).toBe(true);
      expect(res.version).toBe('3.0.0');
    });

    it('should throw ApiError with status on failure', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ detail: 'Backend container restarting' }),
      } as Response);

      await expect(SolverApiClient.getHealth()).rejects.toThrow(ApiError);
    });
  });

  describe('SolverApiClient.getCapabilities', () => {
    it('should query capabilities truthfully', async () => {
      const mockCaps = {
        engine: 'VØRTEX C++20 / CUDA Optimization Engine',
        version: '3.0.0',
        cxx_standard: 'C++20',
        precision: 'IEEE 754 Double Precision (FP64)',
        solvers: ['LP', 'MILP', 'QP'],
        cuda: {
          available: false,
          device_name: null,
          compute_capability: null,
          multiprocessors: null,
          total_vram_bytes: null,
        },
        max_constraints: 10000,
        max_variables: 10000,
      };

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => mockCaps,
      } as Response);

      const res = await SolverApiClient.getCapabilities();
      expect(res.solvers).toContain('LP');
      expect(res.solvers).toContain('MILP');
      expect(res.solvers).toContain('QP');
      expect(res.cuda.available).toBe(false);
    });
  });

  describe('SolverApiClient.solve', () => {
    it('should dispatch POST /api/v1/solve with exact formulation payload', async () => {
      const mockSolution = {
        success: true,
        solver_type: 'LP',
        status: 'OPTIMAL',
        message: 'Optimal solution found within tolerance.',
        objective_value: 36.0,
        variable_values: [2.0, 6.0],
        iterations: 2,
        wall_time_ms: 1.25,
      };

      let capturedUrl = '';
      let capturedBody: any = null;

      globalThis.fetch = vi.fn().mockImplementation(async (url: string, init: RequestInit) => {
        capturedUrl = url;
        capturedBody = JSON.parse(init.body as string);
        return {
          ok: true,
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          json: async () => mockSolution,
        } as Response;
      });

      const payload = {
        solver_type: 'LP' as const,
        sense: 'MAXIMIZE' as const,
        backend: 'AUTO' as const,
        c: [3.0, 5.0],
        A: [
          [1.0, 0.0],
          [0.0, 2.0],
          [3.0, 2.0],
        ],
        b: [4.0, 12.0, 18.0],
        constraint_types: ['LEQ', 'LEQ', 'LEQ'] as const,
      };

      const res = await SolverApiClient.solve(payload);
      expect(capturedUrl).toMatch(/\/api\/v1\/solve$/);
      expect(capturedBody.solver_type).toBe('LP');
      expect(capturedBody.c).toEqual([3.0, 5.0]);
      expect(res.success).toBe(true);
      expect(res.objective_value).toBe(36.0);
      expect(res.variable_values).toEqual([2.0, 6.0]);
    });

    it('should surface detailed server validation errors', async () => {
      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          detail: "Constraint matrix A row 1 length (1) does not match variable dimension n (2).",
        }),
      } as Response);

      await expect(
        SolverApiClient.solve({
          solver_type: 'LP',
          c: [1, 2],
          A: [[1]],
          b: [2],
        })
      ).rejects.toThrow(/does not match/);
    });
  });
});
