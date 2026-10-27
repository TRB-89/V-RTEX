export type SolverType = 'LP' | 'MILP' | 'QP';
export type ObjectiveSense = 'MAXIMIZE' | 'MINIMIZE';
export type SolverBackend = 'CPU' | 'GPU' | 'AUTO';
export type ConstraintType = 'LEQ' | 'GEQ' | 'EQ';
export type VariableType = 'CONTINUOUS' | 'INTEGER' | 'BINARY';

export interface VariableBounds {
  lower: number;
  upper: number;
}

export interface DeviceDiagnostics {
  requested_backend: string;
  executed_backend: string;
  gpu_iterations: number;
  gpu_gemv_calls: number;
  gpu_spmv_calls: number;
  h2d_transfers: number;
  d2h_transfers: number;
  fallback_reason: string;
}

export interface MatrixInfo {
  rows?: number;
  cols?: number;
  q_dim?: number;
  constraints_m?: number;
  constraints_n?: number;
  nnz: number;
  density: number;
}

export interface SolverResult {
  success: boolean;
  solver_type: SolverType;
  status: string;
  message: string;
  objective_value?: number;
  variable_values?: number[];
  slack_values?: number[];
  surplus_values?: number[];
  dual_values?: number[];
  reduced_costs?: number[];
  basis_indices?: number[];
  iterations?: number;
  phase1_iterations?: number;
  phase2_iterations?: number;
  max_primal_residual?: number;
  max_dual_residual?: number;
  basis_condition?: number;
  wall_time_ms?: number;
  
  // QP specific
  lambda?: number[];
  slack?: number[];
  primal_residual?: number;
  dual_residual?: number;
  duality_gap?: number;
  
  // MILP specific
  best_bound?: number;
  absolute_gap?: number;
  relative_gap?: number;
  nodes_explored?: number;
  nodes_pruned_infeasible?: number;
  nodes_pruned_bound?: number;
  nodes_pruned_integer?: number;

  device_diagnostics?: DeviceDiagnostics;
  matrix_info?: MatrixInfo;
  iteration_history?: IterationRecord[];
  request_meta?: {
    solver_type: string;
    requested_backend: string;
    dimension_n: number;
    dimension_m: number;
    sense: string;
  };
}

export interface IterationRecord {
  iteration: number;
  objective?: number;
  primal_residual?: number;
  dual_residual?: number;
  duality_gap?: number;
}

export interface ProblemFormulation {
  solverType: SolverType;
  sense: ObjectiveSense;
  backend: SolverBackend;
  c: number[];
  A: number[][];
  b: number[];
  constraint_types?: ('LEQ' | 'GEQ' | 'EQ')[];
  variable_types?: ('CONTINUOUS' | 'INTEGER' | 'BINARY')[];
  bounds?: { lower: number; upper: number }[];
  Q?: number[][];
  variableNames?: string[];
  constraintNames?: string[];
}

export interface ProblemPreset {
  id: string;
  name: string;
  description: string;
  solver_type: SolverType;
  sense: ObjectiveSense;
  backend: SolverBackend;
  c: number[];
  A: number[][];
  b: number[];
  constraint_types?: ConstraintType[];
  variable_types?: VariableType[];
  bounds?: VariableBounds[];
  Q?: number[][];
  options?: Record<string, any>;
}

export interface GpuDeviceInfo {
  device_id: number;
  name: string;
  compute_capability: string;
  multiprocessor_count: number;
  total_memory_bytes: number;
  free_memory_bytes: number;
  max_threads_per_block: number;
  warp_size: number;
}

export interface DeviceTelemetryResponse {
  cuda_available: boolean;
  gpu: GpuDeviceInfo | null;
  supported_solvers: string[];
  supported_backends: string[];
  system_info: {
    solver_name: string;
    version: string;
    cxx_standard: string;
    precision: string;
  };
}

export interface MemoryFootprint {
  host_bytes: number;
  gpu_bytes: number;
  host_kib: number;
  gpu_kib: number;
  host_mib: number;
  gpu_mib: number;
  storage_format: string;
}

export interface BenchmarkProvenance {
  source_file: string;
  file_format: string;
  generation_time: string;
  record_count: number;
  checksum_sha256: string;
  harness: string;
  timing_methodology: string;
  notes: string;
}

export interface BenchmarkRecord {
  id: string;
  category: string;
  name: string;
  rows: number;
  cols: number;
  nnz: number;
  density: number;
  transpose?: boolean;
  cpu_timing_ms: {
    median: number;
    min?: number;
    max?: number;
    mean?: number;
    stddev?: number;
    iqr?: number;
  };
  gpu_kernel_timing_ms?: {
    median: number;
  };
  gpu_e2e_timing_ms: {
    median: number;
    min?: number;
    max?: number;
    mean?: number;
    stddev?: number;
    iqr?: number;
  };
  speedup_compute?: number;
  speedup_e2e: number;
  speedup_label: string;
  is_slowdown: boolean;
  transfer_overhead_ms?: number;
  max_absolute_error?: number;
  scaled_relative_error?: number;
  notes?: string;
  is_microbenchmark: boolean;
  memory: MemoryFootprint;
}

export interface Phase8EBenchmarkDataset {
  provenance: BenchmarkProvenance;
  system_info: {
    gpu_name: string;
    compute_capability: string;
    multiprocessor_count: number;
    total_vram_bytes: number;
    available_vram_bytes: number;
    core_clock_khz: number;
    memory_clock_khz: number;
    memory_bus_width_bits: number;
    cpu_name: string;
    compiler: string;
    cuda_runtime_version: number;
    build_type: string;
    repetition_count: number;
    warmup_count: number;
  };
  benchmarks: BenchmarkRecord[];
}
