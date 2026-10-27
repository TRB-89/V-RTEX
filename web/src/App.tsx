import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { OptimizationWorkspace } from './components/OptimizationWorkspace';
import { MatrixInspector } from './components/MatrixInspector';
import { ConvergenceLab } from './components/ConvergenceLab';
import { GpuPerformanceLab } from './components/GpuPerformanceLab';
import { ResultsReportModal } from './components/ResultsReportModal';
import type { 
  SolverResult, 
  ProblemPreset, 
  Phase8EBenchmarkDataset,
  ProblemFormulation 
} from './types/solver';
import { 
  SolverApiClient, 
  type ApiCapabilitiesResponse, 
  type ApiHealthResponse 
} from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks'>('overview');
  const [capabilities, setCapabilities] = useState<ApiCapabilitiesResponse | null>(null);
  const [health, setHealth] = useState<ApiHealthResponse | null>(null);
  const [presets, setPresets] = useState<ProblemPreset[]>([]);
  const [benchmarkData, setBenchmarkData] = useState<Phase8EBenchmarkDataset | null>(null);
  const [activeResult, setActiveResult] = useState<SolverResult | null>(null);
  const [problemFormulation, setProblemFormulation] = useState<ProblemFormulation | null>(null);
  const [reducedMotion, setReducedMotion] = useState<boolean>(false);
  const [isReportOpen, setIsReportOpen] = useState<boolean>(false);

  // Fetch telemetry and datasets on mount
  useEffect(() => {
    // 1. Health
    SolverApiClient.getHealth()
      .then(setHealth)
      .catch((err) => console.warn('Could not query API health:', err));

    // 2. Capabilities
    SolverApiClient.getCapabilities()
      .then(setCapabilities)
      .catch((err) => console.warn('Could not query engine capabilities:', err));

    // 3. Presets
    SolverApiClient.getPresets()
      .then(setPresets)
      .catch((err) => console.warn('Could not load problem presets:', err));

    // 4. Phase 8E Benchmarks
    SolverApiClient.getBenchmarks()
      .then(setBenchmarkData)
      .catch((err) => console.warn('Could not load Phase 8E benchmarks:', err));
  }, []);

  // Adapt capabilities to DeviceTelemetryResponse format for LandingPage
  const deviceTelemetry = capabilities ? {
    cuda_available: capabilities.cuda.available,
    gpu: capabilities.cuda.available ? {
      device_id: 0,
      name: capabilities.cuda.device_name || "NVIDIA GeForce RTX 3050 Laptop GPU",
      compute_capability: capabilities.cuda.compute_capability || "8.6",
      multiprocessor_count: capabilities.cuda.multiprocessors || 16,
      total_memory_bytes: capabilities.cuda.total_vram_bytes || 4294443008,
      free_memory_bytes: 3466277684,
      max_threads_per_block: 1024,
      warp_size: 32,
    } : null,
    supported_solvers: ["LP", "MILP", "QP"],
    supported_backends: ["CPU", "GPU", "AUTO"],
    system_info: {
      solver_name: capabilities.engine,
      version: capabilities.version,
      cxx_standard: capabilities.cxx_standard,
      precision: capabilities.precision,
    }
  } : null;

  return (
    <div className="min-h-screen bg-obsidian text-chrome selection:bg-electric selection:text-white font-sans">
      
      {/* Top Fixed Premium Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        capabilities={capabilities}
        health={health}
        reducedMotion={reducedMotion}
        setReducedMotion={setReducedMotion}
        onLaunchSolver={() => setActiveTab('workspace')}
      />

      {/* Main Content Area Based on Active Tab */}
      <main>
        {activeTab === 'overview' && (
          <LandingPage
            deviceInfo={deviceTelemetry}
            reducedMotion={reducedMotion}
            onLaunchSolver={() => setActiveTab('workspace')}
            onViewBenchmarks={() => setActiveTab('benchmarks')}
          />
        )}

        {/* Workspace: preserved across tab switches */}
        <div className={activeTab === 'workspace' ? 'pt-14' : 'hidden'}>
          <OptimizationWorkspace
            presets={presets}
            activeResult={activeResult}
            setActiveResult={setActiveResult}
            onInspectMatrix={() => setActiveTab('matrix')}
            onViewConvergence={() => setActiveTab('convergence')}
            onFormulationChange={setProblemFormulation}
          />
        </div>

        {activeTab === 'matrix' && (
          <div className="pt-14">
            <MatrixInspector 
              activeResult={activeResult} 
              formulation={problemFormulation}
            />
          </div>
        )}

        {activeTab === 'convergence' && (
          <div className="pt-14">
            <ConvergenceLab 
              activeResult={activeResult} 
              formulation={problemFormulation}
              onLaunchSolver={() => setActiveTab('workspace')}
              reducedMotion={reducedMotion}
            />
          </div>
        )}

        {activeTab === 'benchmarks' && (
          <div className="pt-14">
            <GpuPerformanceLab benchmarkData={benchmarkData} />
          </div>
        )}
      </main>

      {/* Results and Export Modal */}
      <ResultsReportModal
        result={activeResult}
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
      />

    </div>
  );
};

export default App;
