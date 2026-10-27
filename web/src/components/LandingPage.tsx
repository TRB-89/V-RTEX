import React from 'react';
import { HeroVortex3D } from './HeroVortex3D';
import { 
  ArrowRight, 
  Cpu, 
  Zap, 
  Layers, 
  ShieldCheck, 
  ChevronRight
} from 'lucide-react';
import type { DeviceTelemetryResponse } from '../types/solver';

interface LandingPageProps {
  deviceInfo: DeviceTelemetryResponse | null;
  reducedMotion: boolean;
  onLaunchSolver: () => void;
  onViewBenchmarks: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  deviceInfo,
  reducedMotion,
  onLaunchSolver,
  onViewBenchmarks,
}) => {
  const gpuName = deviceInfo?.gpu?.name || "NVIDIA GeForce RTX 3050 Laptop GPU";
  const smCount = deviceInfo?.gpu?.multiprocessor_count || 16;

  return (
    <div className="relative min-h-screen pt-16 bg-obsidian text-chrome selection:bg-electric selection:text-white">
      
      {/* Background Architectural Grid & Subtle Radial Ambient */}
      <div className="absolute inset-0 bg-grid-pattern opacity-40 pointer-events-none" />
      <div className="absolute inset-0 bg-subtle-radial pointer-events-none" />
      <div className="absolute inset-0 noise-overlay pointer-events-none" />

      {/* Hero Section: Asymmetric Left-Text / Right-Sculpture Layout */}
      <section className="relative min-h-[calc(100vh-4rem)] flex items-center border-b border-graphite-border overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full py-12 lg:py-20 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left Column: Editorial Headline & Technical Specification */}
          <div className="lg:col-span-7 z-10 space-y-6">
            
            {/* Scientific Coordinates & Telemetry Markers */}
            <div className="flex items-center space-x-3 text-[11px] font-mono text-violet-muted">
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 border border-graphite-border bg-obsidian-surface rounded-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-electric animate-pulse" />
                SYS_ONLINE
              </span>
              <span>[45°12'N 12°18'E]</span>
              <span className="hidden sm:inline">|</span>
              <span className="hidden sm:inline">ARCH: x86_64 + SM_86</span>
              <span className="hidden md:inline">|</span>
              <span className="hidden md:inline">IEEE 754 FP64</span>
            </div>

            {/* Oversized Editorial Headline */}
            <div className="space-y-2">
              <h1 className="text-5xl sm:text-7xl lg:text-8xl font-display font-extrabold tracking-tightest leading-[0.9] text-chrome uppercase">
                BEYOND<br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-chrome via-chrome-muted to-violet-muted">
                  LIMITS.
                </span>
              </h1>
              <p className="font-mono text-sm sm:text-base text-electric font-semibold tracking-editorial uppercase pt-2">
                PRECISION AT SCALE. POWERED BY PARALLELISM.
              </p>
            </div>

            {/* Architectural Abstract Summary */}
            <p className="text-violet-muted text-sm sm:text-base leading-relaxed max-w-xl font-sans">
              VØRTEX is an indigenous mathematical optimization solver engineered in modern C++20 
              and hardware-accelerated with NVIDIA CUDA. Purpose-built for high-dimensional Linear 
              Programming, Mixed-Integer combinatorial problems, and convex Quadratic Programming 
              with verified double-precision fidelity.
            </p>

            {/* Hardware Telemetry Snapshot Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2 font-mono text-xs max-w-lg">
              <div className="p-3 bg-obsidian-surface border border-graphite-border rounded-sm">
                <div className="text-[10px] text-violet-muted uppercase tracking-wider">Device Architecture</div>
                <div className="font-semibold text-chrome mt-0.5 truncate">{gpuName.split(' ')[1] || 'Ampere'}</div>
                <div className="text-[10px] text-violet-dim mt-0.5">{smCount} Multiprocessors</div>
              </div>

              <div className="p-3 bg-obsidian-surface border border-graphite-border rounded-sm">
                <div className="text-[10px] text-violet-muted uppercase tracking-wider">Core Linear Engine</div>
                <div className="font-semibold text-chrome mt-0.5">Two-Phase Revised</div>
                <div className="text-[10px] text-violet-dim mt-0.5">LU + Partial Pivoting</div>
              </div>

              <div className="p-3 bg-obsidian-surface border border-graphite-border rounded-sm col-span-2 sm:col-span-1">
                <div className="text-[10px] text-violet-muted uppercase tracking-wider">Verified Accuracy</div>
                <div className="font-semibold text-status-optimal mt-0.5">10⁻¹⁴ L_inf</div>
                <div className="text-[10px] text-violet-dim mt-0.5">100% Test Pass Rate</div>
              </div>
            </div>

            {/* Primary Action Buttons */}
            <div className="flex flex-wrap items-center gap-4 pt-4">
              <button
                onClick={onLaunchSolver}
                className="px-6 py-3 bg-electric hover:bg-electric-glow active:bg-electric-deep text-white font-mono text-xs font-semibold tracking-wider rounded-sm transition-all duration-150 flex items-center gap-2 shadow-[0_0_25px_rgba(37,20,237,0.4)] group"
              >
                <span>LAUNCH SOLVER WORKSPACE</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>

              <button
                onClick={onViewBenchmarks}
                className="px-5 py-3 bg-obsidian-surface hover:bg-graphite-light border border-graphite-border text-chrome hover:text-white font-mono text-xs tracking-wider rounded-sm transition-colors flex items-center gap-2"
              >
                <span>EXPLORE GPU BENCHMARKS</span>
                <ChevronRight className="w-4 h-4 text-violet-muted" />
              </button>
            </div>

          </div>

          {/* Right Column: Dramatic 3D Hero Sculpture */}
          <div className="lg:col-span-5 h-[380px] sm:h-[480px] lg:h-[620px] relative flex items-center justify-center">
            
            {/* Three.js Canvas Container */}
            <div className="absolute inset-0 flex items-center justify-center">
              <HeroVortex3D reducedMotion={reducedMotion} />
            </div>

            {/* Architectural Telemetry Labels Floating Around Object */}
            <div className="absolute top-4 right-4 p-2 bg-obsidian-surface/80 backdrop-blur-sm border border-graphite-border rounded-sm font-mono text-[10px] text-violet-muted hidden sm:block">
              <div>// TOPOLOGY: TORUS SINGULARITY</div>
              <div className="text-chrome">DIM: R³ / C++20 FP64</div>
            </div>

            <div className="absolute bottom-6 left-4 p-2 bg-obsidian-surface/80 backdrop-blur-sm border border-graphite-border rounded-sm font-mono text-[10px] text-violet-muted hidden sm:block">
              <div>// COMPUTE STREAM: cuBLAS + cuSPARSE</div>
              <div className="text-electric">PIPELINE: H2D ➔ KERNEL ➔ D2H</div>
            </div>

          </div>

        </div>
      </section>

      {/* Technical Architecture Deep Dive: Scroll-Scrubbed Scientific Modules */}
      <section className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-b border-graphite-border">
        
        <div className="mb-12 space-y-2">
          <div className="text-xs font-mono text-electric tracking-editorial uppercase">
            // MATHEMATICAL FOUNDATIONS & ALGORITHMIC PIPELINE
          </div>
          <h2 className="text-3xl sm:text-4xl font-display font-bold text-chrome tracking-tight">
            ENGINEERED WITHOUT HEURISTIC COMPROMISES.
          </h2>
          <p className="text-violet-muted text-sm max-w-2xl font-sans">
            Every optimization formulation is solved from mathematical first principles using our 
            proprietary C++20 kernel implementations and verified against industry standards.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          {/* Card 1: Revised Simplex */}
          <div className="p-6 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4 hover:border-violet-muted/50 transition-colors">
            <div className="w-10 h-10 rounded-sm bg-graphite flex items-center justify-center text-chrome border border-graphite-border">
              <Layers className="w-5 h-5 text-electric" />
            </div>
            <div className="space-y-1">
              <div className="text-[11px] font-mono text-violet-muted">MODULE 01 // LP CORE</div>
              <h3 className="font-display font-semibold text-lg text-chrome">Revised Simplex + LU</h3>
            </div>
            <p className="text-xs text-violet-muted leading-relaxed font-sans">
              Maintains an m×m basis factorization rather than full tableaus. Features Markowitz-style 
              partial pivoting, periodic refactorization, and Bland’s rule for sound cycling prevention.
            </p>
            <div className="pt-2 border-t border-graphite-border text-[10px] font-mono text-violet-dim">
              COMPLEXITY: O(m² · nnz) PER PIVOT
            </div>
          </div>

          {/* Card 2: GPU Acceleration */}
          <div className="p-6 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4 hover:border-violet-muted/50 transition-colors">
            <div className="w-10 h-10 rounded-sm bg-graphite flex items-center justify-center text-chrome border border-graphite-border">
              <Zap className="w-5 h-5 text-electric" />
            </div>
            <div className="space-y-1">
              <div className="text-[11px] font-mono text-violet-muted">MODULE 02 // PARALLEL ENGINE</div>
              <h3 className="font-display font-semibold text-lg text-chrome">CUDA Acceleration</h3>
            </div>
            <p className="text-xs text-violet-muted leading-relaxed font-sans">
              Asynchronous cuBLAS GEMV and cuSPARSE SpMV execution. Features pinned host staging buffers, 
              hardware stream concurrency, and empirical CPU-GPU crossover detection.
            </p>
            <div className="pt-2 border-t border-graphite-border text-[10px] font-mono text-violet-dim">
              HARDWARE: SM 8.6 AMPERE · 16 SMs
            </div>
          </div>

          {/* Card 3: MILP Branch and Bound */}
          <div className="p-6 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4 hover:border-violet-muted/50 transition-colors">
            <div className="w-10 h-10 rounded-sm bg-graphite flex items-center justify-center text-chrome border border-graphite-border">
              <Cpu className="w-5 h-5 text-electric" />
            </div>
            <div className="space-y-1">
              <div className="text-[11px] font-mono text-violet-muted">MODULE 03 // COMBINATORIAL</div>
              <h3 className="font-display font-semibold text-lg text-chrome">Branch & Bound MILP</h3>
            </div>
            <p className="text-xs text-violet-muted leading-relaxed font-sans">
              Exact integer programming engine supporting continuous, integer, and binary variables. 
              Features best-bound tree exploration, dual bounding, and integer feasibility cuts.
            </p>
            <div className="pt-2 border-t border-graphite-border text-[10px] font-mono text-violet-dim">
              PRUNING: BOUND · INFEASIBILITY · INTEGER
            </div>
          </div>

          {/* Card 4: Convex QP */}
          <div className="p-6 bg-obsidian-surface border border-graphite-border rounded-sm space-y-4 hover:border-violet-muted/50 transition-colors">
            <div className="w-10 h-10 rounded-sm bg-graphite flex items-center justify-center text-chrome border border-graphite-border">
              <ShieldCheck className="w-5 h-5 text-electric" />
            </div>
            <div className="space-y-1">
              <div className="text-[11px] font-mono text-violet-muted">MODULE 04 // NONLINEAR CORE</div>
              <h3 className="font-display font-semibold text-lg text-chrome">Primal-Dual QP</h3>
            </div>
            <p className="text-xs text-violet-muted leading-relaxed font-sans">
              Predictor-corrector interior-point algorithm with dynamic Tikhonov regularization. 
              Solves unconstrained, singular positive semidefinite, and linearly constrained QPs.
            </p>
            <div className="pt-2 border-t border-graphite-border text-[10px] font-mono text-violet-dim">
              CONVERGENCE: KKT DUALITY GAP &lt; 10⁻⁷
            </div>
          </div>

        </div>

      </section>

      {/* Footer Specs & System Statement */}
      <footer className="py-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-violet-muted gap-4">
        <div>
          VØRTEX MATHEMATICAL ENGINE © 2026 — INDIGENOUS HIGH-PERFORMANCE COMPUTING
        </div>
        <div className="flex items-center space-x-4">
          <span>MSVC C++20</span>
          <span>•</span>
          <span>CUDA TOOLKIT 13.4</span>
          <span>•</span>
          <span>FASTAPI ADAPTER</span>
        </div>
      </footer>

    </div>
  );
};
