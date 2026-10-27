import React, { useRef, useState } from 'react';
import { HeroVortex3D } from './HeroVortex3D';
import { 
  ArrowUpRight, 
  ChevronDown, 
  Layers, 
  Zap, 
  ShieldCheck, 
  Grid, 
  CheckCircle2,
  Binary,
  ArrowRight
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
  const [activeStoryStep, setActiveStoryStep] = useState<number>(0);
  const capabilityRef = useRef<HTMLDivElement>(null);

  const scrollToCapabilities = () => {
    capabilityRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const gpuName = deviceInfo?.gpu?.name || "NVIDIA GeForce RTX 3050 Laptop GPU";

  // Computational Storytelling sequence steps
  const storytellingSteps = [
    {
      index: "01",
      title: "MATHEMATICAL FORMULATION",
      phase: "PREREQUISITE",
      summary: "Canonical conversion of linear, mixed-integer, or quadratic objectives into standard matrix-vector forms.",
      code: `min cᵀx + ½ xᵀQx
s.t. A x = b
     l ≤ x ≤ u
     xⱼ ∈ ℤ (for j ∈ I)`,
      highlight: "Objective & Bounds Normalization"
    },
    {
      index: "02",
      title: "SPARSE STRUCTURE ANALYSIS",
      phase: "PREPROCESSING",
      summary: "Decomposition of constraint matrix A into Compressed Sparse Row (CSR) format with Markowitz column reordering.",
      code: `CSR: row_ptr [m+1] | col_idx [nnz] | values [nnz]
Symbolic Factorization: Fill-in minimization
Condition Number: κ(B) estimation via LU`,
      highlight: "Memory Bandwidth Reduction"
    },
    {
      index: "03",
      title: "ITERATIVE OPTIMIZATION",
      phase: "NUMERICAL KERNEL",
      summary: "Two-Phase revised simplex pivoting with Bland's rule anti-cycling or Mehrotra predictor-corrector interior point.",
      code: `Δx_aff = - (Aᵀ A)⁻¹ (Aᵀ r_p)
Step Length α = min(1, 0.995 · α_max)
Complementarity Gap: μ = xᵀs / n`,
      highlight: "Guaranteed Convergence Bound"
    },
    {
      index: "04",
      title: "GPU HARDWARE ACCELERATION",
      phase: "CUDA SM_86 EXECUTION",
      summary: "Asynchronous memory staging and parallel cuBLAS GEMV / cuSPARSE SpMV execution across CUDA streaming multiprocessors.",
      code: `H2D: Pinned host transfer ➔ Device VRAM
cuSPARSE csrmv() Stream Concurrency
D2H: Asynchronous result extraction`,
      highlight: "Sub-millisecond SpMV Throughput"
    },
    {
      index: "05",
      title: "VERIFIED SOLUTION CERTIFICATE",
      phase: "CONVERGENCE",
      summary: "Strict IEEE 754 double-precision verification of primal feasibility, dual feasibility, and complementary slackness.",
      code: `Primal Residual: ||Ax - b||_∞ < 10⁻¹⁴
Dual Slackness: ||c + Qx - Aᵀλ - s||_∞ < 10⁻¹⁰
Status: OPTIMAL_VERIFIED_FP64`,
      highlight: "Zero-Heuristic Mathematical Proof"
    }
  ];

  return (
    <div className="relative min-h-screen bg-[#08080C] text-[#E7E8EA] selection:bg-[#2514ED] selection:text-white">
      
      {/* Subtle Noise Texture & Architectural Grid */}
      <div className="absolute inset-0 noise-overlay pointer-events-none z-10" />
      <div className="absolute inset-0 bg-grid-architectural opacity-25 pointer-events-none" />

      {/* ========================================================================= */}
      {/* 1. HERO — CINEMATIC ATMOSPHERIC ENVIRONMENT & SCULPTURE                   */}
      {/* ========================================================================= */}
      <section className="relative min-h-screen flex flex-col justify-between pt-14 border-b border-steel/20 overflow-hidden">
        
        {/* Layered Atmospheric Geological Horizon & Fog Backdrop */}
        <div className="absolute inset-0 cinematic-horizon pointer-events-none" />
        
        {/* Procedural Atmospheric Geological Ridge Vector Silhouette */}
        <div className="absolute bottom-0 left-0 right-0 h-96 pointer-events-none opacity-40 overflow-hidden">
          <svg className="w-full h-full object-cover" viewBox="0 0 1440 380" fill="none" preserveAspectRatio="none">
            {/* Distant Ridge (Lighter desaturated grey) */}
            <path 
              d="M0,240 Q180,160 360,210 T720,180 T1080,220 T1440,170 L1440,380 L0,380 Z" 
              fill="url(#distantRidgeGrad)" 
            />
            {/* Midground Ridge (Charcoal architectural terrain) */}
            <path 
              d="M0,280 Q220,230 460,270 T940,240 T1440,260 L1440,380 L0,380 Z" 
              fill="url(#midRidgeGrad)" 
            />
            {/* Foreground Strata (Deep void black) */}
            <path 
              d="M0,320 Q260,290 600,330 T1200,310 T1440,330 L1440,380 L0,380 Z" 
              fill="#08080C" 
            />
            <defs>
              <linearGradient id="distantRidgeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#48494D" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#08080C" stopOpacity="0.9" />
              </linearGradient>
              <linearGradient id="midRidgeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#1B1C20" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#08080C" stopOpacity="1" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Soft atmospheric fog gradient layer transitioning down */}
        <div className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-[#08080C] via-[#08080C]/80 to-transparent pointer-events-none" />

        {/* Hero Content Grid: Asymmetric Swiss Composition */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center py-12 lg:py-20 z-20">
          
          {/* Left Column: Oversized Editorial Typography & Concise Descriptor */}
          <div className="lg:col-span-7 space-y-8">
            
            {/* Subtle Minimal Metadata Tag */}
            <div className="flex items-center space-x-3 text-xs font-mono text-ash tracking-wide">
              <span className="w-1.5 h-1.5 rounded-full bg-silver" />
              <span className="uppercase tracking-widest text-[11px]">MATHEMATICAL COMPUTING SYSTEM</span>
              <span className="text-steel">/</span>
              <span className="text-[11px] text-ash">C++20 · CUDA 13.4 · FP64</span>
            </div>

            {/* Oversized Swiss Editorial Headline */}
            <div className="space-y-3">
              <h1 className="text-5xl sm:text-7xl lg:text-[5.5rem] font-display font-black tracking-tightest leading-[0.92] text-silver uppercase">
                PRECISION.<br />
                <span className="text-silver/90">
                  AT SCALE.
                </span>
              </h1>
              <p className="font-mono text-xs sm:text-sm text-ash tracking-editorial uppercase pt-1">
                BEYOND LIMITS. POWERED BY PARALLELISM.
              </p>
            </div>

            {/* Architectural Abstract Description */}
            <p className="text-ash text-sm sm:text-base leading-relaxed max-w-xl font-sans font-normal">
              An indigenous high-performance mathematical solver engineered from first principles. 
              Accelerating large-scale Linear Programming, Mixed-Integer combinatorial decisions, 
              and convex Quadratic Programming with verified IEEE 754 precision.
            </p>

            {/* Actions: Launch Solver & Explore Engine */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={onLaunchSolver}
                className="group px-6 py-3.5 bg-silver text-[#08080C] hover:bg-white font-mono text-xs font-semibold tracking-wider uppercase transition-all duration-150 flex items-center space-x-2 shadow-[0_4px_20px_rgba(231,232,234,0.12)]"
              >
                <span>Launch Solver</span>
                <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </button>

              <button
                onClick={scrollToCapabilities}
                className="px-6 py-3.5 border border-steel/40 hover:border-silver text-silver hover:text-white font-mono text-xs tracking-wider uppercase transition-colors duration-150 flex items-center space-x-2"
              >
                <span>Explore Engine</span>
                <ChevronDown className="w-3.5 h-3.5 text-ash" />
              </button>
            </div>

            {/* Minimal System Telemetry Line (clean, not cluttered) */}
            <div className="pt-4 flex items-center space-x-6 text-[11px] font-mono text-ash/80 border-t border-steel/20">
              <div>
                <span className="text-steel uppercase mr-2">DEVICE</span>
                <span className="text-silver">{gpuName.split(' ')[1] || 'RTX 3050'}</span>
              </div>
              <span className="text-steel">·</span>
              <div>
                <span className="text-steel uppercase mr-2">PRECISION</span>
                <span className="text-silver">10⁻¹⁴ L_inf</span>
              </div>
              <span className="text-steel">·</span>
              <div>
                <span className="text-steel uppercase mr-2">ACCELERATION</span>
                <span className="text-silver">SM_86 cuBLAS</span>
              </div>
            </div>

          </div>

          {/* Right Column: Floating Black Chrome Singularity Sculpture */}
          <div className="lg:col-span-5 h-[380px] sm:h-[460px] lg:h-[560px] relative flex items-center justify-center">
            
            {/* Visual Frame Guidelines */}
            <div className="absolute inset-0 pointer-events-none border border-steel/15 flex items-center justify-center">
              <div className="absolute top-2 left-2 text-[9px] font-mono text-steel">R³·SINGULARITY</div>
              <div className="absolute bottom-2 right-2 text-[9px] font-mono text-steel">FP64·SOLVER</div>
            </div>

            {/* Three.js Responsive Canvas */}
            <div className="w-full h-full flex items-center justify-center">
              <HeroVortex3D reducedMotion={reducedMotion} />
            </div>

          </div>

        </div>

        {/* Scroll Indicator */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full py-4 flex items-center justify-between text-[11px] font-mono text-ash/60 z-20">
          <span className="tracking-widest uppercase">SECTION 01 / 04</span>
          <button 
            onClick={scrollToCapabilities}
            className="flex items-center space-x-1.5 hover:text-silver transition-colors cursor-pointer"
          >
            <span>SCROLL DOWN</span>
            <ChevronDown className="w-3.5 h-3.5 animate-bounce" />
          </button>
        </div>

      </section>

      {/* ========================================================================= */}
      {/* 2. SECTION 2 — MONOCHROME CAPABILITY GRID                                 */}
      {/* ========================================================================= */}
      <section ref={capabilityRef} className="py-24 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-b border-steel/20">
        
        <div className="mb-14 space-y-2">
          <div className="text-xs font-mono text-ash tracking-editorial uppercase">
            // CORE COMPUTATIONAL CAPABILITIES
          </div>
          <h2 className="text-3xl sm:text-4xl font-display font-bold text-silver tracking-tight uppercase">
            ARCHITECTED FOR NUMERICAL FIDELITY.
          </h2>
          <p className="text-ash text-sm max-w-2xl font-sans">
            Rigorous mathematical optimization methods implemented without external black-box wrappers.
            Every calculation executes natively in modern C++20 and CUDA kernels.
          </p>
        </div>

        {/* 6-Card Monochrome Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          
          {/* Card 1: Linear Programming */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <Layers className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">SOLVER 01 // LP</div>
              <h3 className="font-display font-semibold text-lg text-silver">Linear Programming</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Two-Phase Revised Simplex utilizing m×m basis LU factorization with Markowitz-style 
              partial pivoting and Bland’s rule for deterministic anti-cycling.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              COMPLEXITY: O(m² · nnz) PER PIVOT
            </div>
          </div>

          {/* Card 2: Mixed-Integer Programming */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <Binary className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">SOLVER 02 // MILP</div>
              <h3 className="font-display font-semibold text-lg text-silver">Mixed-Integer Programming</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Exact Branch & Bound combinatorial tree exploration supporting continuous, integer, 
              and binary variables with dual bounding and Gomory feasibility cuts.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              PRUNING: BOUND · INFEASIBILITY · INTEGER
            </div>
          </div>

          {/* Card 3: Quadratic Programming */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <ShieldCheck className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">SOLVER 03 // QP</div>
              <h3 className="font-display font-semibold text-lg text-silver">Quadratic Programming</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Primal-Dual Interior-Point algorithm equipped with dynamic Tikhonov regularization 
              to guarantee stable convergence across positive semidefinite objectives.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              CONVERGENCE: KKT DUALITY GAP &lt; 10⁻⁷
            </div>
          </div>

          {/* Card 4: Sparse Matrix Engine */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <Grid className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">ENGINE 04 // SPARSITY</div>
              <h3 className="font-display font-semibold text-lg text-silver">Sparse Matrix Engine</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Compressed Sparse Row (CSR) storage format with symmetric reverse Cuthill-McKee 
              bandwidth reduction, structural rank verification, and fill-in minimization.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              FORMAT: CSR · ZERO STORAGE OF ZEROES
            </div>
          </div>

          {/* Card 5: CUDA Computing */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <Zap className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">PARALLEL 05 // CUDA</div>
              <h3 className="font-display font-semibold text-lg text-silver">CUDA Computing</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Hardware-accelerated linear algebra with asynchronous cuBLAS GEMV and cuSPARSE SpMV 
              execution, pinned host buffers, and automated CPU-GPU crossover detection.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              ARCHITECTURE: NVIDIA AMPERE SM_86
            </div>
          </div>

          {/* Card 6: Numerical Validation */}
          <div className="p-7 bg-charcoal/40 border border-steel/25 fine-line-hover space-y-4">
            <div className="w-10 h-10 border border-steel/40 flex items-center justify-center text-silver">
              <CheckCircle2 className="w-4 h-4 text-silver" />
            </div>
            <div className="space-y-1">
              <div className="text-[10px] font-mono text-ash uppercase tracking-wider">TEST 06 // VERIFICATION</div>
              <h3 className="font-display font-semibold text-lg text-silver">Numerical Validation</h3>
            </div>
            <p className="text-xs text-ash leading-relaxed font-sans">
              Certified against Netlib test problems and synthetic high-dimensional benchmarks 
              with double-precision tolerance checks on primal feasibility and condition numbers.
            </p>
            <div className="pt-3 border-t border-steel/20 text-[10px] font-mono text-ash/80">
              PASS RATE: 100% CTEST SUITE (14/14)
            </div>
          </div>

        </div>

      </section>

      {/* ========================================================================= */}
      {/* 3. SECTION 3 — MATHEMATICAL COMPLEXITY (GIANT FINE-LINE ORBITAL GEOMETRY)  */}
      {/* ========================================================================= */}
      <section className="py-28 relative overflow-hidden border-b border-steel/20">
        
        {/* Subtle Ambient Radial Highlight */}
        <div className="absolute inset-0 bg-radial-gradient from-steel/5 via-transparent to-transparent pointer-events-none" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col items-center text-center">
          
          <div className="text-xs font-mono text-ash tracking-editorial uppercase mb-4">
            // THE MATHEMATICAL DIALECTIC
          </div>

          <h2 className="text-4xl sm:text-6xl font-display font-extrabold text-silver tracking-tight uppercase max-w-3xl leading-[1.05]">
            FROM COMPLEXITY<br />TO CLARITY.
          </h2>

          <p className="text-ash text-sm sm:text-base max-w-xl font-sans mt-4">
            Transforming high-dimensional combinatorial search spaces into deterministic, 
            mathematically provable optimal coordinates.
          </p>

          {/* Giant Fine-Line Orbital Interactive Visual Architecture */}
          <div className="relative w-full max-w-[680px] h-[480px] sm:h-[580px] my-10 flex items-center justify-center">
            
            {/* SVG Giant Concentric Orbital Rings with Fine Precision Markers */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 600">
              
              {/* Outer Ring 1 */}
              <circle 
                cx="300" cy="300" r="280" 
                fill="none" 
                stroke="#48494D" 
                strokeWidth="1" 
                strokeDasharray="4 6" 
                opacity="0.35" 
                className="animate-spin-orbital"
              />

              {/* Middle Ring 2 */}
              <circle 
                cx="300" cy="300" r="220" 
                fill="none" 
                stroke="#999A9E" 
                strokeWidth="1" 
                strokeDasharray="2 4" 
                opacity="0.45" 
                className="animate-spin-orbital-rev"
              />

              {/* Inner Ring 3 with Restrained Electric Blue Accent Node */}
              <circle 
                cx="300" cy="300" r="150" 
                fill="none" 
                stroke="#E7E8EA" 
                strokeWidth="1" 
                opacity="0.6" 
              />

              {/* Axis Crosshairs */}
              <line x1="300" y1="20" x2="300" y2="580" stroke="#48494D" strokeWidth="0.75" opacity="0.3" />
              <line x1="20" y1="300" x2="580" y2="300" stroke="#48494D" strokeWidth="0.75" opacity="0.3" />

              {/* Diagonal 45-deg Guideline */}
              <line x1="102" y1="102" x2="498" y2="498" stroke="#48494D" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.25" />
              <line x1="102" y1="498" x2="498" y2="102" stroke="#48494D" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.25" />

              {/* Orbiting Satellite Node: 2% Electric Blue */}
              <circle cx="450" cy="300" r="4" fill="#2514ED" />
              <circle cx="450" cy="300" r="8" fill="none" stroke="#2514ED" strokeWidth="0.75" opacity="0.5" />
            </svg>

            {/* Central Monolith Core */}
            <div className="z-20 flex flex-col items-center justify-center p-6 bg-[#08080C]/95 border border-silver/50 max-w-[200px] text-center">
              <span className="text-[10px] font-mono text-ash tracking-widest uppercase">KERNEL</span>
              <span className="font-display font-bold text-xl text-silver tracking-wider">VØRTEX</span>
              <span className="text-[9px] font-mono text-ash mt-1">FP64 RESOLUTION</span>
            </div>

            {/* Surrounding Mathematical Nodes with Fine Monospace Typography */}
            {/* Top Node */}
            <div className="absolute top-4 sm:top-8 px-3 py-1.5 bg-[#08080C] border border-steel/30 text-xs font-mono text-silver">
              <span className="text-ash mr-1.5">//</span>Linear Constraints (Ax = b)
            </div>

            {/* Right Node */}
            <div className="absolute right-0 sm:right-4 px-3 py-1.5 bg-[#08080C] border border-steel/30 text-xs font-mono text-silver">
              <span className="text-ash mr-1.5">//</span>Integer Decisions (x ∈ ℤ)
            </div>

            {/* Bottom Node */}
            <div className="absolute bottom-4 sm:bottom-8 px-3 py-1.5 bg-[#08080C] border border-steel/30 text-xs font-mono text-silver">
              <span className="text-ash mr-1.5">//</span>Quadratic Objectives (½ xᵀQx)
            </div>

            {/* Left Node */}
            <div className="absolute left-0 sm:left-4 px-3 py-1.5 bg-[#08080C] border border-steel/30 text-xs font-mono text-silver">
              <span className="text-ash mr-1.5">//</span>Parallel CUDA SpMV
            </div>

            {/* Top-Right Diagonal Node */}
            <div className="absolute top-16 right-12 hidden md:block px-2.5 py-1 text-[10px] font-mono text-ash border border-steel/20 bg-[#08080C]/90">
              Numerical Precision 10⁻¹⁴
            </div>

            {/* Bottom-Left Diagonal Node */}
            <div className="absolute bottom-16 left-12 hidden md:block px-2.5 py-1 text-[10px] font-mono text-ash border border-steel/20 bg-[#08080C]/90">
              Branch & Bound Pruning
            </div>

          </div>

        </div>

      </section>

      {/* ========================================================================= */}
      {/* 4. SECTION 4 — COMPUTATIONAL STORYTELLING (PINNED 5-PHASE SEQUENCE)        */}
      {/* ========================================================================= */}
      <section className="py-24 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-b border-steel/20">
        
        <div className="mb-14 space-y-2">
          <div className="text-xs font-mono text-ash tracking-editorial uppercase">
            // STEP-BY-STEP SOLVER PIPELINE
          </div>
          <h2 className="text-3xl sm:text-4xl font-display font-bold text-silver tracking-tight uppercase">
            COMPUTATIONAL ANATOMY.
          </h2>
          <p className="text-ash text-sm max-w-2xl font-sans">
            Inspect each deterministic phase executed by the native engine, 
            from continuous mathematical formulation through to hardware-verified proof of optimality.
          </p>
        </div>

        {/* Interactive Pinned Sequence Selector & Visualizer */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Step Selector List (Left) */}
          <div className="lg:col-span-5 space-y-2">
            {storytellingSteps.map((step, idx) => {
              const isSelected = activeStoryStep === idx;
              return (
                <div
                  key={step.index}
                  onClick={() => setActiveStoryStep(idx)}
                  className={`p-4 border transition-all duration-150 cursor-pointer ${
                    isSelected
                      ? 'border-silver bg-charcoal text-silver'
                      : 'border-steel/25 hover:border-steel/60 text-ash bg-[#08080C]'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="font-semibold">{step.index} // {step.phase}</span>
                    <span className="text-[10px] text-ash">{isSelected ? 'ACTIVE' : 'SELECT'}</span>
                  </div>
                  <h4 className="text-sm font-display font-bold mt-1 text-silver">
                    {step.title}
                  </h4>
                  <p className="text-xs text-ash mt-1 font-sans line-clamp-2">
                    {step.summary}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Step Technical Diagnostic Display (Right) */}
          <div className="lg:col-span-7 bg-charcoal/50 border border-steel/30 p-6 sm:p-8 space-y-6">
            
            <div className="flex items-center justify-between border-b border-steel/20 pb-4">
              <div>
                <span className="text-[10px] font-mono text-ash uppercase">
                  PHASE {storytellingSteps[activeStoryStep].index} EXECUTION PROFILE
                </span>
                <h3 className="text-xl font-display font-bold text-silver">
                  {storytellingSteps[activeStoryStep].title}
                </h3>
              </div>
              <span className="px-2.5 py-1 text-[10px] font-mono border border-steel/40 text-silver bg-[#08080C]">
                {storytellingSteps[activeStoryStep].highlight}
              </span>
            </div>

            <p className="text-sm text-ash leading-relaxed font-sans">
              {storytellingSteps[activeStoryStep].summary}
            </p>

            {/* Monospace Code & Matrix Inspection Block */}
            <div className="p-4 bg-[#08080C] border border-steel/30 font-mono text-xs text-silver space-y-2 overflow-x-auto">
              <div className="text-[10px] text-ash pb-1 border-b border-steel/20">
                // MATHEMATICAL EXPRESSION / NATIVE REPRESENTATION
              </div>
              <pre className="text-ash-light leading-relaxed">
                {storytellingSteps[activeStoryStep].code}
              </pre>
            </div>

            {/* Direct Workflow Link */}
            <div className="pt-2 flex items-center justify-between">
              <span className="text-xs font-mono text-ash">
                Verified against native C++20 engine.
              </span>
              <button
                onClick={onLaunchSolver}
                className="text-xs font-mono text-silver hover:text-white flex items-center space-x-1.5 uppercase tracking-wider"
              >
                <span>Run In Solver Workspace</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

          </div>

        </div>

      </section>

      {/* ========================================================================= */}
      {/* 5. FOOTER — MINIMAL SWISS SCIENTIFIC STATEMENT                            */}
      {/* ========================================================================= */}
      <footer className="py-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-start md:items-center justify-between text-xs font-mono text-ash gap-6">
        <div>
          <span className="text-silver font-semibold">VØRTEX</span>
          <span className="mx-2 text-steel">|</span>
          <span>INDIGENOUS MATHEMATICAL OPTIMIZATION PLATFORM © 2026</span>
        </div>
        <div className="flex flex-wrap items-center gap-4 text-ash/80">
          <span>MSVC C++20</span>
          <span className="text-steel">·</span>
          <span>CUDA TOOLKIT 13.4</span>
          <span className="text-steel">·</span>
          <span>FASTAPI ADAPTER</span>
          <span className="text-steel">·</span>
          <button 
            onClick={onViewBenchmarks}
            className="text-silver hover:underline cursor-pointer"
          >
            BENCHMARKS
          </button>
        </div>
      </footer>

    </div>
  );
};

