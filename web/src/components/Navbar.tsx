import React from 'react';
import { Cpu, Zap, Activity, Eye, Sliders, BarChart3, Terminal } from 'lucide-react';
import type { ApiCapabilitiesResponse, ApiHealthResponse } from '../api/client';

interface NavbarProps {
  activeTab: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks';
  setActiveTab: (tab: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks') => void;
  capabilities: ApiCapabilitiesResponse | null;
  health: ApiHealthResponse | null;
  reducedMotion: boolean;
  setReducedMotion: (v: boolean) => void;
  onLaunchSolver: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  capabilities,
  health,
  reducedMotion,
  setReducedMotion,
  onLaunchSolver,
}) => {
  const isCudaReady = capabilities?.cuda?.available;
  const isOnline = health?.status === 'online';

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-obsidian/90 backdrop-blur-md border-b border-graphite-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div className="flex items-center space-x-4 cursor-pointer" onClick={() => setActiveTab('overview')}>
          <div className="w-8 h-8 rounded-sm bg-obsidian-surface border border-electric/40 flex items-center justify-center relative group">
            <div className="w-3 h-3 border border-chrome rotate-45 group-hover:border-electric transition-colors" />
            <div className="absolute inset-0 bg-electric/10 blur-[6px] rounded-sm pointer-events-none" />
          </div>
          <div className="flex flex-col">
            <span className="font-display font-bold text-lg tracking-wider text-chrome flex items-center gap-1.5">
              VØRTEX
              <span className="text-[10px] font-mono tracking-normal text-violet-muted border border-graphite-border px-1.5 py-0.5 rounded-sm">
                v3.0.0
              </span>
            </span>
            <span className="text-[9px] font-mono text-violet-muted tracking-widest uppercase">
              C++20 · CUDA 13.4 · FP64
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1 font-mono text-xs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1.5 rounded-sm transition-all flex items-center gap-1.5 ${
              activeTab === 'overview'
                ? 'text-chrome bg-graphite-light border border-graphite-muted'
                : 'text-violet-muted hover:text-chrome hover:bg-graphite/40'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Overview
          </button>

          <button
            onClick={() => setActiveTab('workspace')}
            className={`px-3 py-1.5 rounded-sm transition-all flex items-center gap-1.5 ${
              activeTab === 'workspace'
                ? 'text-chrome bg-graphite-light border border-graphite-muted'
                : 'text-violet-muted hover:text-chrome hover:bg-graphite/40'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            Solver API (Phase 9B)
          </button>

          <button
            onClick={() => setActiveTab('matrix')}
            className={`px-3 py-1.5 rounded-sm transition-all flex items-center gap-1.5 ${
              activeTab === 'matrix'
                ? 'text-chrome bg-graphite-light border border-graphite-muted'
                : 'text-violet-muted hover:text-chrome hover:bg-graphite/40'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Matrix Inspector
          </button>

          <button
            onClick={() => setActiveTab('convergence')}
            className={`px-3 py-1.5 rounded-sm transition-all flex items-center gap-1.5 ${
              activeTab === 'convergence'
                ? 'text-chrome bg-graphite-light border border-graphite-muted'
                : 'text-violet-muted hover:text-chrome hover:bg-graphite/40'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Convergence Lab
          </button>

          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`px-3 py-1.5 rounded-sm transition-all flex items-center gap-1.5 ${
              activeTab === 'benchmarks'
                ? 'text-chrome bg-graphite-light border border-graphite-muted'
                : 'text-violet-muted hover:text-chrome hover:bg-graphite/40'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Performance Lab
          </button>
        </nav>

        {/* Right Section: Real Hardware & Health Status Pills & CTA */}
        <div className="flex items-center space-x-3">
          
          {/* Reduced Motion Toggle */}
          <button
            onClick={() => setReducedMotion(!reducedMotion)}
            title={reducedMotion ? "Enable 3D Motion" : "Reduce 3D Motion (Performance Mode)"}
            className={`px-2 py-1 text-[11px] font-mono border rounded-sm transition-colors flex items-center gap-1 ${
              reducedMotion
                ? 'border-electric text-electric bg-electric/10'
                : 'border-graphite-border text-violet-muted hover:text-chrome'
            }`}
          >
            <Activity className="w-3 h-3" />
            <span className="hidden sm:inline">{reducedMotion ? 'Static 3D' : 'Dynamic 3D'}</span>
          </button>

          {/* API Health Pill */}
          <div className="hidden sm:flex items-center space-x-1.5 px-2 py-1 bg-obsidian-surface border border-graphite-border rounded-sm font-mono text-[11px]">
            {isOnline ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-emerald-400 font-medium">API v1</span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                <span className="text-rose-400 font-medium">API Offline</span>
              </>
            )}
          </div>

          {/* Real-time Hardware Telemetry Pill */}
          <div className="hidden lg:flex items-center space-x-2 px-2.5 py-1 bg-obsidian-surface border border-graphite-border rounded-sm font-mono text-[11px]">
            {isCudaReady ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="text-chrome font-medium">CUDA SM_86</span>
                <span className="text-violet-muted truncate max-w-[120px]">
                  {capabilities?.cuda?.device_name?.split(' ')[1] || 'RTX 3050'}
                </span>
              </>
            ) : (
              <>
                <Cpu className="w-3.5 h-3.5 text-violet-muted" />
                <span className="text-violet-muted">CPU Host Active</span>
              </>
            )}
          </div>

          {/* Launch Solver Action Button */}
          <button
            onClick={onLaunchSolver}
            className="px-3.5 py-1.5 bg-electric hover:bg-electric-glow active:bg-electric-deep text-white font-mono text-xs font-medium tracking-wider rounded-sm transition-all duration-150 flex items-center gap-1.5 shadow-[0_0_15px_rgba(37,20,237,0.35)]"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>SOLVE</span>
          </button>
        </div>

      </div>
    </header>
  );
};
