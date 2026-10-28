import React, { useState } from 'react';
import { Activity, Menu, X, ArrowUpRight } from 'lucide-react';
import type { ApiCapabilitiesResponse, ApiHealthResponse } from '../api/client';

interface NavbarProps {
  activeTab: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks';
  setActiveTab: (tab: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks') => void;
  capabilities: ApiCapabilitiesResponse | null;
  health: ApiHealthResponse | null;
  apiStatus: 'checking' | 'online' | 'offline';
  apiError?: string | null;
  onRetryConnect?: () => void;
  reducedMotion: boolean;
  setReducedMotion: (v: boolean) => void;
  onLaunchSolver: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  capabilities,
  health,
  apiStatus,
  apiError,
  onRetryConnect,
  reducedMotion,
  setReducedMotion,
  onLaunchSolver,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isCudaReady = capabilities?.cuda?.available;

  const navItems: Array<{ id: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks'; label: string }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'workspace', label: 'Solver' },
    { id: 'matrix', label: 'Matrix' },
    { id: 'convergence', label: 'Convergence' },
    { id: 'benchmarks', label: 'Performance' },
  ];

  const handleTabClick = (id: 'overview' | 'workspace' | 'matrix' | 'convergence' | 'benchmarks') => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#08080C]/90 backdrop-blur-md border-b border-[#48494D]/25 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        
        {/* Brand Logo & Editorial Wordmark */}
        <div 
          className="flex items-center space-x-3 cursor-pointer group"
          onClick={() => handleTabClick('overview')}
        >
          {/* Minimalist Monolith Symbol */}
          <div className="w-5 h-5 border border-silver/70 rotate-45 flex items-center justify-center transition-transform duration-300 group-hover:rotate-90 group-hover:border-white">
            <div className="w-1.5 h-1.5 bg-silver" />
          </div>

          <div className="flex items-baseline space-x-2">
            <span className="font-display font-bold text-sm tracking-wider text-silver group-hover:text-white transition-colors">
              VØRTEX
            </span>
            <span className="text-[10px] font-mono text-ash tracking-widest hidden sm:inline">
              SYS·2026
            </span>
          </div>
        </div>

        {/* Swiss Minimal Navigation Links */}
        <nav className="hidden md:flex items-center space-x-8 text-xs font-sans tracking-wide">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleTabClick(item.id)}
                className={`relative py-1 transition-colors duration-150 ${
                  isActive
                    ? 'text-silver font-medium'
                    : 'text-ash hover:text-silver'
                }`}
              >
                {item.label}
                {isActive && (
                  <span className="absolute bottom-0 left-0 right-0 h-[1px] bg-silver" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right Section: Telemetry Indicators & Restrained Outlined CTA */}
        <div className="hidden sm:flex items-center space-x-4">
          
          {/* Live API Online / Offline / Checking Status Tag */}
          <div 
            className="flex items-center space-x-2 text-[10px] font-mono px-2.5 py-1 border border-steel/30"
            title={
              apiStatus === 'online' && health
                ? `${health.engine} v${health.version} (${health.platform})`
                : apiStatus === 'offline' && apiError
                ? `Connection error: ${apiError}`
                : undefined
            }
          >
            {apiStatus === 'online' ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
                <span className="text-emerald-300 font-semibold tracking-wider">API ONLINE</span>
                <span className="text-steel">·</span>
                <span className="text-ash">{isCudaReady ? 'CUDA SM_86' : 'CPU HOST'}</span>
              </>
            ) : apiStatus === 'checking' ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-amber-300 tracking-wider" title="Connecting to Render backend (Render Free may take 30–50s to cold start)...">
                  CONNECTING...
                </span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
                <span className="text-rose-400 font-semibold tracking-wider">API OFFLINE</span>
                {onRetryConnect && (
                  <button
                    onClick={onRetryConnect}
                    className="ml-1 px-1.5 py-0.5 bg-steel/30 hover:bg-steel/50 text-silver hover:text-white rounded text-[9px] uppercase tracking-wider transition-colors"
                    title="Retry connecting to backend"
                  >
                    Retry
                  </button>
                )}
              </>
            )}
          </div>

          {/* Reduced Motion Toggle */}
          <button
            onClick={() => setReducedMotion(!reducedMotion)}
            title={reducedMotion ? "3D motion disabled. Click to enable." : "3D motion active. Click to reduce."}
            className={`p-1.5 text-xs transition-colors border ${
              reducedMotion 
                ? 'border-electric text-electric' 
                : 'border-steel/30 text-ash hover:text-silver hover:border-steel/60'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
          </button>

          {/* Restrained Outlined Launch Solver CTA */}
          <button
            onClick={onLaunchSolver}
            className="group px-3.5 py-1.5 border border-silver/60 hover:border-silver text-silver hover:text-white hover:bg-silver/5 font-mono text-xs tracking-wider uppercase transition-all duration-200 flex items-center space-x-1.5"
          >
            <span>Launch Solver</span>
            <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
          </button>
        </div>

        {/* Mobile Hamburger Toggle */}
        <div className="flex items-center space-x-2 md:hidden">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-1.5 text-ash hover:text-silver border border-steel/30"
          >
            {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>

      </div>

      {/* Mobile Dropdown Panel */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#08080C] border-b border-steel/30 px-4 py-4 space-y-3 font-mono text-xs">
          <div className="grid grid-cols-2 gap-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleTabClick(item.id)}
                className={`text-left p-2 border ${
                  activeTab === item.id
                    ? 'border-silver text-silver bg-charcoal'
                    : 'border-steel/30 text-ash'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="pt-2 flex items-center justify-between border-t border-steel/20">
            <div className="flex items-center space-x-2 text-[10px]">
              <span className={`w-1.5 h-1.5 rounded-full ${
                apiStatus === 'online' ? 'bg-emerald-400' : apiStatus === 'checking' ? 'bg-amber-400 animate-pulse' : 'bg-rose-500'
              }`} />
              <span className={apiStatus === 'online' ? 'text-emerald-300' : apiStatus === 'checking' ? 'text-amber-300' : 'text-rose-400'}>
                {apiStatus === 'online'
                  ? (isCudaReady ? 'API ONLINE · CUDA ACTIVE' : 'API ONLINE · CPU ACTIVE')
                  : apiStatus === 'checking'
                  ? 'CONNECTING TO BACKEND...'
                  : 'API OFFLINE'}
              </span>
              {apiStatus === 'offline' && onRetryConnect && (
                <button
                  onClick={onRetryConnect}
                  className="px-1.5 py-0.5 bg-steel/30 text-silver hover:text-white rounded text-[9px] uppercase"
                >
                  Retry
                </button>
              )}
            </div>
            <button
              onClick={() => {
                onLaunchSolver();
                setMobileMenuOpen(false);
              }}
              className="px-3 py-1 border border-silver text-silver hover:bg-silver/10 text-xs uppercase"
            >
              Launch Solver →
            </button>
          </div>
        </div>
      )}
    </header>
  );
};

