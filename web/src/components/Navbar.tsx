import React, { useState } from 'react';
import { Activity, Menu, X, ArrowUpRight } from 'lucide-react';
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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isCudaReady = capabilities?.cuda?.available;
  const isOnline = health?.status === 'online' || health?.status === 'healthy';

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
          
          {/* Architecture / CUDA Status Tag */}
          <div className="flex items-center space-x-2 text-[10px] font-mono text-ash px-2 py-0.5 border border-steel/30">
            <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-silver' : 'bg-ash/50'}`} />
            <span>{isCudaReady ? 'CUDA SM_86' : 'CPU HOST'}</span>
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
            <span className="text-[10px] text-ash">
              {isCudaReady ? 'CUDA SM_86 ACTIVE' : 'CPU HOST ACTIVE'}
            </span>
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

