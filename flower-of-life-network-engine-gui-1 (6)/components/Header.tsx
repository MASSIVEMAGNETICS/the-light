
import React from 'react';
import { NetworkIcon } from './icons';

export const Header: React.FC = () => {
  return (
    <header className="flex-shrink-0 p-3 px-4 border-b-2 border-cyan-500/30 bg-black/50 shadow-[0_5px_25px_rgba(0,255,255,0.1)] z-20">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <NetworkIcon className="w-8 h-8 text-cyan-400" />
          <div>
            <h1 className="text-lg font-bold tracking-wider text-cyan-300">
              Advanced Flower of Life Network Engine
            </h1>
            <p className="text-xs text-cyan-500/80 tracking-widest">
              v1.0.0-GUI-GODCORE
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
           <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(50,255,50,0.8)]"></div>
           <span className="text-green-400 text-sm font-bold">LIVE</span>
        </div>
      </div>
    </header>
  );
};
