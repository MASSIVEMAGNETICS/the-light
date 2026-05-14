
import React from 'react';
import type { EngineStatus } from '../types';
import { CpuIcon, ZapIcon, GitBranchIcon, ActivityIcon } from './icons';

interface StatusBarProps {
  status: EngineStatus;
  statusMessage: string;
  nodeCount: number;
  edgeCount: number;
  tickSpeed: number;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  status,
  statusMessage,
  nodeCount,
  edgeCount,
  tickSpeed
}) => {
  const statusColor = {
    initializing: 'text-yellow-400',
    online: 'text-green-400',
    stale: 'text-orange-400',
    error: 'text-red-500',
  }[status];

  return (
    <footer className="flex-shrink-0 p-2 px-4 border-t-2 border-cyan-500/30 bg-gray-900/50 backdrop-blur-sm z-20">
      <div className="flex items-center justify-between text-xs tracking-wider">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <CpuIcon className="w-4 h-4" />
            <span>NODES: {nodeCount}</span>
          </div>
          <div className="flex items-center space-x-2">
            <GitBranchIcon className="w-4 h-4" />
            <span>EDGES: {edgeCount}</span>
          </div>
          <div className="flex items-center space-x-2">
            <ZapIcon className="w-4 h-4" />
            <span>TICK: {tickSpeed}ms</span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
           <ActivityIcon className={`w-4 h-4 ${statusColor}`} />
           <span className={`font-bold ${statusColor}`}>{statusMessage}</span>
        </div>
      </div>
    </footer>
  );
};
