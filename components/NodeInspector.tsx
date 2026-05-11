
import React from 'react';
import type { NodeData } from '../types';
import { XIcon, CpuIcon, GitBranchIcon, ActivityIcon, ZapIcon, BrainIcon } from './icons';

interface NodeInspectorProps {
  node: NodeData;
  onClose: () => void;
}

const StatLine: React.FC<{ icon: React.ReactNode, label: string, value: string | number }> = ({ icon, label, value }) => (
    <div className="flex items-center justify-between text-sm mb-2">
        <div className="flex items-center">
            {icon}
            <span className="ml-2 text-cyan-400/80">{label}</span>
        </div>
        <span className="font-bold text-white">{value}</span>
    </div>
);


export const NodeInspector: React.FC<NodeInspectorProps> = ({ node, onClose }) => {
  return (
    <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md border border-cyan-500/30 rounded-lg p-4 w-80 text-cyan-300 font-mono shadow-lg shadow-cyan-500/20 z-20">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-cyan-500/20">
        <h2 className="text-lg font-bold tracking-wider text-cyan-300">NODE-ID: {node.id}</h2>
        <button onClick={onClose} className="text-cyan-400 hover:text-white transition-colors">
          <XIcon className="w-6 h-6" />
        </button>
      </div>
      <div className="space-y-1">
        <StatLine icon={<CpuIcon className="w-4 h-4 text-cyan-400"/>} label="ARCHETYPE" value={node.archetype} />
        <StatLine icon={<ZapIcon className="w-4 h-4 text-yellow-400"/>} label="EVOLUTION" value={node.evolution} />
        <StatLine icon={<ActivityIcon className="w-4 h-4 text-green-400"/>} label="ENERGY" value={node.energy.toFixed(3)} />
        <StatLine icon={<ActivityIcon className="w-4 h-4 text-purple-400"/>} label="MOOD" value={node.mood.toFixed(3)} />
        <StatLine icon={<ActivityIcon className="w-4 h-4 text-orange-400"/>} label="NOVELTY" value={node.novelty.toFixed(3)} />
        <StatLine icon={<GitBranchIcon className="w-4 h-4 text-blue-400"/>} label="NEIGHBORS" value={node.neighbors} />
      </div>
      <div className="mt-4 pt-3 border-t border-cyan-500/20">
        <div className="flex items-center text-sm mb-2">
          <BrainIcon className="w-4 h-4 text-pink-400" />
          <span className="ml-2 text-cyan-400/80">CONTEXT</span>
        </div>
        <p className="text-xs font-mono text-white/90 bg-black/30 p-2 rounded break-words h-[72px] overflow-y-auto custom-scrollbar">
          {node.context}
        </p>
      </div>
    </div>
  );
};
