
import React, { useState } from 'react';
import { Header } from './components/Header';
import { StatusBar } from './components/StatusBar';
import { GraphViewer } from './components/GraphViewer';
import { NodeInspector } from './components/NodeInspector';
import { useEngineData } from './hooks/useEngineData';
import type { EngineStatus, NodeData } from './types';

const App: React.FC = () => {
  const { data, status, error } = useEngineData(500);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);

  const getStatusMessage = () => {
    switch (status) {
      case 'initializing':
        return 'INITIALIZING...';
      case 'online':
        return 'SYSTEM ONLINE';
      case 'error':
        return 'CONNECTION LOST';
      case 'stale':
        return 'STALE DATA';
      default:
        return 'STANDBY';
    }
  };
  
  const handleNodeSelect = (node: NodeData) => {
    setSelectedNode(node);
  };
  
  const handleInspectorClose = () => {
    setSelectedNode(null);
  };

  return (
    <div className="h-screen w-screen bg-black text-cyan-300 font-mono flex flex-col overflow-hidden">
      <Header />
      <main className="flex-grow relative bg-black">
        {status === 'initializing' && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <p className="text-2xl animate-pulse">CONNECTING TO FoL ENGINE...</p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-black bg-opacity-80">
            <p className="text-2xl text-red-500">Error: {error}</p>
          </div>
        )}
        {data && <GraphViewer graphData={data} selectedNode={selectedNode} onNodeSelect={handleNodeSelect} onCanvasClick={handleInspectorClose} />}
        {selectedNode && <NodeInspector node={selectedNode} onClose={handleInspectorClose} />}
      </main>
      <StatusBar 
        status={status as EngineStatus}
        statusMessage={getStatusMessage()}
        nodeCount={data?.nodes.length || 0}
        edgeCount={data?.edges.length || 0}
        tickSpeed={500}
      />
    </div>
  );
};

export default App;