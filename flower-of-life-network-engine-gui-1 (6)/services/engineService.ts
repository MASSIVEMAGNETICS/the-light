
import type { GraphData, NodeData, EdgeData, Archetype } from '../types';
import { VictorCore } from './VictorCore';

const NODE_COUNT = 150;
const K_NEAREST = 3;
const SPHERE_RADIUS = 10;

let mesh_nodes: VictorCore[] = [];
let edges: EdgeData[] = [];

const initializeSimulation = () => {
  // 1. Create VictorCore instances with random archetypes
  const tempNodes: { core: VictorCore; pos: [number, number, number] }[] = [];
  const phi = Math.PI * (3 - Math.sqrt(5));
  const archetypes: Archetype[] = ['SENSORY', 'MEMORY', 'COMPUTATIONAL'];

  for (let i = 0; i < NODE_COUNT; i++) {
    const y = 1 - (i / (NODE_COUNT - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;
    const position: [number, number, number] = [x * SPHERE_RADIUS, y * SPHERE_RADIUS, z * SPHERE_RADIUS];
    
    const initialArchetype = Math.random() < 0.2
      ? archetypes[Math.floor(Math.random() * archetypes.length)]
      : 'GENERIC';

    tempNodes.push({ core: new VictorCore(String(i), position, initialArchetype), pos: position });
  }
  mesh_nodes = tempNodes.map(n => n.core);

  // 2. Connect them and create edges for visualization
  const newEdges: EdgeData[] = [];
  for (let i = 0; i < mesh_nodes.length; i++) {
    const source = tempNodes[i];
    const distances = tempNodes
      .map((target, j) => {
        if (i === j) return { dist: Infinity, index: j };
        const dx = source.pos[0] - target.pos[0];
        const dy = source.pos[1] - target.pos[1];
        const dz = source.pos[2] - target.pos[2];
        return { dist: Math.sqrt(dx * dx + dy * dy + dz * dz), index: j };
      })
      .sort((a, b) => a.dist - b.dist);

    for (let k = 0; k < K_NEAREST; k++) {
      const targetIdx = distances[k].index;
      mesh_nodes[i].connect(mesh_nodes[targetIdx]);
      if (!newEdges.some(e => (e.source === targetIdx && e.target === i) || (e.source === i && e.target === targetIdx))) {
        newEdges.push({
          source: i,
          target: targetIdx,
          weight: Math.random() * 0.5 + 0.1,
        });
      }
    }
  }
  edges = newEdges;
};

const runSimulationStep = (): GraphData => {
    // Run one simulation step for each node
    mesh_nodes.forEach(node => {
        node.step();
    });

    const graphNodes: NodeData[] = mesh_nodes.map((node, i) => {
        const summary = node.summary() as {
            id: string;
            archetype: Archetype;
            mood: number;
            novelty: number;
            evolution: number;
            energy: number;
            neighbors: string[];
            context: string;
        };
        
        return {
            id: i,
            position: node.position,
            energy: summary.energy,
            archetype: summary.archetype,
            evolution: summary.evolution,
            mood: summary.mood,
            novelty: summary.novelty,
            neighbors: summary.neighbors.length,
            context: summary.context,
        };
    });

    const updatedEdges = edges.map(edge => ({
        ...edge,
        weight: Math.max(0.1, Math.min(1, edge.weight + (Math.random() - 0.5) * 0.02)),
    }));

    return { nodes: graphNodes, edges: updatedEdges };
};


export const fetchGraphData = async (): Promise<GraphData> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      if (mesh_nodes.length === 0) {
        initializeSimulation();
      }
      
      const updatedGraphData = runSimulationStep();
      resolve(updatedGraphData);

    }, 50 + Math.random() * 100); // Simulate network latency
  });
};