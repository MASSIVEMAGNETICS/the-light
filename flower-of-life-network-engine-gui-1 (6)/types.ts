
export type Archetype = 'GENERIC' | 'SENSORY' | 'MEMORY' | 'COMPUTATIONAL' | 'ADAPTED';

export interface NodeData {
  id: number;
  position: [number, number, number];
  energy: number;
  archetype: Archetype;
  evolution: number;
  mood: number;
  novelty: number;
  neighbors: number;
  context: string;
}

export interface EdgeData {
  source: number;
  target: number;
  weight: number;
}

export interface GraphData {
  nodes: NodeData[];
  edges: EdgeData[];
}

export type EngineStatus = 'initializing' | 'online' | 'stale' | 'error';