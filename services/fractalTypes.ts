// FILE: services/fractalTypes.ts
// PURPOSE: Defines types for the fractal memory system used by VictorCore.

export type FractalEmbedding = number[];

export interface FractalMemoryEntry {
  timestamp: number;
  input: any;
  embedding: FractalEmbedding;
}
