// FILE: services/VictorCore.ts
// PURPOSE: TypeScript port of VictorCore node logic with Archetype system and Fractal Memory.

import type { Archetype } from '../types';
import type { FractalEmbedding, FractalMemoryEntry } from './fractalTypes';

// --- Fractal Memory System ---

// Helper: Cosine similarity for embedding vectors
function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, aLen = 0, bLen = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aLen += a[i] * a[i];
    bLen += b[i] * b[i];
  }
  const denominator = Math.sqrt(aLen) * Math.sqrt(bLen);
  if (denominator === 0) return 0;
  return dot / denominator;
}

// Helper: Fractal embedding logic (placeholder)
function getFractalEmbedding(data: any): FractalEmbedding {
  const s = JSON.stringify(data);
  const vec = Array(16).fill(0).map((_, i) => {
    if (s.length === 0) return 0;
    return (s.charCodeAt(i % s.length) % 128) / 128;
  });
  return vec;
}

class FractalMemory {
  entries: FractalMemoryEntry[];
  capacity: number;

  constructor(capacity = 100) {
    this.entries = [];
    this.capacity = capacity;
  }

  add(entry: FractalMemoryEntry) {
    this.entries.push(entry);
    if (this.entries.length > this.capacity) {
      this.entries.shift(); // Keep memory bounded
    }
  }

  query(embedding: FractalEmbedding, threshold = 0.85): FractalMemoryEntry[] {
    if (!embedding) return [];
    return this.entries.filter(
      (e) => cosineSimilarity(e.embedding, embedding) > threshold
    );
  }
}

// --- Main Node Class ---

export class VictorCore {
    node_id: string;
    position: [number, number, number];
    archetype: Archetype;
    fractalMemory: FractalMemory;
    state: {
        mood: number;
        novelty: number;
        energy: number;
        last_update: number;
        directive: string | null;
        last_embedding?: FractalEmbedding;
    };
    neighbors: VictorCore[];
    evolution_level: number;
    last_mutation: number;

    constructor(node_id: string, position: [number, number, number], archetype: Archetype = "GENERIC") {
        this.node_id = node_id;
        this.position = position;
        this.archetype = archetype;
        this.fractalMemory = new FractalMemory(archetype === 'MEMORY' ? 200 : 100);
        this.state = {
            mood: 0.0,
            novelty: 0.0,
            energy: 1.0,
            last_update: Date.now() / 1000,
            directive: null,
        };
        this.neighbors = [];
        this.evolution_level = 1;
        this.last_mutation = Date.now() / 1000;
    }

    perceive(input_data: any): void {
        const embedding = getFractalEmbedding(input_data);
        this.state.last_embedding = embedding;

        this.fractalMemory.add({
          timestamp: Date.now(),
          input: input_data,
          embedding,
        });

        const similar_count = this.fractalMemory.query(embedding, 0.95).length - 1; // -1 to exclude self
        const novelty = this.fractalMemory.entries.length > 1 ? 1 - Math.min(similar_count / (this.fractalMemory.entries.length -1), 1) : 1;

        let noveltyMultiplier = 1.0;
        if (this.archetype === 'SENSORY') {
            noveltyMultiplier = 1.5;
        }
        this.state.novelty = novelty * noveltyMultiplier;
        
        this.state.mood = (0.8 * this.state.mood) + (0.2 * this.state.novelty);
        
        this.state.energy = Math.max(0.01, this.state.energy - 0.001);
    }
    
    getContextSummary(): string {
        if (!this.state.last_embedding) return "Awaiting input...";
        const mostRelevant = this.fractalMemory.query(this.state.last_embedding, 0.80);
        if (mostRelevant.length === 0) return "Context is sparse.";
        
        return mostRelevant
          .slice(-3)
          .map((e) => JSON.stringify(e.input).slice(0, 25))
          .join(" | ");
    }
    
    think(): void {
        switch(this.archetype) {
            case 'COMPUTATIONAL':
                if (this.state.novelty > 0.6 && this.state.energy > 0.1) {
                    this.evolution_level += 2;
                    this.state.energy = Math.max(0.01, this.state.energy - 0.1);
                }
                break;
            case 'SENSORY':
                if (this.state.novelty > 0.9) {
                    this.evolution_level += 0.5;
                    this.state.energy = Math.min(1.0, this.state.energy + 0.02);
                }
                break;
            default:
                if (this.state.novelty > 0.8 && this.state.energy > 0) {
                    this.evolution_level += 1;
                    this.last_mutation = Date.now() / 1000;
                    this.state.energy = Math.min(1.0, this.state.energy + 0.05);
                }
                break;
        }
        
        const prompt = `ARCHETYPE: ${this.archetype}\nMOOD: ${this.state.mood.toFixed(2)}\nNOVELTY: ${this.state.novelty.toFixed(2)}\nCONTEXT: ${this.getContextSummary()}`;
        this.state.directive = this.generateDirective(prompt);
    }
    
    generateDirective(prompt: string): string {
        if (this.state.mood > 0.9 || prompt.includes("high novelty")) return "broadcast_discovery";
        if (this.state.mood < 0.1) return "seek_input";
        if (prompt.includes("sparse")) return "explore_context";
        return "self_optimize";
    }
    
    communicate(): void {
        const pulse = this.emit_signal();
        for (const n of this.neighbors) {
            n.receive_signal(pulse);
        }
    }
    
    receive_signal(signal: any): void {
        if (signal.embedding && this.state.last_embedding) {
            const similarity = cosineSimilarity(signal.embedding, this.state.last_embedding);
            if (similarity < 0.7) {
                this.fractalMemory.add({
                    timestamp: Date.now(),
                    input: signal.input || signal,
                    embedding: signal.embedding
                });
                this.state.mood += 0.05;
            }
        }
    }
    
    emit_signal(): object {
        return {
            id: this.node_id,
            archetype: this.archetype,
            mood: this.state.mood,
            novelty: this.state.novelty,
            evolution: this.evolution_level,
            directive: this.state.directive,
            embedding: this.state.last_embedding,
            energy: this.state.energy,
            position: this.position,
            timestamp: Date.now(),
        };
    }
    
    mutate(): void {
        const roll = Math.random();
        if (roll < 0.01) {
            if (this.state.energy < 0.2 && this.archetype !== 'SENSORY') {
                this.archetype = "SENSORY";
                this.last_mutation = Date.now() / 1000;
            } else if (this.evolution_level > 20 && this.archetype !== 'COMPUTATIONAL') {
                this.archetype = 'COMPUTATIONAL';
                this.fractalMemory.capacity = 100;
                this.last_mutation = Date.now() / 1000;
            } else if (this.fractalMemory.entries.length / this.fractalMemory.capacity > 0.9 && this.archetype !== 'MEMORY') {
                this.archetype = 'MEMORY';
                this.fractalMemory.capacity = 300;
                this.last_mutation = Date.now() / 1000;
            }
        }

        if (this.state.energy < 0.1 && this.archetype !== 'SENSORY') {
            this.archetype = 'ADAPTED';
            this.state.energy += 0.2;
            this.last_mutation = Date.now() / 1000;
        }
    }
    
    step(input_data: any = { random_thought: Math.random() }): void {
        this.perceive(input_data);
        this.think();
        this.communicate();
        this.mutate();
    }

    connect(neighbor_node: VictorCore): void {
        if (!this.neighbors.includes(neighbor_node)) {
            this.neighbors.push(neighbor_node);
        }
    }

    summary(): object {
        return {
            id: this.node_id,
            archetype: this.archetype,
            mood: Math.round(this.state.mood * 1000) / 1000,
            novelty: Math.round(this.state.novelty * 1000) / 1000,
            evolution: this.evolution_level,
            energy: Math.round(this.state.energy * 1000) / 1000,
            neighbors: this.neighbors.map(n => n.node_id),
            context: this.getContextSummary()
        };
    }
}