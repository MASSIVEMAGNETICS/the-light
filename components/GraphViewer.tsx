
import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';
import type { GraphData, NodeData, EdgeData, Archetype } from '../types';

interface GraphViewerProps {
  graphData: GraphData;
  selectedNode: NodeData | null;
  onNodeSelect: (node: NodeData) => void;
  onCanvasClick: () => void;
}

const getArchetypeColor = (archetype: Archetype, energy: number): THREE.Color => {
  const color = new THREE.Color();
  switch (archetype) {
    case 'SENSORY':
      color.setHSL(0.75, 0.9, 0.6); // Purple
      break;
    case 'MEMORY':
      color.setHSL(0.3, 0.8, 0.6); // Green
      break;
    case 'COMPUTATIONAL':
      color.setHSL(0.15, 0.9, 0.6); // Yellow/Orange
      break;
    case 'ADAPTED':
      color.setHSL(0.0, 0.8, 0.7); // Red
      break;
    case 'GENERIC':
    default:
      color.setHSL(0.5, 0.8, 0.6); // Cyan
      break;
  }
  // Modulate brightness by energy
  return color.multiplyScalar(energy * 0.5 + 0.5);
};


const Node: React.FC<{ node: NodeData; isSelected: boolean; onSelect: () => void; }> = ({ node, isSelected, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null!);
  const baseColor = useMemo(() => getArchetypeColor(node.archetype, node.energy), [node.archetype, node.energy]);
  const scale = useMemo(() => 0.15 + Math.min(node.evolution / 500, 0.2), [node.evolution]);
  const emissiveIntensity = isSelected ? 2.5 : 0.8;

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.lerp(new THREE.Vector3(...node.position), 0.1);
      (meshRef.current.material as THREE.MeshStandardMaterial).color.lerp(baseColor, 0.1);
      (meshRef.current.material as THREE.MeshStandardMaterial).emissive.lerp(baseColor, 0.1);
      (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = THREE.MathUtils.lerp(
        (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity,
        emissiveIntensity,
        0.1
      );
      meshRef.current.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.1);
    }
  });

  return (
    <mesh 
      ref={meshRef} 
      position={node.position}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        color={baseColor}
        emissive={baseColor}
        emissiveIntensity={emissiveIntensity}
        roughness={0.4}
        metalness={0.2}
      />
    </mesh>
  );
};

const Edge: React.FC<{ edge: EdgeData; nodes: NodeData[] }> = ({ edge, nodes }) => {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  if (!sourceNode || !targetNode) return null;

  return (
    <Line
      points={[sourceNode.position, targetNode.position]}
      color="cyan"
      lineWidth={2}
      transparent
      opacity={Math.max(0.1, edge.weight * 0.5)}
    />
  );
};


export const GraphViewer: React.FC<GraphViewerProps> = ({ graphData, selectedNode, onNodeSelect, onCanvasClick }) => {
  const { nodes, edges } = graphData;

  return (
    <Canvas 
      camera={{ position: [0, 0, 25], fov: 50 }} 
      style={{ background: 'transparent' }}
      onPointerMissed={onCanvasClick}
    >
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 0, 15]} color="cyan" intensity={150} distance={40} />
      <pointLight position={[10, 10, -10]} color="magenta" intensity={50} distance={30} />
      <fog attach="fog" args={['#000000', 20, 40]} />
      
      {nodes.map(node => (
        <Node 
          key={node.id} 
          node={node}
          isSelected={selectedNode?.id === node.id}
          onSelect={() => onNodeSelect(node)}
        />
      ))}
      {edges.map((edge, index) => (
        <Edge key={`${edge.source}-${edge.target}-${index}`} edge={edge} nodes={nodes} />
      ))}

      <OrbitControls 
        enableDamping
        dampingFactor={0.05}
        autoRotate
        autoRotateSpeed={0.3}
        minDistance={5}
        maxDistance={50}
      />
    </Canvas>
  );
};