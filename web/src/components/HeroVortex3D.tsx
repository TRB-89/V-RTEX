import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface VortexSculptureProps {
  reducedMotion?: boolean;
}

const VortexSculpture: React.FC<VortexSculptureProps> = ({ reducedMotion }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef1 = useRef<THREE.Mesh>(null);
  const ringRef2 = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const mouse = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });

  // Handle subtle mouse tilt
  React.useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouse.current.targetX = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.current.targetY = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useFrame((_, delta) => {
    if (reducedMotion) return;

    // Smooth cursor responsiveness
    mouse.current.x += (mouse.current.targetX - mouse.current.x) * 0.05;
    mouse.current.y += (mouse.current.targetY - mouse.current.y) * 0.05;

    if (meshRef.current) {
      meshRef.current.rotation.x += delta * 0.25;
      meshRef.current.rotation.y += delta * 0.35;
      meshRef.current.rotation.z = mouse.current.x * 0.4;
      meshRef.current.position.x = mouse.current.x * 0.3;
      meshRef.current.position.y = mouse.current.y * 0.3;
    }

    if (ringRef1.current) {
      ringRef1.current.rotation.x -= delta * 0.15;
      ringRef1.current.rotation.y += delta * 0.2;
    }

    if (ringRef2.current) {
      ringRef2.current.rotation.y -= delta * 0.25;
      ringRef2.current.rotation.z += delta * 0.1;
    }

    if (coreRef.current) {
      coreRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Primary Mathematical Vortex Sculpture */}
      <mesh ref={meshRef}>
        <torusKnotGeometry args={[1.9, 0.45, 180, 24, 2, 3]} />
        <meshStandardMaterial
          color="#161822"
          roughness={0.18}
          metalness={0.92}
          emissive="#0C0D13"
          envMapIntensity={1.2}
        />
      </mesh>

      {/* Outer Industrial Gimbal Ring */}
      <mesh ref={ringRef1} rotation={[Math.PI / 4, 0, 0]}>
        <torusGeometry args={[3.2, 0.025, 16, 120]} />
        <meshStandardMaterial
          color="#77768F"
          roughness={0.3}
          metalness={0.8}
          wireframe={false}
        />
      </mesh>

      {/* Secondary Orbital Precision Ring with subtle Electric Blue pulse */}
      <mesh ref={ringRef2} rotation={[-Math.PI / 3, Math.PI / 6, 0]}>
        <torusGeometry args={[3.8, 0.015, 12, 100]} />
        <meshStandardMaterial
          color="#2514ED"
          roughness={0.2}
          metalness={0.9}
          emissive="#2514ED"
          emissiveIntensity={0.6}
        />
      </mesh>

      {/* Inner Coordinate Singularity Wireframe */}
      <mesh ref={coreRef}>
        <icosahedronGeometry args={[0.9, 2]} />
        <meshStandardMaterial
          color="#E7E8EF"
          roughness={0.1}
          metalness={0.9}
          wireframe={true}
          transparent={true}
          opacity={0.35}
        />
      </mesh>
    </group>
  );
};

export const HeroVortex3D: React.FC<VortexSculptureProps> = ({ reducedMotion }) => {
  return (
    <div className="w-full h-full relative pointer-events-none select-none">
      <Canvas
        camera={{ position: [0, 0, 7.5], fov: 45 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        dpr={[1, 1.5]}
      >
        <ambientLight intensity={0.6} />
        {/* Soft Key Lighting */}
        <directionalLight position={[5, 8, 5]} intensity={1.8} color="#FFFFFF" />
        {/* Cold Rim Fill */}
        <directionalLight position={[-6, -4, -4]} intensity={0.9} color="#77768F" />
        {/* Electric Blue Edge Accent */}
        <pointLight position={[3, -2, 2]} intensity={2.2} color="#2514ED" distance={10} />
        <pointLight position={[-3, 4, 1]} intensity={1.4} color="#38BDF8" distance={8} />

        <VortexSculpture reducedMotion={reducedMotion} />
      </Canvas>
    </div>
  );
};
