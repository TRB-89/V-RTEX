import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface VortexSculptureProps {
  reducedMotion?: boolean;
}

// Responsive Camera Adjuster to ensure the sculpture is NEVER clipped at any breakpoint
const ResponsiveCameraController: React.FC = () => {
  const { camera, size } = useThree();

  useEffect(() => {
    const aspect = size.width / size.height;
    if (camera instanceof THREE.PerspectiveCamera) {
      if (size.width < 640) {
        // Mobile viewport: expand distance and adjust FOV
        camera.position.set(0, 0, 9.5);
        camera.fov = 48;
      } else if (size.width < 1024) {
        // Tablet viewport
        camera.position.set(0, 0, 8.2);
        camera.fov = 42;
      } else {
        // Desktop / Large screen
        camera.position.set(0, 0, 7.6);
        camera.fov = 38;
      }
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    }
  }, [camera, size]);

  return null;
};

const GeometricSingularity: React.FC<VortexSculptureProps> = ({ reducedMotion }) => {
  const groupRef = useRef<THREE.Group>(null);
  const primaryMeshRef = useRef<THREE.Mesh>(null);
  const outerRingRef = useRef<THREE.Mesh>(null);
  const innerRingRef = useRef<THREE.Mesh>(null);
  const facetedCoreRef = useRef<THREE.Mesh>(null);
  const accentFilamentRef = useRef<THREE.Mesh>(null);

  const mouse = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });

  // Subtle, controlled cursor responsiveness
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // Normalize to [-1, 1] with damping
      mouse.current.targetX = (e.clientX / window.innerWidth - 0.5) * 1.5;
      mouse.current.targetY = -(e.clientY / window.innerHeight - 0.5) * 1.5;
    };
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useFrame((_, delta) => {
    if (reducedMotion) return;

    // Smooth cursor interpolation
    mouse.current.x += (mouse.current.targetX - mouse.current.x) * 0.04;
    mouse.current.y += (mouse.current.targetY - mouse.current.y) * 0.04;

    if (groupRef.current) {
      groupRef.current.rotation.y = mouse.current.x * 0.35;
      groupRef.current.rotation.x = -mouse.current.y * 0.25;
    }

    // Steady, hypnotic architectural rotation
    if (primaryMeshRef.current) {
      primaryMeshRef.current.rotation.x += delta * 0.18;
      primaryMeshRef.current.rotation.y += delta * 0.22;
    }

    if (outerRingRef.current) {
      outerRingRef.current.rotation.z -= delta * 0.08;
      outerRingRef.current.rotation.x += delta * 0.05;
    }

    if (innerRingRef.current) {
      innerRingRef.current.rotation.y -= delta * 0.14;
      innerRingRef.current.rotation.z += delta * 0.1;
    }

    if (facetedCoreRef.current) {
      facetedCoreRef.current.rotation.y += delta * 0.3;
      facetedCoreRef.current.rotation.x -= delta * 0.15;
    }

    if (accentFilamentRef.current) {
      accentFilamentRef.current.rotation.x -= delta * 0.12;
      accentFilamentRef.current.rotation.y += delta * 0.16;
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* 1. Primary Architectural Monolith Knot — Black Chrome with Crisp Specular Highlights */}
      <mesh ref={primaryMeshRef}>
        {/* Faceted cross-section (14 tubular segments) creates crisp architectural bevels */}
        <torusKnotGeometry args={[1.65, 0.42, 160, 14, 2, 3]} />
        <meshStandardMaterial
          color="#18191F"
          roughness={0.16}
          metalness={0.96}
          flatShading={false}
          emissive="#08080C"
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* 2. Faceted Chrome Interlocking Outer Monolith Ring */}
      <mesh ref={outerRingRef} rotation={[Math.PI / 3.5, 0, Math.PI / 6]}>
        <torusGeometry args={[2.7, 0.045, 6, 96]} />
        <meshStandardMaterial
          color="#D8D9DE"
          roughness={0.18}
          metalness={0.95}
          flatShading={true}
        />
      </mesh>

      {/* 3. Steel Architectural Orbit Ring with Beveled Hex Profile */}
      <mesh ref={innerRingRef} rotation={[-Math.PI / 4, Math.PI / 4, 0]}>
        <torusGeometry args={[3.2, 0.022, 6, 80]} />
        <meshStandardMaterial
          color="#5E6066"
          roughness={0.28}
          metalness={0.9}
        />
      </mesh>

      {/* 4. Central Geometric Singularity — Faceted Polyhedron with Chrome Edges */}
      <mesh ref={facetedCoreRef}>
        <icosahedronGeometry args={[0.72, 1]} />
        <meshStandardMaterial
          color="#E7E8EA"
          roughness={0.12}
          metalness={0.98}
          wireframe={true}
          transparent={true}
          opacity={0.45}
        />
      </mesh>

      {/* 5. Restrained 2% Electric Blue Filament Precision Guide */}
      <mesh ref={accentFilamentRef} rotation={[0, Math.PI / 3, Math.PI / 4]}>
        <torusGeometry args={[3.55, 0.012, 4, 120]} />
        <meshStandardMaterial
          color="#2514ED"
          roughness={0.2}
          metalness={0.9}
          emissive="#2514ED"
          emissiveIntensity={0.45}
        />
      </mesh>
    </group>
  );
};

export const HeroVortex3D: React.FC<VortexSculptureProps> = ({ reducedMotion }) => {
  const [hasWebGL, setHasWebGL] = useState(true);

  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) setHasWebGL(false);
    } catch {
      setHasWebGL(false);
    }
  }, []);

  if (!hasWebGL) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="w-64 h-64 rounded-full border border-steel/30 flex items-center justify-center text-ash text-xs font-mono">
          [GPU SCULPTURE STANDBY]
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative select-none" style={{ minHeight: '340px' }}>
      <Canvas
        camera={{ position: [0, 0, 7.6], fov: 38 }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance',
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.15,
        }}
        dpr={[1, 1.75]}
      >
        <ResponsiveCameraController />

        {/* Studio Lighting Rig tailored for Black Chrome Reflections */}
        {/* 1. Neutral Ambient Lighting ensures dark metallic surfaces never vanish into the background */}
        <ambientLight intensity={1.05} color="#D0D2D8" />

        {/* 2. Key Daylight Highlight (Top Front-Right) */}
        <directionalLight position={[6, 9, 7]} intensity={2.6} color="#FFFFFF" />

        {/* 3. Opposite Silver Rim Light (Back-Left) for Silhouette Separation */}
        <directionalLight position={[-7, -4, -6]} intensity={2.2} color="#E7E8EA" />

        {/* 4. Bottom Ground Fill (Soft Architectural Bounce) */}
        <directionalLight position={[0, -6, 4]} intensity={0.9} color="#48494D" />

        {/* 5. Subtle High-Contrast Lateral Edge Rim */}
        <pointLight position={[-4, 5, 2]} intensity={1.2} color="#F4F5F7" distance={15} />

        {/* 6. Restrained Electric Blue Accent Spot */}
        <pointLight position={[4, -4, 2]} intensity={0.9} color="#2514ED" distance={10} />

        <GeometricSingularity reducedMotion={reducedMotion} />
      </Canvas>
    </div>
  );
};

