import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { Info, RotateCcw } from 'lucide-react';
import type { ProblemFormulation, SolverResult } from '../types/solver';

interface ConvexQPSurface3DProps {
  formulation: ProblemFormulation | null;
  activeResult: SolverResult | null;
  reducedMotion?: boolean;
}

export const ConvexQPSurface3D: React.FC<ConvexQPSurface3DProps> = ({
  formulation,
  activeResult,
  reducedMotion = false,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [isRotating, setIsRotating] = useState<boolean>(!reducedMotion);

  const n = formulation?.c?.length ?? 0;
  const isTwoVariableQP = n === 2 && formulation?.solverType === 'QP' && Boolean(formulation?.Q);

  useEffect(() => {
    if (!isTwoVariableQP || !mountRef.current) return;

    const container = mountRef.current;
    const width = container.clientWidth || 600;
    const height = 400;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0C0D13); // Obsidian

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(6, 7, 9);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 2. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x2514ED, 1.8); // Electric blue key light
    dirLight1.position.set(5, 10, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xE7E8EF, 0.6); // Chrome rim light
    dirLight2.position.set(-5, 5, -5);
    scene.add(dirLight2);

    // 3. Grid Floor Helper
    const gridHelper = new THREE.GridHelper(10, 20, 0x2514ED, 0x191923);
    gridHelper.position.y = -2;
    scene.add(gridHelper);

    // 4. Paraboloid Objective Surface Geometry: f(x1, x2) = 0.5 * x^T Q x + c^T x
    const Q = formulation?.Q || [[2, 0], [0, 2]];
    const c = formulation?.c || [0, 0];
    const q11 = Q[0]?.[0] ?? 2;
    const q12 = Q[0]?.[1] ?? 0;
    const q22 = Q[1]?.[1] ?? 2;
    const c1 = c[0] ?? 0;
    const c2 = c[1] ?? 0;

    const segments = 45;
    const range = 3.5;
    const geometry = new THREE.PlaneGeometry(range * 2, range * 2, segments, segments);
    geometry.rotateX(-Math.PI / 2); // Orient horizontally

    const posAttr = geometry.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const x1 = posAttr.getX(i);
      const x2 = posAttr.getZ(i);
      // f(x1, x2) = 0.5 * (q11*x1^2 + 2*q12*x1*x2 + q22*x2^2) + c1*x1 + c2*x2
      const zVal = 0.5 * (q11 * x1 * x1 + 2 * q12 * x1 * x2 + q22 * x2 * x2) + c1 * x1 + c2 * x2;
      // Scale height to fit in viewport
      posAttr.setY(i, zVal * 0.25 - 1.5);
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0x1E1F30,
      roughness: 0.35,
      metalness: 0.8,
      wireframe: false,
      side: THREE.DoubleSide,
    });
    const surfaceMesh = new THREE.Mesh(geometry, material);
    scene.add(surfaceMesh);

    // Wireframe overlay for technical grid aesthetic
    const wireframeMat = new THREE.MeshBasicMaterial({
      color: 0x2514ED,
      wireframe: true,
      transparent: true,
      opacity: 0.3,
    });
    const wireframeMesh = new THREE.Mesh(geometry, wireframeMat);
    scene.add(wireframeMesh);

    // 5. Optimal Solution Marker Sphere
    let optMesh: THREE.Mesh | null = null;
    if (activeResult?.variable_values && activeResult.variable_values.length >= 2) {
      const optX1 = activeResult.variable_values[0];
      const optX2 = activeResult.variable_values[1];
      const optObj = activeResult.objective_value ?? 0;

      const sphereGeom = new THREE.SphereGeometry(0.18, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color: 0xE7E8EF });
      optMesh = new THREE.Mesh(sphereGeom, sphereMat);
      optMesh.position.set(optX1, optObj * 0.25 - 1.5, optX2);
      scene.add(optMesh);
    }

    // 6. Interactive Drag & Rotation Controls
    let isMouseDown = false;
    let prevMousePos = { x: 0, y: 0 };
    let rotationAngle = 0;

    const onMouseDown = (e: MouseEvent) => {
      isMouseDown = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isMouseDown) return;
      const deltaX = e.clientX - prevMousePos.x;
      prevMousePos = { x: e.clientX, y: e.clientY };

      surfaceMesh.rotation.y += deltaX * 0.01;
      wireframeMesh.rotation.y += deltaX * 0.01;
      if (optMesh) optMesh.rotation.y += deltaX * 0.01;
    };

    const onMouseUp = () => {
      isMouseDown = false;
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    // 7. Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (isRotating && !isMouseDown && !reducedMotion) {
        rotationAngle += 0.004;
        surfaceMesh.rotation.y = rotationAngle;
        wireframeMesh.rotation.y = rotationAngle;
      }
      renderer.render(scene, camera);
    };
    animate();

    // 8. Cleanup & Disposal on unmount
    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);

      geometry.dispose();
      material.dispose();
      wireframeMat.dispose();
      renderer.dispose();
      if (container.contains(dom)) {
        container.removeChild(dom);
      }
    };
  }, [isTwoVariableQP, formulation, activeResult, isRotating, reducedMotion]);

  if (!isTwoVariableQP) {
    return (
      <div className="p-8 bg-obsidian border border-graphite-border rounded-sm text-center space-y-3 font-mono text-xs">
        <Info className="w-8 h-8 text-violet-muted mx-auto" />
        <span className="font-semibold text-chrome text-sm block">
          3D Quadratic Surface Visualization Unavailable
        </span>
        <p className="text-violet-muted max-w-md mx-auto leading-relaxed">
          3D objective surfaces are mathematically defined for two-variable Quadratic Programs ($n = 2$ with active $Q$ Hessian). Switch to a 2-variable QP model (such as the Constrained Portfolio QP preset) to inspect the 3D paraboloid.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-obsidian-surface border border-graphite-border rounded-sm space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-graphite-border pb-2">
        <div>
          <span className="font-semibold text-chrome text-xs uppercase tracking-wider block">
            3D Convex QP Objective Paraboloid Surface (Three.js)
          </span>
          <span className="text-[10px] text-violet-muted">
            f(x) = ½ xᵀ Q x + cᵀ x. Drag with mouse to orbit and inspect convexity curvature.
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsRotating((r) => !r)}
            className="px-2.5 py-1 bg-graphite hover:bg-graphite-light text-chrome border border-graphite-border rounded text-[11px] flex items-center gap-1"
          >
            <RotateCcw className="w-3 h-3" />
            <span>{isRotating ? 'Pause Rotation' : 'Auto Rotate'}</span>
          </button>
        </div>
      </div>

      <div
        ref={mountRef}
        className="relative bg-obsidian rounded-sm border border-graphite-border overflow-hidden cursor-grab active:cursor-grabbing flex justify-center items-center h-[400px]"
      />

      <div className="flex items-center justify-between text-[11px] text-violet-muted pt-1">
        <div className="flex items-center space-x-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-electric inline-block" />
            Wireframe Paraboloid Curvature
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-chrome border border-electric inline-block" />
            Optimal Minimizer x*
          </span>
        </div>
        {activeResult?.objective_value !== undefined && (
          <span className="text-chrome font-semibold">
            Optimal Z* = {Number(activeResult.objective_value).toFixed(4)}
          </span>
        )}
      </div>
    </div>
  );
};
