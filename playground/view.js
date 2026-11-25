// Draws the playground with three.js and turns pointer input into robot goals.
// Usage: import { mount } from './playground/view.js'; mount(element)
import * as THREE from '../vendor/three/three.module.min.js';
import { Sim, TAXEL_COLS, TAXEL_ROWS } from './sim.js';

const url = (path) => new URL(path, import.meta.url).href;
const IDLE_SECONDS = 5; // after this long without input the robot plays by itself

async function loadFiles() {
  const get = async (path, as) => {
    const res = await fetch(url(path));
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res[as]();
  };
  const [pandaXml, gripperXml, sceneXml, meshes] = await Promise.all([
    get('./model/panda.xml', 'text'),
    get('./model/2f85.xml', 'text'),
    get('./model/scene.xml', 'text'),
    get('./model/meshes.json', 'json'),
  ]);
  return { pandaXml, gripperXml, sceneXml, meshes };
}

function melonTexture() {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#3f9a4b';
  g.fillRect(0, 0, 256, 64);
  g.fillStyle = '#1c5a2c';
  for (let i = 0; i < 10; i++) {
    g.beginPath();
    g.moveTo(i * 25.6 + 4, 0);
    for (let y = 0; y <= 64; y += 8) g.lineTo(i * 25.6 + 4 + Math.sin(y * 0.35 + i) * 2.5, y);
    for (let y = 64; y >= 0; y -= 8) g.lineTo(i * 25.6 + 13 + Math.sin(y * 0.3 + i * 2) * 2.5, y);
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  return t;
}

/** One three.js mesh per visible MuJoCo geom. */
function buildGeoms(mujoco, model, scene, skins) {
  const T = mujoco.mjtGeom;
  const materials = new Map();
  const material = (rgba, extra = {}) => {
    const key = Array.from(rgba).join() + JSON.stringify(extra);
    if (!materials.has(key)) {
      materials.set(key, new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
        roughness: 0.55, metalness: 0.05, ...extra,
      }));
    }
    return materials.get(key);
  };
  const meshGeometry = new Map();
  const items = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] === 3) continue; // collision-only shapes
    const type = model.geom_type[g];
    const size = model.geom_size.subarray(3 * g, 3 * g + 3);
    const mat = model.geom_matid[g];
    const rgba = mat >= 0 ? model.mat_rgba.subarray(4 * mat, 4 * mat + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    let geometry, mesh;
    if (type === T.mjGEOM_PLANE.value) continue; // the page itself is the table
    if (type === T.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      mesh = new THREE.Mesh(geometry, skins.includes(g) ? material(rgba, { roughness: 0.4 }).clone() : material(rgba, { roughness: 0.7 }));
    } else if (type === T.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0], 40, 24);
      geometry.rotateX(Math.PI / 2); // poles along z
      mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ map: melonTexture(), roughness: 0.45 }));
    } else if (type === T.mjGEOM_MESH.value) {
      const id = model.geom_dataid[g];
      if (!meshGeometry.has(id)) {
        const v0 = model.mesh_vertadr[id], vn = model.mesh_vertnum[id];
        const f0 = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(model.mesh_vert.subarray(3 * v0, 3 * (v0 + vn))), 3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.subarray(3 * f0, 3 * (f0 + fn))), 1));
        geometry.computeVertexNormals();
        meshGeometry.set(id, geometry);
      }
      mesh = new THREE.Mesh(meshGeometry.get(id), material(rgba));
    } else continue;
    mesh.matrixAutoUpdate = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    items.push({ g, mesh });
  }
  return items;
}

export async function mount(el, { onState = () => {} } = {}) {
  onState('loading');
  const [{ default: loadMujoco }, files] = await Promise.all([
    import('../vendor/mujoco/mujoco.js'),
    loadFiles(),
  ]);
  const mujoco = await loadMujoco();
  const sim = new Sim(mujoco, files, { seed: (Math.random() * 1e9) | 0 });

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  const canvas = renderer.domElement;
  canvas.setAttribute('aria-label', 'A simulated Franka robot arm playing with cubes and a watermelon');
  canvas.setAttribute('role', 'img');
  el.appendChild(canvas);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.05, 20);
  camera.up.set(0, 0, 1);
  const DIST = 1.75;
  const orbit = { az: -0.62, el: 0.42, dist: DIST, look: new THREE.Vector3(0.34, 0, 0.22) };
  const placeCamera = () => {
    const c = Math.cos(orbit.el);
    camera.position.set(
      orbit.look.x + orbit.dist * c * Math.cos(orbit.az),
      orbit.look.y + orbit.dist * c * Math.sin(orbit.az),
      orbit.look.z + orbit.dist * Math.sin(orbit.el),
    );
    camera.lookAt(orbit.look);
  };
  placeCamera();

  scene.add(new THREE.HemisphereLight(0xffffff, 0x9a958b, 1.25));
  const sun = new THREE.DirectionalLight(0xffffff, 1.7);
  sun.position.set(0.9, -1.4, 2.6);
  sun.castShadow = true;
  const coarse = matchMedia('(pointer: coarse)').matches; // phones and tablets
  sun.shadow.mapSize.set(coarse ? 1024 : 2048, coarse ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.5, far: 6 });
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.012;
  sun.shadow.radius = 5;
  scene.add(sun);

  // Nothing but shadows on the floor, so the robot appears to stand on the page.
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: 0.26 }));
  floor.receiveShadow = true;
  scene.add(floor);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.034, 0.04, 48),
    new THREE.MeshBasicMaterial({ color: 0xe23d5c, transparent: true, opacity: 0.9, depthWrite: false }),
  );
  ring.position.z = 0.0015;
  ring.visible = false;
  scene.add(ring);

  const skinIds = ['left', 'right'].map((side) => mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM.value, `g_${side}_skin`));
  const geoms = buildGeoms(mujoco, sim.model, scene, skinIds);
  const skins = skinIds.map((id) => geoms.find((item) => item.g === id).mesh.material);

  // ---- tactile readout: one small grid per fingertip ----
  const CELL = 9, GAP = 2, PAD_GAP = 12;
  const gridW = TAXEL_COLS * (CELL + GAP) - GAP, gridH = TAXEL_ROWS * (CELL + GAP) - GAP;
  const hud = document.createElement('canvas');
  hud.className = 'tactile';
  hud.setAttribute('aria-hidden', 'true');
  const hudScale = Math.min(2, window.devicePixelRatio);
  hud.width = (2 * gridW + PAD_GAP) * hudScale;
  hud.height = gridH * hudScale;
  hud.style.width = `${2 * gridW + PAD_GAP}px`;
  hud.style.height = `${gridH}px`;
  const label = document.createElement('div');
  label.className = 'tactile-box';
  label.append(hud, Object.assign(document.createElement('span'), { textContent: 'fingertip touch' }));
  el.appendChild(label);
  const pen = hud.getContext('2d');
  const drawTactile = () => {
    const css = getComputedStyle(el);
    const idle = css.getPropertyValue('--line'), hot = css.getPropertyValue('--melon');
    const grids = sim.tactile();
    pen.setTransform(hudScale, 0, 0, hudScale, 0, 0);
    pen.clearRect(0, 0, hud.width, hud.height);
    grids.forEach((grid, side) => {
      let total = 0;
      for (let row = 0; row < TAXEL_ROWS; row++) for (let col = 0; col < TAXEL_COLS; col++) {
        const v = 1 - Math.exp(-grid[row * TAXEL_COLS + col] / 12);
        total += v;
        const x = side * (gridW + PAD_GAP) + col * (CELL + GAP);
        const y = row * (CELL + GAP); // fingertip at the bottom, as the hand points down
        pen.globalAlpha = 1;
        pen.fillStyle = idle;
        pen.fillRect(x, y, CELL, CELL);
        pen.globalAlpha = v;
        pen.fillStyle = hot;
        pen.fillRect(x, y, CELL, CELL);
      }
      skins[side].emissive.setRGB(0.9, 0.15, 0.3).multiplyScalar(Math.min(1, total / 10));
    });
  };
  const m4 = new THREE.Matrix4();
  const syncGeoms = () => {
    const p = sim.data.geom_xpos, r = sim.data.geom_xmat;
    for (const { g, mesh } of geoms) {
      const a = 9 * g, b = 3 * g;
      m4.set(
        r[a], r[a + 1], r[a + 2], p[b],
        r[a + 3], r[a + 4], r[a + 5], p[b + 1],
        r[a + 6], r[a + 7], r[a + 8], p[b + 2],
        0, 0, 0, 1,
      );
      mesh.matrix.copy(m4);
    }
  };

  // ---- input ----
  const ray = new THREE.Raycaster();
  const table = new THREE.Plane(new THREE.Vector3(0, 0, 1), -0.02);
  const hit = new THREE.Vector3();
  let lastInput = -Infinity;
  let locked = false; // "Let it play" was pressed: only a click hands control back, not a passing mouse
  const tablePoint = (ev) => {
    const box = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - box.left) / box.width) * 2 - 1,
      -((ev.clientY - box.top) / box.height) * 2 + 1,
    );
    ray.setFromCamera(ndc, camera);
    return ray.ray.intersectPlane(table, hit);
  };
  const takeOver = () => {
    lastInput = performance.now();
    if (sim.auto.on) { sim.auto.on = false; onState('manual'); }
  };
  let orbiting = null;
  canvas.addEventListener('pointermove', (ev) => {
    if (orbiting) {
      orbit.az -= (ev.clientX - orbiting.x) * 0.006;
      orbit.el = Math.min(1.3, Math.max(0.12, orbit.el + (ev.clientY - orbiting.y) * 0.005));
      orbiting = { x: ev.clientX, y: ev.clientY };
      placeCamera();
      return;
    }
    if (ev.pointerType !== 'mouse' || locked) return; // touch steers by tapping, so the page can still scroll
    const p = tablePoint(ev);
    if (!p) return;
    takeOver();
    sim.setPointer(p.x, p.y);
  });
  canvas.addEventListener('pointerdown', (ev) => {
    if (ev.pointerType === 'mouse' && (ev.button === 2 || ev.shiftKey)) {
      orbiting = { x: ev.clientX, y: ev.clientY };
      canvas.setPointerCapture(ev.pointerId);
      return;
    }
    if (ev.button === 0) down = { x: ev.clientX, y: ev.clientY, t: ev.timeStamp };
  });
  // A click or tap, not a press: a finger that starts scrolling the page must not grab anything.
  let down = null;
  canvas.addEventListener('pointerup', (ev) => {
    orbiting = null;
    const tap = down && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) < 10 && ev.timeStamp - down.t < 600;
    down = null;
    if (!tap) return;
    const p = tablePoint(ev);
    if (!p) return;
    locked = false;
    takeOver();
    sim.pressAt(p.x, p.y);
  });
  canvas.addEventListener('pointercancel', () => { orbiting = null; down = null; });
  canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

  // ---- sizing, visibility ----
  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Keep the whole workspace in frame on narrow or short stages.
    orbit.dist = DIST * Math.max(1, 1.05 / camera.aspect);
    placeCamera();
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();
  let visible = true;
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(el);

  // ---- loop ----
  let last = performance.now();
  const selfPlay = () => {
    sim.auto.on = true;
    sim.auto.state = 'think';
    sim.auto.wait = 0.2;
    onState(locked ? 'locked' : 'auto');
  };
  const frame = (now) => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!visible || document.hidden) return;
    if (!sim.auto.on && now - lastInput > IDLE_SECONDS * 1000) selfPlay();
    sim.advance(dt);
    syncGeoms();
    drawTactile();
    ring.visible = !sim.auto.on;
    ring.position.x = sim.pointer[0];
    ring.position.y = sim.pointer[1];
    renderer.render(scene, camera);
  };
  syncGeoms();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
  onState('auto');

  return {
    sim,
    /** Hand the robot back to itself until the visitor clicks the stage. */
    auto() { locked = true; selfPlay(); },
    takeOver, // stop the self-play, as if the visitor had just touched the stage
    mess() { sim.mess(); },
    reset() { sim.reset(); locked = false; sim.auto.on = true; onState('auto'); },
    /** Where a world point lands in the viewport (client pixels). */
    toScreen(x, y, z) {
      const box = canvas.getBoundingClientRect();
      const v = new THREE.Vector3(x, y, z).project(camera);
      return [box.left + (v.x + 1) / 2 * box.width, box.top + (1 - v.y) / 2 * box.height];
    },
    /** For tests: run the simulation without waiting for real time. */
    fastForward(seconds) {
      for (let t = 0; t < seconds; t += 0.02) sim.advance(0.02);
      syncGeoms();
      drawTactile();
      renderer.render(scene, camera);
    },
  };
}
