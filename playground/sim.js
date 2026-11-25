// Physics and control for the playground. No rendering in here, so the same
// file runs in the browser and under node (see tools/sim-test.mjs).

const HOME = [0, -0.35, 0, -2.25, 0, 1.9, 0.785];
const SHOULDER = [0, 0, 0.333];
const HOVER_Z = 0.17;
const CARRY_Z = 0.17;
const CONTROL_EVERY = 5; // physics steps per control tick (10 ms)
export const TAXEL_COLS = 4, TAXEL_ROWS = 7;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/** Solve A x = b for a small dense system (in place, partial pivoting). */
function solve(A, b, n) {
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[p * n + c])) p = r;
    if (p !== c) {
      for (let k = 0; k < n; k++) [A[c * n + k], A[p * n + k]] = [A[p * n + k], A[c * n + k]];
      [b[c], b[p]] = [b[p], b[c]];
    }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      if (!f) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    b[r] = s / A[r * n + r];
  }
  return b;
}

export function loadModel(mujoco, files) {
  const FS = mujoco.FS;
  for (const dir of ['/model', '/model/assets']) {
    try { FS.mkdir(dir); } catch { /* already there */ }
  }
  FS.writeFile('/model/panda.xml', files.pandaXml);
  FS.writeFile('/model/2f85.xml', files.gripperXml);
  FS.writeFile('/model/scene.xml', files.sceneXml);
  for (const [name, text] of Object.entries(files.meshes)) FS.writeFile(`/model/assets/${name}`, text);
  return mujoco.MjModel.from_xml_path('/model/scene.xml');
}

export class Sim {
  constructor(mujoco, files, { seed = 1 } = {}) {
    this.mj = mujoco;
    this.model = loadModel(mujoco, files);
    this.data = new mujoco.MjData(this.model);
    this.ik = new mujoco.MjData(this.model);
    const m = this.model;
    this.nv = m.nv;
    this.jacp = new mujoco.DoubleBuffer(3 * this.nv);
    this.jacr = new mujoco.DoubleBuffer(3 * this.nv);
    const id = (kind, name) => mujoco.mj_name2id(m, mujoco.mjtObj[kind].value, name);
    this.site = id('mjOBJ_SITE', 'g_pinch');
    // Fingertips: [left, right], each with its body and the two collision boxes of its pad.
    this.pads = ['left', 'right'].map((side) => ({
      body: id('mjOBJ_BODY', `g_${side}_pad`),
      geoms: [id('mjOBJ_GEOM', `g_${side}_pad1`), id('mjOBJ_GEOM', `g_${side}_pad2`)],
      taxels: new Float32Array(TAXEL_COLS * TAXEL_ROWS),
    }));
    this.force = new mujoco.DoubleBuffer(6);
    this.seed = seed;

    // Everything with a free joint is a toy.
    this.toys = [];
    for (let b = 1; b < m.nbody; b++) {
      const j = m.body_jntadr[b];
      if (j < 0 || m.jnt_type[j] !== mujoco.mjtJoint.mjJNT_FREE.value) continue;
      const g = m.body_geomadr[b];
      const sphere = m.geom_type[g] === mujoco.mjtGeom.mjGEOM_SPHERE.value;
      this.toys.push({
        body: b,
        q: m.jnt_qposadr[j],
        v: m.jnt_dofadr[j],
        half: sphere ? m.geom_size[3 * g] : m.geom_size[3 * g + 2],
        sphere,
        lostFor: 0,
      });
    }
    this.reset();
  }

  rand() { // mulberry32
    let t = (this.seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  reset() {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    for (let i = 0; i < 7; i++) { d.qpos[i] = HOME[i]; d.ctrl[i] = HOME[i]; }
    d.ctrl[7] = 0; // Robotiq: 0 is open, 255 is closed
    mj.mj_forward(m, d);
    this.qdes = HOME.slice();
    const p = this.sitePos();
    const R = d.site_xmat;
    const s = 9 * this.site;
    this.yaw0 = Math.atan2(R[s + 3], R[s]) - Math.atan2(p[1], p[0]);
    this.target = { pos: [p[0], p[1], HOVER_Z], yaw: this.naturalYaw(p[0], p[1]), grip: 1 };
    this.pointer = [p[0], p[1]];
    this.setMode('hover');
    this.held = null;
    this.pending = 0;
    this.carryZ = CARRY_Z;
    this.steps = 0;
    this.auto = { on: true, state: 'think', wait: 0.8, toy: null };
    this.handSpeed = 0;
    this.lastSite = p;
  }

  // ---- reading the world ----
  sitePos() {
    const x = this.data.site_xpos, s = 3 * this.site;
    return [x[s], x[s + 1], x[s + 2]];
  }
  toyPos(t) {
    const q = this.data.qpos;
    return [q[t.q], q[t.q + 1], q[t.q + 2]];
  }
  toyYaw(t) {
    const q = this.data.qpos;
    const w = q[t.q + 3], x = q[t.q + 4], y = q[t.q + 5], z = q[t.q + 6];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }
  /** Gap between the two fingertip pads, in metres. */
  fingerWidth() {
    const x = this.data.xpos, a = 3 * this.pads[0].body, b = 3 * this.pads[1].body;
    return Math.hypot(x[a] - x[b], x[a + 1] - x[b + 1], x[a + 2] - x[b + 2]) - 0.0132;
  }

  /**
   * Tactile fingertips: spread each contact's normal force over a small grid on
   * the pad it touches. Returns [left, right] grids, row-major. The values grow
   * with contact force but are not calibrated.
   */
  tactile() {
    const { mj, model: m, data: d } = this;
    for (const pad of this.pads) pad.taxels.fill(0);
    const contacts = d.contact;
    const n = contacts.size();
    for (let i = 0; i < n; i++) {
      const c = contacts.get(i);
      const pad = this.pads.find((p) => p.geoms.includes(c.geom1) || p.geoms.includes(c.geom2));
      if (pad) {
        mj.mj_contactForce(m, d, i, this.force);
        const f = Math.abs(this.force.GetView()[0]);
        // contact point in the pad's own frame: x across the pad, z along the finger
        const o = 3 * pad.body, r = 9 * pad.body, P = c.pos;
        const dx = P[0] - d.xpos[o], dy = P[1] - d.xpos[o + 1], dz = P[2] - d.xpos[o + 2];
        const u = (d.xmat[r] * dx + d.xmat[r + 3] * dy + d.xmat[r + 6] * dz + 0.011) / 0.022;
        const w = (d.xmat[r + 2] * dx + d.xmat[r + 5] * dy + d.xmat[r + 8] * dz) / 0.0375;
        for (let row = 0; row < TAXEL_ROWS; row++) for (let col = 0; col < TAXEL_COLS; col++) {
          const du = (col + 0.5) / TAXEL_COLS - u, dw = ((row + 0.5) / TAXEL_ROWS - w) * 1.7;
          pad.taxels[row * TAXEL_COLS + col] += f * Math.exp(-(du * du + dw * dw) / 0.06);
        }
      }
      c.delete?.();
    }
    contacts.delete();
    return this.pads.map((p) => p.taxels);
  }
  naturalYaw(x, y) { return this.yaw0 + Math.atan2(y, x); }

  /** Topmost toy near (x, y), ignoring `skip`. */
  toyAt(x, y, radius, skip = null) {
    let best = null, bestZ = -1;
    for (const t of this.toys) {
      if (t === skip) continue;
      const p = this.toyPos(t);
      if (Math.hypot(p[0] - x, p[1] - y) > radius) continue;
      if (p[2] > bestZ) { best = t; bestZ = p[2]; }
    }
    return best;
  }

  /** Keep a requested point inside what the arm can reach. */
  clampReach(x, y, z) {
    let r = Math.hypot(x, y);
    const a = clamp(Math.atan2(y, x), -2.2, 2.2);
    r = clamp(r, 0.3, 0.74);
    z = clamp(z, 0.012, 0.6);
    const dz = z - SHOULDER[2];
    const max = Math.sqrt(Math.max(0.05, 0.8 * 0.8 - dz * dz));
    r = Math.min(r, max);
    return [r * Math.cos(a), r * Math.sin(a), z];
  }

  // ---- what the page calls ----
  setPointer(x, y) {
    const p = this.clampReach(x, y, 0.1);
    this.pointer = [p[0], p[1]];
  }
  /** One button: grab what is under the hand, or put down what it holds. */
  /** Go to (x, y), then grab or put down once the hand has arrived. */
  pressAt(x, y) {
    this.setPointer(x, y);
    this.pending = 2.5;
  }
  press() {
    this.pending = 0;
    if (this.mode === 'hover') this.setMode('descend');
    else if (this.mode === 'carry') {
      if (this.handSpeed > 0.45) { this.target.grip = 1; this.setMode('open'); } // let it fly
      else this.setMode('lower');
    }
  }
  nudgeHeight(dz) { this.carryZ = clamp(this.carryZ + dz, 0.06, 0.45); }
  /** Give every toy a shove. */
  mess() {
    const v = this.data.qvel;
    for (const t of this.toys) {
      v[t.v] += (this.rand() - 0.5) * 1.6;
      v[t.v + 1] += (this.rand() - 0.5) * 1.6;
      v[t.v + 2] += 1.2 + this.rand() * 1.2;
      for (let k = 3; k < 6; k++) v[t.v + k] += (this.rand() - 0.5) * 14;
    }
  }
  setMode(mode) { this.mode = mode; this.modeTime = 0; this.entered = true; }

  // ---- hand state machine, runs every control tick ----
  think(dt) {
    const T = this.target;
    const site = this.sitePos();
    this.modeTime += dt;
    if (this.pending > 0) {
      this.pending -= dt;
      const ready = this.mode === 'hover' || this.mode === 'carry';
      if (ready && Math.hypot(site[0] - this.pointer[0], site[1] - this.pointer[1]) < 0.015) this.press();
    }
    let [gx, gy] = this.pointer;
    let gz = HOVER_Z, yaw = this.naturalYaw(gx, gy);
    const entered = this.entered;
    this.entered = false;

    switch (this.mode) {
      case 'hover':
        T.grip = 1;
        break;
      case 'descend': {
        if (entered) { // first tick: lock on to whatever is under the hand
          this.pick = this.toyAt(gx, gy, 0.05);
          this.pickXY = [gx, gy];
          this.pickYaw = yaw;
          if (this.pick) {
            const p = this.toyPos(this.pick);
            this.pickXY = [p[0], p[1]];
            if (!this.pick.sphere) { // square the fingers up with the cube
              let best = yaw, bestD = 9;
              for (let n = 0; n < 4; n++) {
                const c = this.toyYaw(this.pick) + n * Math.PI / 2;
                const dd = Math.abs(wrap(c - yaw));
                if (dd < bestD) { bestD = dd; best = yaw + wrap(c - yaw); }
              }
              this.pickYaw = best;
            }
          }
        }
        [gx, gy] = this.pickXY;
        yaw = this.pickYaw;
        const aligned = Math.hypot(site[0] - gx, site[1] - gy) < 0.012;
        const low = this.pick ? Math.max(0.014, this.toyPos(this.pick)[2] - 0.004) : 0.02;
        gz = aligned || this.modeTime > 1.2 ? low : HOVER_Z * 0.6;
        if (Math.abs(site[2] - low) < 0.006 || this.modeTime > 2.4) this.setMode('close');
        break;
      }
      case 'close':
        [gx, gy] = this.pickXY; yaw = this.pickYaw; gz = T.pos[2];
        T.grip = 0;
        if (this.modeTime > 0.55) {
          this.held = null;
          if (this.pick && this.fingerWidth() > 0.02) {
            const p = this.toyPos(this.pick);
            if (Math.hypot(p[0] - site[0], p[1] - site[1], p[2] - site[2]) < 0.04) this.held = this.pick;
          }
          this.carryZ = CARRY_Z;
          this.setMode(this.held ? 'carry' : 'open');
        }
        break;
      case 'carry': {
        T.grip = 0;
        gz = this.carryZ;
        const p = this.toyPos(this.held);
        if (Math.hypot(p[0] - site[0], p[1] - site[1], p[2] - site[2]) > 0.07) { // dropped it
          this.held = null;
          this.setMode('open');
        }
        break;
      }
      case 'lower': {
        T.grip = 0;
        if (entered) {
          const h = this.toyPos(this.held);
          const under = this.toyAt(site[0], site[1], 0.04, this.held);
          let support = 0;
          this.placeXY = [T.pos[0], T.pos[1]];
          if (under) {
            const u = this.toyPos(under);
            support = u[2] + under.half;
            if (!under.sphere) this.placeXY = [u[0] + (site[0] - h[0]), u[1] + (site[1] - h[1])];
          }
          this.placeZ = support + this.held.half + (site[2] - h[2]) + 0.003;
          this.placeYaw = T.yaw;
        }
        [gx, gy] = this.placeXY; yaw = this.placeYaw; gz = this.placeZ;
        if ((Math.abs(site[2] - gz) < 0.005 && Math.hypot(site[0] - gx, site[1] - gy) < 0.008) || this.modeTime > 2) {
          this.setMode('open');
        }
        break;
      }
      case 'open':
        gx = T.pos[0]; gy = T.pos[1]; gz = T.pos[2]; yaw = T.yaw;
        T.grip = 1;
        if (this.modeTime > 0.3) { this.held = null; this.setMode('rise'); }
        break;
      case 'rise':
        gx = T.pos[0]; gy = T.pos[1]; yaw = T.yaw;
        if (site[2] > HOVER_Z - 0.05 || this.modeTime > 1) this.setMode('hover');
        break;
    }

    // Move the commanded pose toward the goal at a bounded speed.
    const goal = this.clampReach(gx, gy, gz);
    const maxXY = 0.9 * dt, maxZ = 0.55 * dt;
    const dx = goal[0] - T.pos[0], dy = goal[1] - T.pos[1];
    const dxy = Math.hypot(dx, dy);
    const k = dxy > maxXY ? maxXY / dxy : 1;
    T.pos[0] += dx * k;
    T.pos[1] += dy * k;
    T.pos[2] += clamp(goal[2] - T.pos[2], -maxZ, maxZ);
    T.yaw += clamp(wrap(yaw - T.yaw), -3 * dt, 3 * dt);

    const v = Math.hypot(site[0] - this.lastSite[0], site[1] - this.lastSite[1], site[2] - this.lastSite[2]) / dt;
    this.handSpeed += (v - this.handSpeed) * 0.2;
    this.lastSite = site;
  }

  // ---- the robot entertaining itself ----
  autopilot(dt) {
    const A = this.auto;
    if (!A.on) return;
    const site = this.sitePos();
    const near = (x, y, tol) => Math.hypot(site[0] - x, site[1] - y) < tol;
    A.wait -= dt;
    switch (A.state) {
      case 'think': {
        if (this.mode === 'carry') { A.toy = this.held; A.state = 'grab'; A.wait = 4; break; }
        if (A.wait > 0 || this.mode !== 'hover') break;
        const free = this.toys.filter((t) => {
          const p = this.toyPos(t);
          const r = Math.hypot(p[0], p[1]);
          return r > 0.33 && r < 0.7 && this.toyAt(p[0], p[1], 0.04) === t;
        });
        if (!free.length) { A.wait = 1; break; }
        A.toy = free[Math.floor(this.rand() * free.length)];
        A.state = 'reach'; A.wait = 4;
        break;
      }
      case 'reach': {
        const p = this.toyPos(A.toy);
        this.pointer = [p[0], p[1]];
        if (near(p[0], p[1], 0.012) && this.mode === 'hover') { this.press(); A.state = 'grab'; A.wait = 4; }
        else if (A.wait < 0) A.state = 'think';
        break;
      }
      case 'grab':
        if (this.mode === 'carry') {
          const bases = this.toys.filter((t) => {
            if (t === A.toy || t.sphere) return false;
            const p = this.toyPos(t);
            const r = Math.hypot(p[0], p[1]);
            return r > 0.36 && r < 0.66 && p[2] < 0.1 && this.toyAt(p[0], p[1], 0.04, A.toy) === t;
          });
          if (bases.length && this.rand() < 0.6) {
            const p = this.toyPos(bases[Math.floor(this.rand() * bases.length)]);
            A.dest = [p[0], p[1]];
          } else {
            for (let i = 0; i < 20; i++) {
              const r = 0.4 + this.rand() * 0.24, a = (this.rand() - 0.5) * 1.9;
              A.dest = [r * Math.cos(a), r * Math.sin(a)];
              if (!this.toyAt(A.dest[0], A.dest[1], 0.09, A.toy)) break;
            }
          }
          A.state = 'move'; A.wait = 5;
        } else if (this.mode === 'hover' || A.wait < 0) { A.state = 'think'; A.wait = 0.4; }
        break;
      case 'move':
        this.pointer = A.dest.slice();
        if (this.mode !== 'carry') { A.state = 'think'; A.wait = 0.4; }
        else if ((near(A.dest[0], A.dest[1], 0.01) && site[2] > this.carryZ - 0.02) || A.wait < 0) {
          this.press(); A.state = 'think'; A.wait = 1.4;
        }
        break;
    }
  }

  // ---- inverse kinematics: damped least squares on the pinch site ----
  solveIK() {
    const { mj, model: m, ik, nv } = this;
    const T = this.target;
    const c = Math.cos(T.yaw), s = Math.sin(T.yaw);
    const Rd = [c, s, 0, s, -c, 0, 0, 0, -1]; // row-major: columns are x, y, z of the tool
    const q = this.qdes;
    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < 7; i++) ik.qpos[i] = q[i];
      mj.mj_kinematics(m, ik);
      mj.mj_comPos(m, ik);
      mj.mj_jacSite(m, ik, this.jacp, this.jacr, this.site);
      const Jp = this.jacp.GetView(), Jr = this.jacr.GetView();
      const x = ik.site_xpos, R = ik.site_xmat, o = 3 * this.site, r = 9 * this.site;
      const e = [T.pos[0] - x[o], T.pos[1] - x[o + 1], T.pos[2] - x[o + 2], 0, 0, 0];
      for (let col = 0; col < 3; col++) { // 0.5 * sum(cur_i x des_i)
        const a0 = R[r + col], a1 = R[r + 3 + col], a2 = R[r + 6 + col];
        const b0 = Rd[col], b1 = Rd[3 + col], b2 = Rd[6 + col];
        e[3] += 0.5 * (a1 * b2 - a2 * b1);
        e[4] += 0.5 * (a2 * b0 - a0 * b2);
        e[5] += 0.5 * (a0 * b1 - a1 * b0);
      }
      const J = new Float64Array(42);
      for (let row = 0; row < 3; row++) for (let j = 0; j < 7; j++) {
        J[row * 7 + j] = Jp[row * nv + j];
        J[(row + 3) * 7 + j] = Jr[row * nv + j];
      }
      const A = new Float64Array(36);
      for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) {
        let sum = a === b ? 2e-3 : 0;
        for (let j = 0; j < 7; j++) sum += J[a * 7 + j] * J[b * 7 + j];
        A[a * 6 + b] = sum;
      }
      // Secondary goal: drift back toward a comfortable posture, in the null space.
      const pull = HOME.map((h, i) => 0.04 * (h - q[i]));
      const Jpull = new Float64Array(6);
      for (let a = 0; a < 6; a++) for (let j = 0; j < 7; j++) Jpull[a] += J[a * 7 + j] * pull[j];
      const rhs = e.map((v, a) => v - Jpull[a]);
      const y = solve(A, rhs, 6);
      for (let j = 0; j < 7; j++) {
        let dq = pull[j];
        for (let a = 0; a < 6; a++) dq += J[a * 7 + j] * y[a];
        const lo = m.jnt_range[2 * j] + 0.02, hi = m.jnt_range[2 * j + 1] - 0.02;
        q[j] = clamp(q[j] + clamp(dq, -0.12, 0.12), lo, hi);
      }
    }
    const ctrl = this.data.ctrl;
    for (let i = 0; i < 7; i++) ctrl[i] = q[i];
    ctrl[7] = (1 - T.grip) * 255;
  }

  /** Put toys that left the table back within reach. */
  tidy(dt) {
    const q = this.data.qpos, v = this.data.qvel;
    for (const t of this.toys) {
      const r = Math.hypot(q[t.q], q[t.q + 1]);
      const lost = r > 0.8 || r < 0.24 || q[t.q + 2] < -0.05;
      t.lostFor = lost && t !== this.held ? t.lostFor + dt : 0;
      if (t.lostFor < 1.5) continue;
      const rr = 0.42 + this.rand() * 0.2, a = (this.rand() - 0.5) * 1.8;
      q[t.q] = rr * Math.cos(a); q[t.q + 1] = rr * Math.sin(a); q[t.q + 2] = 0.25;
      q[t.q + 3] = 1; q[t.q + 4] = q[t.q + 5] = q[t.q + 6] = 0;
      for (let k = 0; k < 6; k++) v[t.v + k] = 0;
      t.lostFor = 0;
    }
  }

  /** Advance the world by `seconds` of simulated time. */
  advance(seconds) {
    const { mj, model: m, data: d } = this;
    const n = Math.min(40, Math.round(seconds / m.opt.timestep));
    for (let i = 0; i < n; i++) {
      if (this.steps % CONTROL_EVERY === 0) {
        const dt = CONTROL_EVERY * m.opt.timestep;
        this.autopilot(dt);
        this.think(dt);
        this.solveIK();
        this.tidy(dt);
      }
      mj.mj_step(m, d);
      this.steps++;
    }
  }
}
