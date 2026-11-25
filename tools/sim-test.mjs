// Headless check of the playground: `node tools/sim-test.mjs`
// Runs the self-playing robot for a while and reports what it managed to do.
import { readFileSync } from 'node:fs';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { Sim } from '../playground/sim.js';

const dir = new URL('../playground/model/', import.meta.url);
const read = (f) => readFileSync(new URL(f, dir), 'utf8');
const mujoco = await loadMujoco();
const t0 = performance.now();
const sim = new Sim(mujoco, {
  pandaXml: read('panda.xml'),
  gripperXml: read('2f85.xml'),
  sceneXml: read('scene.xml'),
  meshes: JSON.parse(read('meshes.json')),
}, { seed: Number(process.argv[2] ?? 1) });
console.log('load ms', (performance.now() - t0).toFixed(0), 'nv', sim.nv, 'toys', sim.toys.length);

const seconds = Number(process.argv[3] ?? 120);
let grabs = 0, tries = 0, prev = 'hover', maxStack = 0, trackErr = 0, n = 0, maxForce = 0;
const t1 = performance.now();
for (let t = 0; t < seconds; t += 0.02) {
  sim.advance(0.02);
  if (sim.mode !== prev) {
    if (sim.mode === 'descend') tries++;
    if (sim.mode === 'carry') grabs++;
    prev = sim.mode;
  }
  const s = sim.sitePos();
  trackErr += Math.hypot(s[0] - sim.target.pos[0], s[1] - sim.target.pos[1], s[2] - sim.target.pos[2]); n++;
  if (sim.mode === 'carry') maxForce = Math.max(maxForce, ...sim.tactile().map((g) => Math.max(...g)));
  for (const toy of sim.toys) if (toy !== sim.held) maxStack = Math.max(maxStack, sim.toyPos(toy)[2]);
  if (process.argv[4] && Math.round(t * 50) % 25 === 0) console.log(t.toFixed(1), sim.mode, sim.auto.state, s.map((v) => v.toFixed(3)).join(' '), sim.fingerWidth().toFixed(3));
}
const wall = performance.now() - t1;
console.log(`grabs ${grabs}/${tries}  highest resting toy z=${maxStack.toFixed(3)}  mean tracking err ${(1000 * trackErr / n).toFixed(1)} mm`);
console.log(`peak taxel while carrying ${maxForce.toFixed(2)} N, finger gap now ${sim.fingerWidth().toFixed(3)} m`);
console.log(`speed ${(seconds * 1000 / wall).toFixed(1)}x realtime`);
console.log('toys', sim.toys.map((t) => sim.toyPos(t).map((v) => v.toFixed(2)).join(',')).join('  '));
