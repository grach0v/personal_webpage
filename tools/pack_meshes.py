"""Decimate robot meshes and pack them as text OBJ into one JSON file.

usage: uv run --with trimesh --with pyfqmr --with numpy python pack_meshes.py \\
           out.json ratio <menagerie>/franka_emika_panda/assets <menagerie>/robotiq_2f85/assets
"""
import json, sys, pathlib
import numpy as np, trimesh, pyfqmr

out, ratio = pathlib.Path(sys.argv[1]), float(sys.argv[2])
panda, gripper = pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
bundle, tot_in, tot_out = {}, 0, 0

def add(f, name, simplify):
    global tot_in, tot_out
    m = trimesh.load(f, force="mesh", process=True)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    v, fc = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces)
    tot_in += len(fc)
    target = max(500, int(len(fc) * ratio))
    if simplify and target < len(fc):
        # preserve_border keeps the open rims where two links meet from fraying
        q = pyfqmr.Simplify()
        q.setMesh(v, fc)
        q.simplify_mesh(target_count=target, aggressiveness=4, preserve_border=True, verbose=False)
        v, fc, _ = q.getMesh()
    tot_out += len(fc)
    lines = ["v %.5g %.5g %.5g" % tuple(p) for p in v]
    lines += ["f %d %d %d" % tuple(t + 1) for t in fc]
    bundle[name] = "\n".join(lines) + "\n"

for f in sorted(panda.iterdir()):
    if f.stem.startswith(("hand", "finger")):
        continue  # the Franka hand is replaced by the Robotiq gripper
    if f.suffix == ".stl":
        add(f, f.stem + "_c.obj", False)       # collision hulls, already tiny
    elif f.suffix == ".obj":
        add(f, f.name, "collision" not in f.name)
for f in sorted(gripper.iterdir()):
    if f.suffix == ".stl":
        add(f, f.stem + ".obj", True)
out.write_text(json.dumps(bundle, separators=(",", ":")))
print(len(bundle), "meshes", tot_in, "->", tot_out, "faces", out.stat().st_size, "bytes")
