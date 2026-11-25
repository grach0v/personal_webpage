"""Shrink a robot episode recording (.rrd) so it is small enough to serve from the site.

    python tools/slim_rrd.py data/episode_000052.rrd playground/recordings/stack.rrd

What it does:
  * re-encodes the camera videos at a lower bitrate (the wrist camera also at half size)
  * turns the 16-bit depth PNGs into colour-mapped H.264 video
  * decimates the robot meshes and drops the collision meshes
  * drops the second stereo eye, the fixed cameras' depth, the log_time timeline, and cell details
    (robot IP, camera serials) that have no business on a public page
  * replaces the blueprint with a layout that fits an embedded viewer

Needs: rerun-sdk (same version as the recording), av, numpy, pillow, trimesh, pycollada, pyfqmr.
"""
import io
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyfqmr
import rerun as rr
import rerun.blueprint as rrb
import trimesh
from PIL import Image
from rerun.chunk import Chunk, LazyChunkStream, RrdReader

CAMERAS = "/world/cameras"
RGB = {  # entity -> (scale, crf)
    f"{CAMERAS}/d455/rgb": (1.0, 30),
    f"{CAMERAS}/d415/rgb": (1.0, 30),
    f"{CAMERAS}/zed_left/rgb": (0.5, 28),
}
DEPTH = [f"{CAMERAS}/zed_depth/depth"]  # the two fixed cameras' depth is noisy and costs 2 MB each
DROP_PREFIXES = (
    f"{CAMERAS}/zed_right", f"{CAMERAS}/d455_depth", f"{CAMERAS}/d415_depth",
    "/world/franka_cell/collision_geometries",
    "/__properties/cell", "/__properties/d415", "/__properties/d455", "/__properties/zed",
)
MESH_KEEP = 0.12  # fraction of triangles kept
TIMELINE = "capture_time"
FRUSTUM = 0.22  # metres from each camera to its picture in the 3D view


def blob(column, row):
    return column[row].values[0].values.to_numpy(zero_copy_only=False).tobytes()


def times(batch):
    return batch.column(TIMELINE).cast(pa.int64()).to_numpy(zero_copy_only=False)


def turbo(x):
    """Polynomial fit of the Turbo colour map; x in [0, 1] -> uint8 RGB."""
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def decode_h264(samples):
    decoder = av.CodecContext.create("h264", "r")
    frames = []
    for sample in samples + [None]:
        for frame in decoder.decode(av.Packet(sample) if sample else None):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def encode_h264(frames, crf):
    """RGB frames -> [(annex-b bytes, is_keyframe)], one per frame, no B-frames."""
    height, width = frames[0].shape[:2]
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = width, height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {"crf": str(crf), "preset": "veryslow", "bf": "0", "g": "60", "x264-params": "repeat-headers=1"}
    out = []
    for index, image in enumerate(frames + [None]):
        frame = None
        if image is not None:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            frame.pts = index
        for packet in encoder.encode(frame):
            out.append((bytes(packet), bool(packet.is_keyframe)))
    assert len(out) == len(frames)
    return out


def video_chunk(entity, stamps, packets):
    return Chunk.from_columns(
        entity,
        indexes=[rr.TimeColumn(TIMELINE, timestamp=np.asarray(stamps, dtype="datetime64[ns]"))],
        columns=rr.VideoStream.columns(sample=[p for p, _ in packets], is_keyframe=[k for _, k in packets]),
    )


def static_chunk(entity, columns):
    return Chunk.from_columns(entity, indexes=[], columns=columns)


def simplify(mesh):
    target = int(len(mesh.faces) * MESH_KEEP)
    if target < 200 or target >= len(mesh.faces):
        return mesh
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    simplifier.simplify_mesh(target_count=target, preserve_border=True, verbose=False)
    vertices, faces, _ = simplifier.getMesh()
    return trimesh.Trimesh(vertices, faces, visual=None, process=False)


def slim_mesh(data, media_type):
    """Returns (bytes, media_type) for a decimated copy of the mesh."""
    if media_type == "model/stl":
        mesh = simplify(trimesh.load(io.BytesIO(data), file_type="stl"))
        return mesh.export(file_type="stl"), media_type
    scene = trimesh.load(io.BytesIO(data), file_type="dae", force="scene")
    for name, mesh in list(scene.geometry.items()):
        colour = mesh.visual.material.main_color if hasattr(mesh.visual, "material") else [200, 200, 200, 255]
        small = simplify(mesh)
        small.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=colour, metallicFactor=0.1, roughnessFactor=0.6))
        scene.geometry[name] = small
    return scene.export(file_type="glb"), "model/gltf-binary"


def blueprint():
    def tactile(side):
        return rrb.TensorView(
            origin=f"/tactile/{side}/contact", name=f"Touch, {side} finger",
            scalar_mapping=rrb.TensorScalarMapping(colormap="viridis", mag_filter="nearest"),
        )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/world", name="Robot",
                    contents=["+ /world/**", "- /world/cameras/zed_depth/**"],
                    eye_controls=rrb.EyeControls3D(position=(1.2, -0.95, 0.75), look_target=(0.35, 0.0, 0.25), eye_up=(0, 0, 1)),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin=f"{CAMERAS}/zed_left", name="Wrist camera"),
                    rrb.Spatial2DView(origin=f"{CAMERAS}/zed_depth", name="Wrist depth"),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin=f"{CAMERAS}/d455", name="Left camera"),
                    rrb.Spatial2DView(origin=f"{CAMERAS}/d415", name="Right camera"),
                ),
                column_shares=[5, 3, 3],
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/task/prompt", name="Task"),
                tactile("left"),
                tactile("right"),
                rrb.TimeSeriesView(origin="/tactile/total", name="Fingertip force"),
                rrb.TimeSeriesView(origin="/gripper", name="Gripper"),
                column_shares=[2, 1.6, 1.6, 3, 3],
            ),
            row_shares=[5, 2],
        ),
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
        rrb.TimePanel(state="collapsed", timeline=TIMELINE),
    )


def main(source, target):
    reader = RrdReader(source)
    (entry,) = reader.recordings()
    videos = {entity: [] for entity in [*RGB, *DEPTH]}  # entity -> [(time, payload)]
    depth_range = {}
    kept = []

    for chunk in reader.stream(store=entry):
        entity = chunk.entity_path
        if entity.startswith(DROP_PREFIXES):
            continue
        batch = chunk.to_record_batch()
        names = batch.schema.names
        if entity in videos:
            if "VideoStream:sample" in names:
                videos[entity] += [(t, blob(batch.column("VideoStream:sample"), i)) for i, t in enumerate(times(batch))]
            elif "EncodedDepthImage:blob" in names:
                videos[entity] += [(t, blob(batch.column("EncodedDepthImage:blob"), i)) for i, t in enumerate(times(batch))]
                depth_range[entity] = batch.column("EncodedDepthImage:depth_range")[0].as_py()[0]
            elif chunk.is_static:
                kept.append(chunk)  # the codec
            continue  # keyframe flags are rebuilt
        if "Asset3D:blob" in names:
            data, media_type = slim_mesh(blob(batch.column("Asset3D:blob"), 0), batch.column("Asset3D:media_type")[0].as_py()[0])
            albedo = batch.column("Asset3D:albedo_factor")[0].as_py() if "Asset3D:albedo_factor" in names else None
            kept.append(static_chunk(entity, rr.Asset3D.columns(blob=[data], media_type=[media_type], albedo_factor=albedo)))
            continue
        if entity + "/rgb" in RGB and "Pinhole:resolution" in names:
            scale = RGB[entity + "/rgb"][0]
            matrix = np.array(batch.column("Pinhole:image_from_camera")[0].as_py()[0]).reshape(3, 3).T  # column-major
            matrix[:2] *= scale
            resolution = np.array(batch.column("Pinhole:resolution")[0].as_py()[0]) * scale
            kept.append(static_chunk(entity, rr.Pinhole.columns(
                image_from_camera=[matrix], resolution=[resolution], camera_xyz=[rr.ViewCoordinates.RDF],
                image_plane_distance=[FRUSTUM])))
            continue
        kept.append(chunk)

    for entity, samples in videos.items():
        samples.sort(key=lambda s: s[0])
        stamps = [t for t, _ in samples]
        if entity in RGB:
            scale, crf = RGB[entity]
            frames = decode_h264([s for _, s in samples])
            if scale != 1.0:
                size = (round(frames[0].shape[1] * scale), round(frames[0].shape[0] * scale))
                frames = [np.asarray(Image.fromarray(f).resize(size, Image.LANCZOS)) for f in frames]
        else:
            near, far = depth_range[entity]
            frames = []
            for _, png in samples:
                depth = np.asarray(Image.open(io.BytesIO(png))).astype(np.float32)
                colour = turbo(np.clip((depth - near) / (far - near), 0, 1))
                colour[depth == 0] = (24, 22, 30)  # no reading
                frames.append(colour)
            crf = 30
            kept.append(static_chunk(entity, rr.VideoStream.columns(codec=[rr.VideoCodec.H264])))
        assert len(frames) == len(stamps), (entity, len(frames), len(stamps))
        packets = encode_h264(frames, crf)
        print(f"{entity}: {len(frames)} frames, {sum(len(p) for p, _ in packets) / 1e6:.2f} MB")
        kept.append(video_chunk(entity, stamps, packets))

    with tempfile.TemporaryDirectory() as tmp:
        raw, layout, merged = (str(Path(tmp) / name) for name in ("raw.rrd", "layout.rbl", "merged.rrd"))
        LazyChunkStream.from_iter(kept).write_rrd(raw, application_id=entry.application_id, recording_id=entry.recording_id)
        blueprint().save(entry.application_id, layout)
        cli = [sys.executable, "-m", "rerun", "rrd"]
        subprocess.run([*cli, "merge", raw, layout, "-o", merged], check=True)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([*cli, "filter", "--drop-timeline", "log_time", merged, "-o", target], check=True)
    print(f"{target}: {Path(target).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
