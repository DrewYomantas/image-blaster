import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


WIDTH, HEIGHT = 640, 480
VERSION = "synthetic-room-v1"


FONT = {
    "A": "01110100011000111111100011000110001", "C": "01111100001000010000100001000001111",
    "D": "11110100011000110001100011000111110", "E": "11111100001000011110100001000011111",
    "F": "11111100001000011110100001000010000", "G": "01111100001000010111100011000101111",
    "H": "10001100011000111111100011000110001", "I": "11111001000010000100001000010011111",
    "L": "10000100001000010000100001000011111", "N": "10001110011010110011100011000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "R": "11110100011000111110101001001010001", "S": "01111100001000001110000010000111110",
    "T": "11111001000010000100001000010000100", "U": "10001100011000110001100011000101110",
    "V": "10001100011000110001100010101000100", "W": "10001100011000110101101011101110001",
    "Y": "10001100010101000100001000010000100", "0": "01110100011001110101110011000101110",
    "1": "00100011000010000100001000010001110", "2": "01110100010000100010001000100011111",
    "3": "11110000010000101110000010000111110", "4": "00010001100101010010111110001000010",
    "5": "11111100001000011110000010000111110", "6": "01110100001000011110100011000101110",
    "7": "11111000010001000100010000100001000", "8": "01110100011000101110100011000101110",
    "9": "01110100011000101111000010000101110", "-": "00000000000000011111000000000000000",
}


def png(path, pixels, label=None):
    pixels = np.asarray(pixels, dtype=np.uint8)
    if label:
        banner = np.full((22, pixels.shape[1], 3), 15, dtype=np.uint8)
        for index, char in enumerate(label):
            glyph = np.array(list(FONT.get(char, "0" * 35)), dtype=np.uint8).reshape(7, 5)
            glyph = np.repeat(np.repeat(glyph, 2, 0), 2, 1)
            left = 6 + index * 12
            if left + 10 <= banner.shape[1]:
                banner[4:18, left:left + 10] = np.where(glyph[..., None] > 0, 235, 15)
        pixels = np.concatenate([banner, pixels])
    height, width, _ = pixels.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def camera(position, target):
    forward = np.array(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.column_stack([rotation, -rotation @ position])


def scene():
    items = []
    def box(name, label, low, high, color):
        items.append(dict(name=name, label=label, low=low, high=high, color=color))
    box("floor", 1, [-2.5, -.1, -.45], [2.5, 0, 3.7], [.53, .37, .23])
    box("ceiling", 2, [-2.5, 2.7, -.45], [2.5, 2.8, 3.7], [.86, .85, .8])
    box("left-wall", 3, [-2.5, 0, -.45], [-2.4, 2.7, 3.7], [.68, .72, .70])
    box("right-wall", 4, [2.4, 0, -.45], [2.5, 2.7, 3.7], [.68, .72, .70])
    box("rear-wall", 5, [-2.5, 0, 3.6], [2.5, 2.7, 3.7], [.75, .75, .70])
    box("front-left", 6, [-2.4, 0, -.1], [-.6, 2.7, 0], [.79, .76, .69])
    box("front-right", 6, [.6, 0, -.1], [2.4, 2.7, 0], [.79, .76, .69])
    box("front-bottom", 6, [-.6, 0, -.1], [.6, .5, 0], [.65, .62, .56])
    box("front-top", 6, [-.6, 1.4, -.1], [.6, 2.7, 0], [.79, .76, .69])
    box("opening-back", 7, [-.6, .5, -.45], [.6, 1.4, -.35], [.12, .13, .14])
    box("opening-left", 8, [-.7, .5, -.35], [-.6, 1.4, 0], [.23, .23, .23])
    box("opening-right", 8, [.6, .5, -.35], [.7, 1.4, 0], [.23, .23, .23])
    box("opening-bottom", 8, [-.6, .4, -.35], [.6, .5, 0], [.23, .23, .23])
    box("opening-top", 8, [-.6, 1.4, -.35], [.6, 1.5, 0], [.23, .23, .23])
    box("hearth", 9, [-.9, 0, 0], [.9, .14, .55], [.37, .38, .38])
    box("mantel", 10, [-.95, 1.6, 0], [.95, 1.72, .24], [.28, .16, .08])
    box("tall-left-box", 11, [-1.68, 0, .55], [-1.2, 1, .97], [.27, .37, .42])
    box("foreground-stool", 12, [-.11, 0, 1.79], [.41, .6, 2.31], [.58, .31, .18])
    box("right-seat-box", 13, [.97, 0, 1.02], [1.73, .8, 1.68], [.34, .44, .28])
    return items


def rays(intrinsics, extrinsics, width, height):
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1)
    local = pixels @ np.linalg.inv(intrinsics).T
    rotation, translation = extrinsics[:, :3], extrinsics[:, 3]
    return -rotation.T @ translation, local @ rotation


def intersect(origin, direction, items):
    depth = np.full(direction.shape[:2], np.inf)
    labels = np.zeros(depth.shape, dtype=np.uint16)
    normals = np.zeros(direction.shape)
    colors = np.zeros(direction.shape)
    safe = np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    for item in items:
        low, high = np.array(item["low"]), np.array(item["high"])
        first, second = (low - origin) / safe, (high - origin) / safe
        near, far = np.minimum(first, second), np.maximum(first, second)
        entering, leaving = near.max(-1), far.min(-1)
        distance = np.where(entering > 1e-7, entering, leaving)
        hit = (leaving >= np.maximum(entering, 1e-7)) & (distance < depth)
        axis = np.where(entering > 1e-7, near.argmax(-1), far.argmin(-1))
        normal = np.eye(3)[axis] * np.where(entering[..., None] > 1e-7, -np.sign(safe), np.sign(safe))
        depth[hit], labels[hit] = distance[hit], item["label"]
        normals[hit], colors[hit] = normal[hit], item["color"]
    return depth, labels, normals, colors


def render(intrinsics, extrinsics, items):
    origin, direction = rays(intrinsics, extrinsics, WIDTH, HEIGHT)
    depth, labels, normal, color = intersect(origin, direction, items)
    point = origin + direction * depth[..., None]
    light_position = np.array([-1.4, 2.5, 2.3])
    to_light = light_position - point
    light_distance = np.linalg.norm(to_light, axis=-1)
    light_direction = to_light / light_distance[..., None]
    shadow_depth, _, _, _ = intersect(point + normal * 1e-5, light_direction, items)
    lit = shadow_depth > light_distance - 1e-4
    diffuse = np.maximum(0, (normal * light_direction).sum(-1))
    shade = .44 + .52 * diffuse * np.where(lit, 1, .22)
    wood = 1 + .055 * np.sin(point[..., 0] * 85 + np.sin(point[..., 2] * 12))
    boards = np.where(np.mod(point[..., 0] + 2.4, .19) < .008, .75, 1)
    texture = np.where(labels == 1, wood * boards, 1 + .015 * np.sin(point[..., 0] * 51) * np.sin(point[..., 1] * 48))
    image = np.power(np.clip(color * shade[..., None] * texture[..., None], 0, 1), 1 / 2.2)
    return (image * 255 + .5).astype(np.uint8), depth.astype(np.float32), labels


def depth_image(depth):
    normalized = np.clip(depth / 6, 0, 1)
    rgb = np.stack([normalized, 1 - np.abs(normalized * 2 - 1), 1 - normalized], -1)
    rgb[~np.isfinite(depth) | (depth <= 0)] = 0
    return (rgb * 255).astype(np.uint8)


def generate(output):
    output = Path(output)
    input_dir, truth_dir = output / "inputs", output / "ground-truth"
    input_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = np.repeat(np.array([[[350., 0, 319.5], [0, 350., 239.5], [0, 0, 1]]]), 4, axis=0)
    positions = [[-1.55, 1.6, 3.0], [-.55, 1.45, 3.1], [.65, 1.65, 3.0], [1.6, 1.5, 2.85]]
    extrinsics = np.array([camera(np.array(p), [0, 1.15, .35]) for p in positions])
    images, depths, labels = zip(*(render(k, e, scene()) for k, e in zip(intrinsics, extrinsics)))
    hashes = []
    for index, rgb in enumerate(images):
        name = f"view-{index + 1:02}.png"
        png(input_dir / name, rgb)
        hashes.append(dict(file=name, sha256=hashlib.sha256((input_dir / name).read_bytes()).hexdigest()))
        png(truth_dir / f"SYNTHETIC-input-{index + 1:02}.png", rgb, f"SYNTHETIC INPUT {index + 1:02}")
        png(truth_dir / f"GT-depth-{index + 1:02}.png", depth_image(depths[index]), f"SYNTHETIC GT DEPTH {index + 1:02}")
    provider_inputs = dict(fixture=VERSION, kind="synthetic", rights="project-authored synthetic fixture; MIT", ordered_images=hashes)
    (input_dir / "manifest.json").write_text(json.dumps(provider_inputs, indent=2) + "\n")
    np.savez_compressed(truth_dir / "truth.npz", depths=depths, labels=labels, intrinsics=intrinsics, extrinsics=extrinsics)
    anchor_points = np.array([[-.54, .95, -.35], [.54, .95, -.35]])
    local = anchor_points @ extrinsics[1, :, :3].T + extrinsics[1, :, 3]
    projected = local @ intrinsics[1].T
    anchor_pixels = (projected[:, :2] / projected[:, 2:]).tolist()
    dimensions = [dict(id="room", labels=[1, 2, 3, 4, 5, 6], dimensions_m=[4.8, 2.7, 3.6]),
                  dict(id="opening-back", labels=[7], dimensions_m=[1.2, .9, 0.0]),
                  dict(id="opening-recess", labels=[7, 8], dimensions_m=[1.2, .9, .35]),
                  *[dict(id=item["name"], labels=[item["label"]], dimensions_m=(np.array(item["high"]) - item["low"]).tolist()) for item in scene() if item["label"] >= 9]]
    metadata = dict(version=VERSION, kind="synthetic", renderer="numpy-analytic-box-cpu-v1", numpy_version=np.__version__,
                    renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), resolution=[WIDTH, HEIGHT],
                    settings=dict(randomness="none; deterministic formula shading", seed=None, camera_z_near_m=1e-7, far_clipping_m=None,
                                  occlusion="nearest exact box intersection", exposure="fixed gamma 1/2.2", antialiasing="none; one integer-centre ray per pixel"),
                    camera_names=["left-oblique", "front-left", "front-right", "right-context"],
                    seed=None, world_frame="metres, right-handed, x-right y-up z-toward-rear-wall", camera_frame="world-to-camera; x-right y-down z-forward; integer pixel centres",
                    depth="positive camera-z metres, not Euclidean ray range", intrinsics=intrinsics.tolist(), extrinsics=extrinsics.tolist(),
                    primitives=scene(), dimensions=dimensions, anchor=dict(view_index=1, pixels=anchor_pixels, fraction_of_opening_width=.9, known_width_m=1.2),
                    evaluation=dict(dimension_estimator="axis-aligned observed point bounds; visibility limits reported, no hidden completion", boundary_threshold_m=.05,
                                    fireplace_boundary_roi=dict(labels=[7, 8], margin_prediction_pixels=3),
                                    confidence_policy="finite positive depths and finite nonnegative confidence; all valid samples, no tuned cutoff", alignment="one opening-width scale, then camera-centre rigid rotation/translation; no further scale fit"),
                    input_hashes=hashes)
    (truth_dir / "geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".image-blaster/benchmark/synthetic-room-v1")
    arguments = parser.parse_args()
    result = generate(arguments.out)
    print(json.dumps(dict(fixture=result["version"], output=str(Path(arguments.out).resolve()), images=result["input_hashes"]), indent=2))
