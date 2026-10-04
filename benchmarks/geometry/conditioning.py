import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PERTURBATION = {
    "declared_before_inference": True,
    "focal_multiplier": 1.05,
    "local_xyz_euler_degrees": [[2, 0, 0], [0, -2, 0], [0, 0, 2], [-1.2, 1.6, 0]],
    "world_camera_centre_offsets_m": [[.03, .04, 0], [-.04, 0, .03], [0, -.03, -.04], [-.03, .04, 0]],
    "rotation_application": "Rz @ Ry @ Rx @ original_world_to_camera_rotation; translation recomputed from perturbed centre",
    "fixture_images_and_cameras_unchanged": True,
}


def conditioning_files(truth_path, output):
    truth_path, output = Path(truth_path), Path(output)
    truth = json.loads(truth_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    k = np.array(truth["intrinsics"])
    e = np.repeat(np.eye(4)[None], len(k), axis=0)
    e[:, :3] = truth["extrinsics"]
    base = dict(provenance="experiment-oracle", cameraConvention="opencv-world-to-camera-scene-y-up-meters")
    lanes = dict(B=dict(mode="intrinsics-only", **base, intrinsics=k.tolist()),
                 C=dict(mode="pose", **base, intrinsics=k.tolist(), extrinsics=e.tolist()))
    perturbed = e.copy()
    for i, (angles, offset) in enumerate(zip(PERTURBATION["local_xyz_euler_degrees"], PERTURBATION["world_camera_centre_offsets_m"])):
        x, y, z = np.deg2rad(angles)
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        centre = -e[i, :3, :3].T @ e[i, :3, 3] + offset
        perturbed[i, :3, :3] = rz @ ry @ rx @ e[i, :3, :3]
        perturbed[i, :3, 3] = -perturbed[i, :3, :3] @ centre
    kd = k.copy()
    kd[:, 0, 0] *= PERTURBATION["focal_multiplier"]
    kd[:, 1, 1] *= PERTURBATION["focal_multiplier"]
    lanes["D"] = dict(mode="pose", **base, intrinsics=kd.tolist(), extrinsics=perturbed.tolist())
    receipts = {}
    for lane, value in lanes.items():
        file = output / f"lane-{lane}-conditioning.json"
        text = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if file.exists() and file.read_text() != text:
            raise ValueError("Refusing to change a predeclared conditioning file")
        file.write_text(text)
        receipts[lane] = dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    receipt = dict(kind="experiment-only oracle cameras derived from frozen synthetic evaluation truth", truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
                   perturbation=PERTURBATION, lanes=receipts, forbidden_inputs_absent=["depth", "labels", "object dimensions", "opening scale anchor"])
    (output / "conditioning-declaration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(conditioning_files(args.truth, args.out), indent=2))
