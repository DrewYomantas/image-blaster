import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import anchor_scale, boundary_metrics, cross_view_metrics, depth_metrics, dilate, held_out_dimensions, load_prediction, points, project_image
from fixture import depth_image, intersect, png, rays


SURFACE_GROUPS = {"front-wall": [6], "floor": [1], "side-walls": [3, 4], "opening-back": [7],
                  "opening-recess": [7, 8], "hearth": [9], "mantel": [10]}
SURFACE_DECLARATION = dict(version="known-planes-v1", membership="analytic frozen truth ray/box nearest face; same primitive and face throughout fixed 3x3 neighbourhood; image rim excluded",
                           plane="signed perpendicular distance to known outward primitive face: mean, RMS, standard deviation; no fitted plane or prediction rejection",
                           normals="signed normalized cosine against outward analytic visible face; mean/median angle and fraction <=15 degrees; no absolute cosine",
                           groups=SURFACE_GROUPS, neighbourhood_radius_pixels=1, normal_threshold_degrees=15)


def truth_geometry(fixture, width, height):
    fixture = Path(fixture)
    metadata = json.loads((fixture / "ground-truth/geometry.json").read_text())
    for item in metadata["input_hashes"]:
        if hashlib.sha256((fixture / "inputs" / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("fixture input hash mismatch")
    intrinsics = np.array(metadata["intrinsics"])
    sx, sy = width / metadata["resolution"][0], height / metadata["resolution"][1]
    intrinsics[:, 0] *= sx
    intrinsics[:, 1] *= sy
    intrinsics[:, 0, 2] += (sx - 1) / 2
    intrinsics[:, 1, 2] += (sy - 1) / 2
    extrinsics = np.array(metadata["extrinsics"])
    depths, labels, normals, colors, cloud = [], [], [], [], []
    for k, e in zip(intrinsics, extrinsics):
        origin, direction = rays(k, e, width, height)
        depth, label, normal, color = intersect(origin, direction, metadata["primitives"])
        depths.append(depth)
        labels.append(label)
        normals.append(normal)
        colors.append((np.power(color, 1 / 2.2) * 255).astype(np.uint8))
        cloud.append(origin + direction * depth[..., None])
    return dict(metadata=metadata, intrinsics=intrinsics, extrinsics=extrinsics, depths=np.array(depths),
                labels=np.array(labels), normals=np.array(normals), colors=np.array(colors), cloud=np.array(cloud))


def surface_membership(truth_cloud, labels, truth_normals, metadata):
    face_ids = np.full(labels.shape, -1, dtype=np.int32)
    faces = {}
    for primitive_index, item in enumerate(metadata["primitives"]):
        low, high = np.array(item["low"]), np.array(item["high"])
        selected = (labels == item["label"]) & (truth_cloud >= low - 1e-7).all(-1) & (truth_cloud <= high + 1e-7).all(-1)
        if not selected.any():
            continue
        distances = np.concatenate([np.abs(truth_cloud[selected] - low), np.abs(truth_cloud[selected] - high)], axis=-1)
        closest = distances.argmin(-1)
        for side in range(6):
            axis, sign = side % 3, -1 if side < 3 else 1
            normal = np.eye(3)[axis] * sign
            selected_pixels = np.zeros(labels.shape, dtype=bool)
            selected_pixels[selected] = (closest == side) & (distances[:, side] < 1e-7) & ((truth_normals[selected] * normal).sum(-1) > .999)
            if not selected_pixels.any():
                continue
            identity = primitive_index * 6 + side
            face_ids[selected_pixels] = identity
            faces[identity] = dict(id=f"{item['name']}-{'xyz'[axis]}-{'low' if sign < 0 else 'high'}", primitive=item["name"],
                                   label=item["label"], axis=axis, sign=sign, plane_m=float(low[axis] if sign < 0 else high[axis]), normal=normal.tolist())
    padded = np.pad(face_ids, ((0, 0), (1, 1), (1, 1)), constant_values=-2)
    eligible = face_ids >= 0
    for y in range(3):
        for x in range(3):
            eligible &= padded[:, y:y + labels.shape[1], x:x + labels.shape[2]] == face_ids
    return face_ids, eligible, faces


def residual_statistics(residual, count):
    return dict(truth_eligible_samples=count, valid_samples=len(residual), prediction_coverage=len(residual) / count if count else 0.,
                signed_mean_cm=float(np.mean(residual) * 100) if len(residual) else None,
                rms_distance_cm=float(np.sqrt(np.mean(residual ** 2)) * 100) if len(residual) else None,
                scatter_std_cm=float(np.std(residual) * 100) if len(residual) else None)


def normal_statistics(angles, count):
    return dict(truth_eligible_samples=count, valid_samples=len(angles), prediction_coverage=len(angles) / count if count else 0.,
                mean_angular_error_degrees=float(np.mean(angles)) if len(angles) else None,
                median_angular_error_degrees=float(np.median(angles)) if len(angles) else None,
                fraction_within_15_degrees=float(np.mean(angles <= 15)) if len(angles) else None)


def surface_metrics(cloud, valid, membership, predicted_normals=None):
    face_ids, eligible, faces = membership
    residual_map = np.full(valid.shape, np.nan)
    angle_map = np.full(valid.shape, np.nan)
    face_results, group_results = [], {}
    if predicted_normals is not None:
        norms = np.linalg.norm(predicted_normals, axis=-1)
        normal_valid = valid & np.isfinite(predicted_normals).all(-1) & (norms > 1e-12)
        normalized = np.zeros_like(predicted_normals)
        normalized[normal_valid] = predicted_normals[normal_valid] / norms[normal_valid, None]
    for identity, face in faces.items():
        target = eligible & (face_ids == identity)
        selected = target & valid
        residual = (cloud[..., face["axis"]][selected] - face["plane_m"]) * face["sign"]
        residual_map[selected] = residual
        result = dict(**face, plane=residual_statistics(residual, int(target.sum())))
        if predicted_normals is not None:
            normal_selected = target & normal_valid
            angles = np.degrees(np.arccos(np.clip(normalized[normal_selected] @ face["normal"], -1, 1)))
            angle_map[normal_selected] = angles
            result["normals"] = normal_statistics(angles, int(target.sum()))
        face_results.append(result)
    for name, group_labels in SURFACE_GROUPS.items():
        group_faces = [identity for identity, face in faces.items() if face["label"] in group_labels]
        target = eligible & np.isin(face_ids, group_faces)
        residual = residual_map[target & valid]
        result = dict(labels=group_labels, plane=residual_statistics(residual, int(target.sum())))
        if predicted_normals is not None:
            result["normals"] = normal_statistics(angle_map[target & normal_valid], int(target.sum()))
        group_results[name] = result
    return dict(declaration=SURFACE_DECLARATION, faces=face_results, groups=group_results,
                normal_comparability="MoGe signed output normals scored directly" if predicted_normals is not None else "no provider normals available; normal accuracy not comparable"), residual_map, angle_map


def load_moge(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ["camera_points", "depths", "masks", "normals", "intrinsics", "intrinsics_normalized", "conditioning_mode"]
        if any(key not in archive for key in required):
            raise ValueError("MoGe NPZ missing required point/depth/mask/normal/intrinsic arrays")
        data = {key: np.array(archive[key]) for key in required}
        data["processed_images"] = np.array(archive["processed_images"]) if "processed_images" in archive else None
    mode = data["conditioning_mode"]
    if mode.shape != () or str(mode) not in ("none", "known-fov"):
        raise ValueError("invalid MoGe NPZ conditioning mode")
    data["conditioning_mode"] = str(mode)
    depth = data["depths"]
    if data["processed_images"] is not None and (data["processed_images"].shape != (4, 480, 640, 3) or data["processed_images"].dtype != np.uint8):
        raise ValueError("MoGe source RGB must be original-resolution uint8")
    if depth.shape != (4, 480, 640):
        raise ValueError("frozen MoGe benchmark requires four original 640x480 outputs")
    if data["camera_points"].shape != depth.shape + (3,) or data["normals"].shape != depth.shape + (3,) or data["masks"].shape != depth.shape:
        raise ValueError("MoGe prediction shape mismatch")
    if data["masks"].dtype != np.bool_:
        raise ValueError("MoGe validity masks must be boolean official output masks")
    for key in ["intrinsics", "intrinsics_normalized"]:
        k = data[key]
        if k.shape != (4, 3, 3) or not np.isfinite(k).all() or (k[:, [0, 1], [0, 1]] <= 0).any() or not np.allclose(k[:, 2], [0, 0, 1]) or not np.allclose(k[:, [0, 1], [1, 0]], 0):
            raise ValueError("invalid MoGe intrinsics")
    converted = data["intrinsics_normalized"].copy()
    converted[:, 0] *= 640
    converted[:, 1] *= 480
    converted[:, :2, 2] -= .5
    if not np.allclose(converted, data["intrinsics"], rtol=1e-6, atol=1e-4):
        raise ValueError("normalized MoGe intrinsics do not match integer-centre pixel conversion")
    valid = data["masks"] & np.isfinite(depth) & (depth > 0) & np.isfinite(data["camera_points"]).all(-1)
    if not np.allclose(data["camera_points"][..., 2][valid], depth[valid], rtol=1e-5, atol=1e-5):
        raise ValueError("MoGe point z differs from camera-z depth")
    data["valid"] = valid
    return data


def world_points(camera_points, extrinsics):
    return np.array([(p - e[:, 3]) @ e[:, :3] for p, e in zip(camera_points, extrinsics)])


def evaluate_moge(fixture, prediction, output, worker_receipt, include_anchor=True):
    fixture, output = Path(fixture), Path(output)
    data = load_moge(prediction)
    receipt = json.loads(Path(worker_receipt).read_text(encoding="utf-8-sig"))
    if receipt.get("artifactSha256") != hashlib.sha256(Path(prediction).read_bytes()).hexdigest():
        raise ValueError("MoGe worker receipt artifact hash mismatch")
    conditioning = receipt.get("conditioning")
    if not isinstance(conditioning, dict) or conditioning.get("mode") != data["conditioning_mode"]:
        raise ValueError("MoGe receipt conditioning mode missing or differs from NPZ")
    if conditioning["mode"] == "none":
        if set(conditioning) != {"mode"}:
            raise ValueError("unconditioned MoGe receipt contains unexpected conditioning inputs")
    else:
        fovs = conditioning.get("horizontalFovDegrees")
        if conditioning.get("provenance") != "experiment-oracle" or not isinstance(fovs, list) or len(fovs) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 1 < value < 179 for value in fovs):
            raise ValueError("known-FOV MoGe receipt requires four ordered finite oracle horizontal FOV values")
        output_fov = np.degrees(2 * np.arctan(640 / (2 * data["intrinsics"][:, 0, 0])))
        if not np.allclose(output_fov, fovs, rtol=0, atol=.001):
            raise ValueError("supplied MoGe FOV values differ from output intrinsic FOV")
    truth = truth_geometry(fixture, 640, 480)
    metadata = truth["metadata"]
    if [item.get("sourceIndex") for item in receipt.get("inputs", [])] != list(range(4)):
        raise ValueError("MoGe worker source indices must preserve four input positions")
    if [item["sha256"] for item in receipt.get("inputs", [])] != [item["sha256"] for item in metadata["input_hashes"]]:
        raise ValueError("MoGe worker source order/hash mismatch")
    coordinates = receipt.get("coordinates", {})
    if coordinates.get("units") != "meters" or coordinates.get("cameraConvention") != "opencv" or coordinates.get("axes") != "x-right,y-down,z-forward" or coordinates.get("depth") != "camera-z" or coordinates.get("worldFrame") != "independent-camera-frames":
        raise ValueError("MoGe primary evaluation requires explicit metric OpenCV independent camera points")
    if coordinates.get("normal") != "camera-facing signed unit vectors":
        raise ValueError("MoGe signed normal evaluation requires the declared camera-facing convention")
    if receipt.get("parameters", {}).get("forceProjection") is not True or receipt.get("parameters", {}).get("applyMask") is not True:
        raise ValueError("frozen MoGe evaluation requires official projection-constrained points and applied mask")
    valid, depths = data["valid"], data["depths"]
    extrinsics = truth["extrinsics"]
    cloud = world_points(data["camera_points"], extrinsics)
    normal_world = np.array([normal @ e[:, :3] for normal, e in zip(data["normals"], extrinsics)])
    membership = surface_membership(truth["cloud"], truth["labels"], truth["normals"], metadata)
    opening_roi = np.isin(truth["labels"], metadata["evaluation"]["fireplace_boundary_roi"]["labels"])
    for _ in range(metadata["evaluation"]["fireplace_boundary_roi"]["margin_prediction_pixels"]):
        opening_roi = dilate(opening_roi)
    report = dict(kind="synthetic monocular metric point benchmark; oracle extrinsics only in evaluation, no camera reconstruction",
                  fixture=metadata["version"], prediction=str(Path(prediction).resolve()), prediction_sha256=receipt["artifactSha256"],
                  fixture_geometry_sha256=hashlib.sha256((fixture / "ground-truth/geometry.json").read_bytes()).hexdigest(),
                  input_hashes=metadata["input_hashes"], conditioning=receipt.get("conditioning", {}), worker_identity=receipt.get("identity"),
                  evaluation_pose_provenance="exact frozen synthetic oracle world-to-camera E used after inference only; not independently solved model cameras",
                  alignment="none; no camera fit, ICP, deformation or per-view/per-object scale",
                  comparability="MoGe native640x480 versus DA3 native252x196: identical metric formulas and original per-grid ROI/1px boundary tolerance; different physical sampling/tolerance means F1 and sample-weighted metrics are not strictly equivalent. No resampling to flatter a model.",
                  valid_fraction=float(valid.mean()), invalid_fraction=float(1 - valid.mean()),
                  validity_policy="official binary mask, finite positive depth and finite points; no confidence cutoff, tail trimming or missing filling",
                  depth={}, boundaries={}, held_out_dimensions={}, cross_view={}, surfaces={}, per_view={},
                  scale_sources=dict(NATIVE="MoGe learned metric scale; no camera-pose scale or benchmark anchor injected"),
                  unverified=["real photography", "independent camera poses", "hidden surfaces", "semantics", "fusion", "mesh", "client integration", "installation truth"])
    geometries = {"NATIVE": (depths, cloud)}
    if include_anchor:
        try:
            scale, inferred_width = anchor_scale(np.where(valid, depths, np.nan), data["intrinsics"], metadata, 1, 1)
            geometries["ANCHOR-ALIGNED"] = (depths * scale, world_points(data["camera_points"] * scale, extrinsics))
            report["scale_anchor"] = dict(known_m=metadata["anchor"]["known_width_m"], inferred_native_m=inferred_width, fixed_scale=scale,
                                        interpretation="original frozen one-anchor depth/K contract; scales direct camera points about each fixed oracle camera origin; no pose fit")
            report["scale_sources"]["ANCHOR-ALIGNED"] = "both native learned metric scale and existing 1.20m post-inference benchmark anchor"
        except ValueError as error:
            report["scale_anchor"] = dict(unavailable_reason=str(error), interpretation="invalid anchor prevents optional derivative; native evaluation retained")
    output.mkdir(parents=True, exist_ok=True)
    for lane, (lane_depth, lane_cloud) in geometries.items():
        report["depth"][lane] = depth_metrics(lane_depth, truth["depths"], valid)
        report["boundaries"][lane] = boundary_metrics(lane_depth, truth["depths"], valid)
        report["boundaries"][f"fireplace_opening_{lane}"] = boundary_metrics(lane_depth, truth["depths"], valid, opening_roi)
        report["held_out_dimensions"][lane] = held_out_dimensions(lane_cloud, truth["cloud"], truth["labels"], valid, metadata)
        report["cross_view"][lane] = cross_view_metrics(lane_cloud, truth["cloud"], truth["depths"], truth["labels"], valid, truth["intrinsics"], extrinsics, metadata["dimensions"])
        surfaces, residual, angles = surface_metrics(lane_cloud, valid, membership, normal_world)
        report["surfaces"][lane] = surfaces
        report["per_view"][lane] = [dict(view_index=index, valid_fraction=float(valid[index].mean()), invalid_fraction=float(1 - valid[index].mean()), depth=depth_metrics(lane_depth[index], truth["depths"][index], valid[index])) for index in range(4)]
        np.savez_compressed(output / f"{lane.lower()}.npz", points=lane_cloud.astype(np.float32), valid=valid)
        rgb = data["processed_images"] if data["processed_images"] is not None else truth["colors"]
        for index in range(4):
            png(output / f"{lane}-depth-{index + 1:02}.png", depth_image(lane_depth[index]), f"SYNTHETIC MOGE {lane} {index + 1:02}")
            source = (index + 1) % 4
            selected = valid[source]
            png(output / f"{lane}-cross-view-{source + 1:02}-to-{index + 1:02}.png", project_image(lane_cloud[source][selected], rgb[source][selected], truth["intrinsics"][index], extrinsics[index], 640, 480), f"SYNTHETIC MOGE {lane} VIEW {source + 1:02} TO {index + 1:02}")
            plane_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            plane_valid = np.isfinite(residual[index])
            intensity = np.clip(residual[index][plane_valid] / .25, -1, 1)
            plane_rgb[plane_valid] = np.column_stack([np.maximum(intensity, 0), 1 - np.abs(intensity), np.maximum(-intensity, 0)]) * 255
            png(output / f"{lane}-plane-residual-{index + 1:02}.png", plane_rgb, f"SYNTHETIC PLANE RESIDUAL {index + 1:02}")
            if lane == "NATIVE":
                normal_rgb = np.clip((data["normals"][index] + 1) * 127.5, 0, 255).astype(np.uint8)
                normal_rgb[~valid[index]] = 0
                png(output / f"NATIVE-normal-{index + 1:02}.png", normal_rgb, f"SYNTHETIC MOGE NORMAL {index + 1:02}")
                png(output / f"GT-depth-{index + 1:02}.png", depth_image(truth["depths"][index]), f"SYNTHETIC GT DEPTH {index + 1:02}")
    fov = np.degrees(2 * np.arctan(640 / (2 * data["intrinsics"][:, 0, 0])))
    truth_fov = np.degrees(2 * np.arctan(640 / (2 * truth["intrinsics"][:, 0, 0])))
    supplied = receipt.get("conditioning", {}).get("mode") == "known-fov"
    report["focal"] = dict(truth_pixels=truth["intrinsics"][:, [0, 1], [0, 1]].tolist(), output_pixels=data["intrinsics"][:, [0, 1], [0, 1]].tolist(),
                          output_fov_x_degrees=fov.tolist(), truth_fov_x_degrees=truth_fov.tolist(),
                          independent_accuracy_claim=not supplied, principal_points=data["intrinsics"][:, :2, 2].tolist(),
                          interpretation="supplied oracle FOV, not model camera accuracy" if supplied else "predicted monocular FOV; centered principal point is an upstream assumption",
                          absolute_error_pixels=None if supplied else np.abs(data["intrinsics"][:, [0, 1], [0, 1]] - truth["intrinsics"][:, [0, 1], [0, 1]]).tolist(),
                          absolute_fov_error_degrees=None if supplied else np.abs(fov - truth_fov).tolist())
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def supplement_da3_planes(fixture, prediction, metric_report, scale_path, output):
    data = load_prediction(prediction)
    report = json.loads(Path(metric_report).read_text(encoding="utf-8-sig"))
    report = report.get("metrics", report)
    digest = hashlib.sha256(Path(prediction).read_bytes()).hexdigest()
    if digest != report["prediction_sha256"]:
        raise ValueError("DA3 supplemental source hash mismatch")
    height, width = data["depths"].shape[1:]
    truth = truth_geometry(fixture, width, height)
    cloud = np.array([points(d, k, e) for d, k, e in zip(data["depths"], data["intrinsics"], data["extrinsics"])])
    if scale_path == "ALIGNED":
        cloud = cloud * report["scale_anchor"]["fixed_scale"] @ np.array(report["alignment"]["rotation"]).T + np.array(report["alignment"]["translation_m"])
    elif scale_path != "NATIVE" or report.get("conditioning", {}).get("mode") != "pose":
        raise ValueError("DA3 native supplement requires preserved supplied-camera metric report")
    valid = np.isfinite(data["depths"]) & (data["depths"] > 0) & np.isfinite(data["confidences"]) & (data["confidences"] >= 0)
    membership = surface_membership(truth["cloud"], truth["labels"], truth["normals"], truth["metadata"])
    surfaces, _, _ = surface_metrics(cloud, valid, membership)
    result = dict(kind="additive DA3 known-plane evaluation; original receipt and arrays unchanged", source_prediction_sha256=digest,
                  scale_path=scale_path, resolution=[width, height], surfaces=surfaces,
                  comparability="same analytic plane and 1px GT-only membership rule at each native resolution; sample counts/resolution differ from MoGe 640x480; no normals supplied by DA3")
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=".image-blaster/benchmark/synthetic-room-v1")
    parser.add_argument("--prediction")
    parser.add_argument("--worker-receipt")
    parser.add_argument("--out")
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--da3-prediction")
    parser.add_argument("--da3-report")
    parser.add_argument("--da3-scale-path", choices=["ALIGNED", "NATIVE"])
    parser.add_argument("--da3-out")
    arguments = parser.parse_args()
    if arguments.da3_prediction:
        if not all([arguments.da3_report, arguments.da3_scale_path, arguments.da3_out]):
            parser.error("DA3 supplemental evaluation requires --da3-report, --da3-scale-path and --da3-out")
        supplement_da3_planes(arguments.fixture, arguments.da3_prediction, arguments.da3_report, arguments.da3_scale_path, arguments.da3_out)
    if not arguments.prediction:
        if not arguments.da3_prediction:
            parser.error("provide --prediction for MoGe or --da3-prediction for additive DA3 plane evaluation")
        raise SystemExit(0)
    if not all([arguments.worker_receipt, arguments.out]):
        parser.error("MoGe evaluation requires --worker-receipt and --out")
    report = evaluate_moge(arguments.fixture, arguments.prediction, arguments.out, arguments.worker_receipt, not arguments.no_anchor)
    print(json.dumps(dict(depth=report["depth"], output=str(Path(arguments.out).resolve())), indent=2))
