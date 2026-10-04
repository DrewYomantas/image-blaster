import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fixture import depth_image, intersect, png, rays


def camera_centres(extrinsics):
    return -np.einsum("nji,nj->ni", extrinsics[:, :, :3], extrinsics[:, :, 3])


def points(depth, intrinsics, extrinsics):
    origin, directions = rays(intrinsics, extrinsics, depth.shape[1], depth.shape[0])
    return origin + directions * depth[..., None]


def bilinear(depth, coordinates):
    values = []
    for x, y in coordinates:
        if not (0 <= x < depth.shape[1] - 1 and 0 <= y < depth.shape[0] - 1):
            raise ValueError("opening scale anchor lies outside prediction image")
        left, top = int(x), int(y)
        dx, dy = x - left, y - top
        patch = depth[top:top + 2, left:left + 2]
        if not np.isfinite(patch).all() or (patch <= 0).any():
            raise ValueError("opening scale anchor has invalid depth")
        values.append(patch[0, 0] * (1 - dx) * (1 - dy) + patch[0, 1] * dx * (1 - dy) + patch[1, 0] * (1 - dx) * dy + patch[1, 1] * dx * dy)
    return np.array(values)


def anchor_scale(depths, intrinsics, metadata, sx, sy):
    anchor = metadata["anchor"]
    coordinates = (np.array(anchor["pixels"]) + .5) * [sx, sy] - .5
    view = anchor["view_index"]
    samples = bilinear(depths[view], coordinates)
    local = np.column_stack([coordinates, np.ones(2)]) @ np.linalg.inv(intrinsics[view]).T * samples[:, None]
    span = np.linalg.norm(local[1] - local[0])
    inferred_width = span / anchor["fraction_of_opening_width"]
    if inferred_width <= 1e-8:
        raise ValueError("degenerate opening scale anchor")
    return anchor["known_width_m"] / inferred_width, float(inferred_width)


def rigid_alignment(predicted_centres, truth_centres):
    predicted_mean, truth_mean = predicted_centres.mean(0), truth_centres.mean(0)
    covariance = (predicted_centres - predicted_mean).T @ (truth_centres - truth_mean)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] < 1e-10:
        raise ValueError("camera trajectory cannot determine rigid alignment: collinear/coincident centres")
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = truth_mean - rotation @ predicted_mean
    return rotation, translation


def depth_metrics(prediction, truth, valid):
    if not valid.any():
        return dict(samples=0, abs_rel=None, rmse_m=None, scale_invariant_log_rmse=None)
    predicted, target = prediction[valid], truth[valid]
    log_error = np.log(predicted) - np.log(target)
    return dict(samples=int(valid.sum()), abs_rel=float(np.mean(np.abs(predicted - target) / target)),
                rmse_m=float(np.sqrt(np.mean((predicted - target) ** 2))),
                scale_invariant_log_rmse=float(np.sqrt(np.maximum(0, np.mean(log_error ** 2) - np.mean(log_error) ** 2))))


def boundaries(depths, valid, threshold=.05):
    result = np.zeros(depths.shape, dtype=bool)
    horizontal = (np.abs(depths[:, :, 1:] - depths[:, :, :-1]) > threshold) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (np.abs(depths[:, 1:, :] - depths[:, :-1, :]) > threshold) & valid[:, 1:, :] & valid[:, :-1, :]
    result[:, :, 1:] |= horizontal
    result[:, :, :-1] |= horizontal
    result[:, 1:, :] |= vertical
    result[:, :-1, :] |= vertical
    return result


def dilate(mask):
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)))
    return np.logical_or.reduce([padded[:, y:y + mask.shape[1], x:x + mask.shape[2]] for y in range(3) for x in range(3)])


def boundary_metrics(prediction, truth, valid, roi=None):
    predicted = boundaries(prediction, valid)
    target = boundaries(truth, np.isfinite(truth) & (truth > 0))
    if roi is not None:
        predicted &= roi
        target &= roi
    precision = float((predicted & dilate(target)).sum() / predicted.sum()) if predicted.any() else 0.0
    recall = float((target & dilate(predicted)).sum() / target.sum()) if target.any() else 0.0
    return dict(precision=precision, recall=recall, f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                threshold_m=.05, tolerance_pixels=1, predicted_pixels=int(predicted.sum()), truth_pixels=int(target.sum()))


def pose_metrics(extrinsics, truth_extrinsics, rotation, translation, scale):
    original_centres, truth_centres = camera_centres(extrinsics), camera_centres(truth_extrinsics)
    aligned_centres = original_centres * scale @ rotation.T + translation
    aligned_rotations = extrinsics[:, :, :3] @ rotation.T
    relative = np.einsum("nij,nkj->nik", aligned_rotations, truth_extrinsics[:, :, :3])
    degrees = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    baseline = []
    for first in range(len(extrinsics)):
        for second in range(first + 1, len(extrinsics)):
            target = np.linalg.norm(truth_centres[first] - truth_centres[second])
            original = np.linalg.norm(original_centres[first] - original_centres[second])
            aligned = original * scale
            baseline.append(dict(views=[first, second], truth_m=float(target), inference_units=float(original), aligned_m=float(aligned),
                                 aligned_abs_cm=float(abs(aligned - target) * 100), aligned_error_percent=float(abs(aligned - target) / target * 100)))
    return dict(aligned_rotation_error_degrees=degrees.tolist(), aligned_translation_error_cm=(np.linalg.norm(aligned_centres - truth_centres, axis=1) * 100).tolist(),
                camera_baselines=baseline, original_world_to_camera=extrinsics.tolist(), aligned_world_to_camera=np.concatenate([aligned_rotations, (-np.einsum("nij,nj->ni", aligned_rotations, aligned_centres))[:, :, None]], 2).tolist())


def dimensions_metrics(cloud, truth_cloud, labels, valid, metadata):
    result = []
    for item in metadata["dimensions"]:
        truth_mask = np.isin(labels, item["labels"])
        mask = truth_mask & valid
        target = np.array(item["dimensions_m"])
        observed_truth = np.ptp(truth_cloud[truth_mask], axis=0) if truth_mask.any() else None
        observed = np.ptp(cloud[mask], axis=0) if mask.any() else None
        axes = []
        for axis, name in enumerate(["width", "height", "depth"]):
            value = None if observed is None else float(observed[axis])
            actual = float(target[axis])
            coverage = None if observed_truth is None or actual == 0 else float(observed_truth[axis] / actual)
            axes.append(dict(axis=name, truth_m=actual, aligned_observed_extent_m=value,
                             absolute_error_cm=None if value is None else abs(value - actual) * 100,
                             relative_error_percent=None if value is None or actual == 0 else abs(value - actual) / actual * 100,
                             truth_visible_extent_m=None if observed_truth is None else float(observed_truth[axis]),
                             truth_visibility_fraction=coverage, complete_axis_observed=coverage is not None and .97 <= coverage <= 1.03))
        result.append(dict(id=item["id"], samples=int(mask.sum()), estimator="evaluation-only semantic point bounds; incomplete extents are lower bounds, not hidden geometry", axes=axes))
    return result


def held_out_dimensions(cloud, truth_cloud, labels, valid, metadata):
    result = dimensions_metrics(cloud, truth_cloud, labels, valid, metadata)
    for item in result:
        truth_count = int(np.isin(labels, next(d["labels"] for d in metadata["dimensions"] if d["id"] == item["id"])).sum())
        item["predicted_valid_fraction_of_visible_samples"] = item["samples"] / truth_count if truth_count else 0.
        for axis in item["axes"]:
            excluded = item["id"].startswith("opening-") and axis["axis"] == "width"
            value, visible = axis["aligned_observed_extent_m"], axis["truth_visible_extent_m"]
            planar = axis["truth_m"] == 0
            target = 0. if planar else axis["truth_m"] if axis["complete_axis_observed"] else visible
            axis["observed_extent_m"] = value
            axis["independent_accuracy_claim"] = not excluded and value is not None and target is not None and target > 0
            axis["calibration_dimension_excluded"] = excluded
            axis["planarity_diagnostic"] = planar
            axis["evaluation_target"] = "planarity diagnostic; zero authored extent" if planar else "authored full dimension" if axis["complete_axis_observed"] else "visible truth span only; hidden extent unverified"
            axis["held_out_target_m"] = target
            axis["held_out_absolute_error_cm"] = None if excluded or value is None or target is None else abs(value - target) * 100
            axis["held_out_relative_error_percent"] = None if excluded or value is None or not target else abs(value - target) / target * 100
    return result


def project_coordinates(cloud, intrinsics, extrinsics):
    local = cloud @ extrinsics[:, :3].T + extrinsics[:, 3]
    projected = local @ intrinsics.T
    coordinates = np.full(projected.shape[:-1] + (2,), np.nan)
    positive = np.isfinite(projected).all(-1) & (local[..., 2] > 0)
    coordinates[positive] = projected[positive, :2] / projected[positive, 2:]
    return coordinates, local[..., 2]


def cross_view_metrics(cloud, truth_cloud, truth_depths, labels, valid, truth_intrinsics, truth_extrinsics, dimension_items=None):
    count, height, width = truth_depths.shape
    pairs, reprojections, scatters = [], [], []
    groups = dimension_items if dimension_items is not None else [dict(id=str(label), labels=[int(label)]) for label in np.unique(labels)]
    entity_totals = {item["id"]: dict(labels=item["labels"], truth_count=0, valid_count=0, projection_count=0,
                                    nonprojectable_count=0, reprojection_squared=0., scatter_squared=0.) for item in groups}
    for target in range(count):
        source = (target + 1) % count
        coordinates, target_z = project_coordinates(truth_cloud[source], truth_intrinsics[target], truth_extrinsics[target])
        inside = np.isfinite(coordinates).all(-1) & (coordinates[..., 0] >= 0) & (coordinates[..., 0] < width - 1) & (coordinates[..., 1] >= 0) & (coordinates[..., 1] < height - 1)
        yy, xx = np.nonzero(inside)
        xy = coordinates[yy, xx]
        left, top = xy.astype(int).T
        dx, dy = (xy - np.column_stack([left, top])).T
        weights = np.column_stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
        tx = np.column_stack([left, left + 1, left, left + 1])
        ty = np.column_stack([top, top, top + 1, top + 1])
        samples = truth_depths[target][ty, tx]
        target_truth_depth = np.sum(samples * weights, axis=1)
        target_labels = labels[target][ty, tx]
        visible = (np.abs(target_truth_depth - target_z[yy, xx]) <= .02) & (np.ptp(samples, axis=1) <= .05) & (target_labels == labels[source][yy, xx, None]).all(1)
        yy, xx, xy, weights, tx, ty = [array[visible] for array in (yy, xx, xy, weights, tx, ty)]
        available = valid[source][yy, xx] & valid[target][ty, tx].all(1)
        predicted_xy, _ = project_coordinates(cloud[source][yy, xx], truth_intrinsics[target], truth_extrinsics[target])
        projection_valid = available & np.isfinite(predicted_xy).all(1)
        reprojection = np.linalg.norm(predicted_xy[projection_valid] - xy[projection_valid], axis=1)
        target_points = np.sum(cloud[target][ty[available], tx[available]] * weights[available, :, None], axis=1)
        difference = cloud[source][yy[available], xx[available]] - target_points
        scatter = np.linalg.norm(difference, axis=1)
        mean_offset = difference.mean(0) if len(difference) else None
        pairs.append(dict(source_view=source, target_view=target, truth_visible_correspondences=len(yy), valid_prediction_correspondences=int(available.sum()),
                          prediction_coverage=float(available.mean()) if len(available) else 0., nonprojectable_prediction_correspondences=int((available & ~projection_valid).sum()),
                          reprojection_rmse_pixels=float(np.sqrt(np.mean(reprojection ** 2))) if len(reprojection) else None,
                          duplicate_surface_scatter_rmse_cm=float(np.sqrt(np.mean(scatter ** 2)) * 100) if len(scatter) else None,
                          relative_view_offset_cm=None if mean_offset is None else (mean_offset * 100).tolist(),
                          relative_view_residual_scatter_rmse_cm=float(np.sqrt(np.mean(np.sum((difference - mean_offset) ** 2, axis=1))) * 100) if len(difference) else None))
        source_labels = labels[source][yy, xx]
        for entity in entity_totals.values():
            selected = np.isin(source_labels, entity["labels"])
            entity["truth_count"] += int(selected.sum())
            entity["valid_count"] += int((selected & available).sum())
            entity["projection_count"] += int((selected & projection_valid).sum())
            entity["nonprojectable_count"] += int((selected & available & ~projection_valid).sum())
            entity["reprojection_squared"] += float(np.sum(reprojection[selected[projection_valid]] ** 2))
            entity["scatter_squared"] += float(np.sum(scatter[selected[available]] ** 2))
        reprojections.extend(reprojection)
        scatters.extend(scatter)
    entity_metrics = {}
    for name, entity in entity_totals.items():
        entity_metrics[name] = dict(labels=entity["labels"], truth_visible_correspondences=entity["truth_count"],
                                    valid_prediction_correspondences=entity["valid_count"], nonprojectable_prediction_correspondences=entity["nonprojectable_count"],
                                    prediction_coverage=entity["valid_count"] / entity["truth_count"] if entity["truth_count"] else 0.,
                                    reprojection_rmse_pixels=float(np.sqrt(entity["reprojection_squared"] / entity["projection_count"])) if entity["projection_count"] else None,
                                    duplicate_surface_scatter_rmse_cm=float(np.sqrt(entity["scatter_squared"] / entity["valid_count"]) * 100) if entity["valid_count"] else None)
    total = sum(pair["truth_visible_correspondences"] for pair in pairs)
    usable = sum(pair["valid_prediction_correspondences"] for pair in pairs)
    return dict(entities=entity_metrics, method="fixed adjacent source-to-target pairs; GT-only bounds, 2cm visibility tolerance, same labels in target bilinear footprint and <=5cm truth depth footprint span; no prediction quality cutoff or missing filling; predicted points projected into fixed truth target camera", pairs=pairs,
                truth_visible_correspondences=total, valid_prediction_correspondences=usable, prediction_coverage=usable / total if total else 0.,
                nonprojectable_prediction_correspondences=sum(pair["nonprojectable_prediction_correspondences"] for pair in pairs),
                reprojection_rmse_pixels=float(np.sqrt(np.mean(np.square(reprojections)))) if len(reprojections) else None,
                duplicate_surface_scatter_rmse_cm=float(np.sqrt(np.mean(np.square(scatters))) * 100) if len(scatters) else None)


def conditioning_metadata(receipt, data, prediction):
    if receipt is None:
        if data.get("conditioning_mode") not in (None, "none"):
            raise ValueError("conditioned prediction requires explicit worker receipt")
        return dict(mode="none", depthScaleSource="nowhere", provenance="inferred-camera", cameraPredictionIndependent=True)
    payload = json.loads(Path(receipt).read_text(encoding="utf-8-sig"))
    if payload.get("artifactSha256") != hashlib.sha256(Path(prediction).read_bytes()).hexdigest():
        raise ValueError("worker receipt prediction artifact hash mismatch")
    conditioning = payload.get("conditioning", dict(mode="none", depthScaleSource="nowhere"))
    if conditioning.get("mode") not in ("none", "intrinsics-only", "pose"):
        raise ValueError("invalid receipt conditioning mode")
    if data.get("conditioning_mode") not in (None, conditioning["mode"]):
        raise ValueError("worker receipt conditioning mode differs from prediction")
    if conditioning["mode"] == "pose":
        if conditioning.get("provenance") != "experiment-oracle" or conditioning.get("cameraPredictionIndependent") is not False:
            raise ValueError("conditioned evaluation requires explicit oracle provenance")
        if payload.get("coordinates", {}).get("units") != "meters":
            raise ValueError("native conditioned evaluation requires metre camera coordinates")
        if not np.allclose(conditioning.get("processedIntrinsics"), data["intrinsics"], atol=1e-4):
            raise ValueError("conditioned processed intrinsics differ from output cameras")
    return conditioning


def project_image(cloud, colors, intrinsics, extrinsics, width, height):
    local = cloud @ extrinsics[:, :3].T + extrinsics[:, 3]
    projected = local @ intrinsics.T
    usable = np.isfinite(projected).all(1) & (local[:, 2] > 0)
    coordinates = np.zeros((len(cloud), 2), dtype=np.int64)
    coordinates[usable] = np.rint(projected[usable, :2] / projected[usable, 2:]).astype(np.int64)
    usable &= (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height)
    indices = np.flatnonzero(usable)
    indices = indices[np.argsort(local[indices, 2])[::-1]]
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    canvas[coordinates[indices, 1], coordinates[indices, 0]] = colors[indices]
    return canvas


def load_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ["depths", "confidences", "intrinsics", "extrinsics"]
        if any(key not in archive for key in required):
            raise ValueError("prediction NPZ requires depths, confidences, intrinsics, extrinsics")
        data = {key: np.asarray(archive[key], dtype=float) for key in required}
        data["processed_images"] = np.array(archive["processed_images"]) if "processed_images" in archive else None
        if "conditioning_mode" in archive:
            mode = np.asarray(archive["conditioning_mode"])
            if mode.shape != () or str(mode) not in ("none", "intrinsics-only", "pose"):
                raise ValueError("invalid prediction conditioning mode")
            data["conditioning_mode"] = str(mode)
    depth = data["depths"]
    if depth.ndim != 3 or depth.shape[0] != 4 or min(depth.shape[1:]) < 14:
        raise ValueError("fixture requires four depth maps, each at least 14x14")
    if data["confidences"].shape != depth.shape or data["intrinsics"].shape != (4, 3, 3) or data["extrinsics"].shape not in [(4, 3, 4), (4, 4, 4)]:
        raise ValueError("prediction shape mismatch")
    if data["extrinsics"].shape[-2] == 4:
        if not np.allclose(data["extrinsics"][:, 3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("extrinsics must be rigid homogeneous transforms")
        data["extrinsics"] = data["extrinsics"][:, :3]
    for key in ["intrinsics", "extrinsics"]:
        if not np.isfinite(data[key]).all():
            raise ValueError("nonfinite cameras")
    rotations = data["extrinsics"][:, :, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("extrinsics must use proper orthonormal rotations")
    if (data["intrinsics"][:, [0, 1], [0, 1]] <= 0).any() or not np.allclose(data["intrinsics"][:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError("invalid pinhole intrinsics")
    return data


def evaluate(fixture, prediction, output, worker_receipt=None):
    fixture, output = Path(fixture), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((fixture / "ground-truth/geometry.json").read_text())
    for item in metadata["input_hashes"]:
        digest = hashlib.sha256((fixture / "inputs" / item["file"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("fixture input hash mismatch")
    data = load_prediction(prediction)
    conditioning = conditioning_metadata(worker_receipt, data, prediction)
    pose_conditioned = conditioning["mode"] == "pose"
    depths, intrinsics, extrinsics = (data[key] for key in ["depths", "intrinsics", "extrinsics"])
    count, height, width = depths.shape
    sx, sy = width / metadata["resolution"][0], height / metadata["resolution"][1]
    truth_intrinsics = np.array(metadata["intrinsics"])
    truth_intrinsics[:, 0] *= sx
    truth_intrinsics[:, 1] *= sy
    truth_intrinsics[:, 0, 2] += (sx - 1) / 2
    truth_intrinsics[:, 1, 2] += (sy - 1) / 2
    truth_extrinsics = np.array(metadata["extrinsics"])
    truth_depths, labels, colors = [], [], []
    for k, e in zip(truth_intrinsics, truth_extrinsics):
        origin, direction = rays(k, e, width, height)
        depth, label, _, color = intersect(origin, direction, metadata["primitives"])
        truth_depths.append(depth)
        labels.append(label)
        colors.append((np.power(color, 1 / 2.2) * 255).astype(np.uint8))
    truth_depths, labels, colors = map(np.array, [truth_depths, labels, colors])
    valid = np.isfinite(depths) & (depths > 0) & np.isfinite(data["confidences"]) & (data["confidences"] >= 0)
    scale, inferred_width = anchor_scale(np.where(valid, depths, np.nan), intrinsics, metadata, sx, sy)
    rotation, translation = rigid_alignment(camera_centres(extrinsics) * scale, camera_centres(truth_extrinsics))
    cloud = np.array([points(d, k, e) for d, k, e in zip(depths, intrinsics, extrinsics)])
    aligned_cloud = cloud * scale @ rotation.T + translation
    truth_cloud = np.array([points(d, k, e) for d, k, e in zip(truth_depths, truth_intrinsics, truth_extrinsics)])
    focal_truth, focal_inference = truth_intrinsics[:, [0, 1], [0, 1]], intrinsics[:, [0, 1], [0, 1]]
    original_metrics = depth_metrics(depths, truth_depths, valid)
    original_metrics["gauge_dependent_numeric_rmse_vs_metre_truth"] = original_metrics.pop("rmse_m")
    original_metrics["interpretation"] = "unscaled relative units; AbsRel/RMSE versus metre truth are gauge-dependent diagnostic numbers, not physical metre accuracy"
    opening_roi = np.isin(labels, metadata["evaluation"]["fireplace_boundary_roi"]["labels"])
    for _ in range(metadata["evaluation"]["fireplace_boundary_roi"]["margin_prediction_pixels"]):
        opening_roi = dilate(opening_roi)
    report = dict(kind="synthetic geometry benchmark; evaluation-only GT never supplied to provider", fixture=metadata["version"], prediction=str(Path(prediction).resolve()),
                  prediction_sha256=hashlib.sha256(Path(prediction).read_bytes()).hexdigest(),
                  fixture_geometry_sha256=hashlib.sha256((fixture / "ground-truth/geometry.json").read_bytes()).hexdigest(),
                  input_hashes=metadata["input_hashes"], inference_frame="original provider relative units and gauge retained in prediction NPZ",
                  resized_truth=dict(width=width, height=height, scale_x=sx, scale_y=sy, crop=[0, 0], convention="pixel-centre resize; exact analytic re-raycast, no depth interpolation"),
                  valid_fraction=float(valid.mean()), confidence_policy=metadata["evaluation"]["confidence_policy"],
                  scale_anchor=dict(id="opening-width", known_m=1.2, inferred_original_units=inferred_width, fixed_scale=scale, evaluation_only_pixel_annotations=True),
                  alignment=dict(order="fixed single-anchor scale, then rigid camera-centre rotation/translation; no joint similarity fit", rotation=rotation.tolist(), translation_m=translation.tolist()),
                  depth=dict(INFERENCE=original_metrics, ALIGNED=depth_metrics(depths * scale, truth_depths, valid)),
                  boundaries=dict(ALIGNED=boundary_metrics(depths * scale, truth_depths, valid),
                                  fireplace_opening_ALIGNED=boundary_metrics(depths * scale, truth_depths, valid, opening_roi),
                                  fireplace_opening_roi=metadata["evaluation"]["fireplace_boundary_roi"]),
                  cameras=pose_metrics(extrinsics, truth_extrinsics, rotation, translation, scale),
                  focal=dict(truth_pixels=focal_truth.tolist(), inferred_pixels=focal_inference.tolist(), absolute_error_pixels=np.abs(focal_inference - focal_truth).tolist(), error_percent=(np.abs(focal_inference - focal_truth) / focal_truth * 100).tolist()),
                  dimensions=dimensions_metrics(aligned_cloud, truth_cloud, labels, valid, metadata),
                  unverified=["real photography", "semantic segmentation by provider", "hidden-surface completion", "mesh topology", "materials", "client import", "installation truth"])
    report["conditioning"] = conditioning
    report["scale_sources"] = dict(INFERENCE="DA3 supplied-camera path" if pose_conditioned else "nowhere", ALIGNED="both DA3 supplied-camera path and existing 1.20m benchmark anchor" if pose_conditioned else "existing 1.20m post-inference benchmark anchor")
    report["held_out_dimensions"] = dict(ALIGNED=held_out_dimensions(aligned_cloud, truth_cloud, labels, valid, metadata))
    report["cross_view"] = dict(ALIGNED=cross_view_metrics(aligned_cloud, truth_cloud, truth_depths, labels, valid, truth_intrinsics, truth_extrinsics, metadata["dimensions"]))
    if conditioning["mode"] == "intrinsics-only":
        report["kind"] = "synthetic intrinsics-only experimental control; supplied oracle calibration, no backbone pose conditioning"
    if pose_conditioned:
        report["kind"] = "synthetic camera-conditioning experiment; explicit oracle cameras supplied, depth inferred"
        report["inference_frame"] = "provider-native metres in supplied scene-y-up camera frame; raw decoder arrays retained in prediction NPZ"
        report["depth"]["NATIVE"] = depth_metrics(depths, truth_depths, valid)
        report["boundaries"]["fireplace_opening_NATIVE"] = boundary_metrics(depths, truth_depths, valid, opening_roi)
        report["held_out_dimensions"]["NATIVE"] = held_out_dimensions(cloud, truth_cloud, labels, valid, metadata)
        report["cross_view"]["NATIVE"] = cross_view_metrics(cloud, truth_cloud, truth_depths, labels, valid, truth_intrinsics, truth_extrinsics, metadata["dimensions"])
        report["scale_sources"]["NATIVE"] = "DA3 supplied-camera path; no benchmark anchor or rigid fit"
        report["native_alignment"] = "none: supplied camera frame and official DA3 metric scale retained; no rigid fit of perturbed cameras"
        report["cameras"]["independent_model_accuracy"] = False
        report["cameras"]["interpretation"] = "camera outputs are supplied oracle/test input, not independently reconstructed model achievements"
        report["depth"]["INFERENCE"]["rmse_m"] = report["depth"]["INFERENCE"].pop("gauge_dependent_numeric_rmse_vs_metre_truth")
        report["focal"]["interpretation"] = "output intrinsics are supplied processed oracle calibration, not independently inferred focal accuracy"
        report["depth"]["INFERENCE"]["interpretation"] = "provider-native metric conditioned inference; use NATIVE for metre accuracy"
        report["processed_intrinsics_against_frozen_truth"] = dict(processed=intrinsics.tolist(), frozen_evaluation=truth_intrinsics.tolist(), difference=(intrinsics-truth_intrinsics).tolist(), interpretation="frozen evaluator retains pixel-centre resize; official supplied-K processing can differ in principal point; neither path is silently adjusted")
    for item in report["dimensions"]:
        for axis in item["axes"]:
            value = axis["aligned_observed_extent_m"]
            axis["original_oriented_extent_meters" if pose_conditioned else "original_oriented_extent_relative_units"] = None if value is None else value / scale
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "aligned.npz", points=aligned_cloud.astype(np.float32), depths=(depths * scale).astype(np.float32), valid=valid, rotation=rotation, translation=translation, scale=scale)
    rgb = data["processed_images"]
    if rgb is None or rgb.shape != (count, height, width, 3):
        rgb = colors
    elif np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1:
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    for index in range(count):
        png(output / f"GT-depth-{index + 1:02}.png", depth_image(truth_depths[index]), f"SYNTHETIC GT DEPTH {index + 1:02}")
        png(output / f"INFERENCE-depth-{index + 1:02}.png", depth_image(depths[index]), f"SYNTHETIC INFERENCE DEPTH {index + 1:02}")
        png(output / f"ALIGNED-depth-{index + 1:02}.png", depth_image(depths[index] * scale), f"SYNTHETIC ALIGNED DEPTH {index + 1:02}")
        if pose_conditioned:
            png(output / f"NATIVE-depth-{index + 1:02}.png", depth_image(depths[index]), f"SYNTHETIC NATIVE DEPTH {index + 1:02}")
        # Source view differs from the target, exposing camera/depth consistency.
        source = (index + 1) % count
        selected = valid[source]
        if pose_conditioned:
            png(output / f"NATIVE-cross-view-{source + 1:02}-to-{index + 1:02}.png", project_image(cloud[source][selected], rgb[source][selected], truth_intrinsics[index], truth_extrinsics[index], width, height), f"SYNTHETIC NATIVE VIEW {source + 1:02} TO {index + 1:02}")
        png(output / f"ALIGNED-cross-view-{source + 1:02}-to-{index + 1:02}.png", project_image(aligned_cloud[source][selected], rgb[source][selected], truth_intrinsics[index], truth_extrinsics[index], width, height), f"SYNTHETIC ALIGNED VIEW {source + 1:02} TO {index + 1:02}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default=".image-blaster/benchmark/synthetic-room-v1")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker-receipt", help="explicit worker result required to interpret supplied-camera metric outputs")
    arguments = parser.parse_args()
    report = evaluate(arguments.fixture, arguments.prediction, arguments.out, arguments.worker_receipt)
    print(json.dumps(dict(scale_anchor=report["scale_anchor"], depth=report["depth"], output=str(Path(arguments.out).resolve())), indent=2))
