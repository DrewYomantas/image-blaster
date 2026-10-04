import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

GEOMETRY = Path(__file__).resolve().parents[1] / 'geometry'
sys.path.insert(0, str(GEOMETRY))
from evaluate_moge import truth_geometry, surface_membership

THRESHOLD_M = .05
SAMPLE_STEP_M = .025


def project(points, k, e):
    local = np.asarray(points) @ np.asarray(e)[:, :3].T + np.asarray(e)[:, 3]
    projected = local @ np.asarray(k).T
    valid = local[:, 2] > 1e-7
    xy = np.full((len(local), 2), np.nan)
    xy[valid] = projected[valid, :2] / projected[valid, 2:]
    return xy, local[:, 2]


def datum_translation(metadata):
    primitives = metadata['primitives']
    back = next(p for p in primitives if p['name'] == 'opening-back')
    wall = next(p for p in primitives if p['name'] == 'front-left')
    return np.array([(back['low'][0] + back['high'][0]) / 2, back['low'][1], wall['high'][2]])


def polygon_samples(vertices, step=SAMPLE_STEP_M):
    vertices = np.asarray(vertices, float)
    result = []
    for index in range(1, len(vertices) - 1):
        a, b, c = vertices[[0, index, index + 1]]
        count = max(1, int(np.ceil(max(np.linalg.norm(b-a), np.linalg.norm(c-a), np.linalg.norm(c-b)) / step)))
        for row in range(count + 1):
            u = row / count
            v = np.arange(count - row + 1) / count
            result.append(a + u * (b-a) + v[:, None] * (c-a))
    return np.concatenate(result) if result else np.empty((0, 3))


def plane_error(vertices, normal, plane):
    vertices, normal = np.asarray(vertices, float), np.asarray(normal, float)
    cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    length = np.linalg.norm(cross)
    if length < 1e-10:
        return dict(status='degenerate', orientation_error_degrees=None, signed_centroid_offset_m=None)
    direction = cross / length
    return dict(status='evaluated', signed_normal_cosine=float(direction @ normal), signed_normal_note='polygon winding only; fitter does not guarantee semantic outward normal', orientation_error_degrees=float(np.degrees(np.arccos(np.clip(abs(direction @ normal), 0, 1)))),
                signed_centroid_offset_m=float(vertices.mean(0) @ normal - plane),
                vertex_plane_rmse_m=float(np.sqrt(np.mean((vertices @ normal - plane)**2))))


def surface_vertices(surface, points):
    refs = surface.get('vertexIds', surface.get('pointIds', surface.get('vertices', surface.get('corners', []))))
    return np.array([points[v] if isinstance(v, str) else v for v in refs], float).reshape(-1, 3)


def visible_samples(samples, k, e, depths):
    xy, z = project(samples, k, e)
    height, width = depths.shape
    valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < width-.5) & (xy[:, 1] >= 0) & (xy[:, 1] < height-.5)
    indices = np.flatnonzero(valid)
    pixel = np.rint(xy[indices]).astype(int)
    visible = z[indices] <= depths[pixel[:, 1], pixel[:, 0]] + THRESHOLD_M
    indices = indices[visible]
    return indices, xy[indices], z[indices]



def axis_supported(names, entity, axis):
    if entity == 'opening-back' or (entity in ('room', 'opening-recess') and axis == 2):
        return False
    pairs = [('Left', 'Right'), ('Floor', 'Ceiling') if entity == 'room' else ('bottom', 'top'), ('Back', 'Front')]
    return all(any(token in name for name in names) for token in pairs[axis])


def named_truth_points(metadata):
    boxes = {item['name']: item for item in metadata['primitives']}
    opening, wall = boxes['opening-back'], boxes['front-left']
    points = {}
    for vertical, yi in [('bottom', 0), ('top', 1)]:
        for side, xi in [('Left', 0), ('Right', 1)]:
            points[f'opening.{vertical}{side}'] = [(opening['low'], opening['high'])[xi][0], (opening['low'], opening['high'])[yi][1], wall['high'][2]]
    for entity in ['hearth', 'mantel']:
        box = boxes[entity]
        for vertical, yi in [('bottom', 0), ('top', 1)]:
            for depth, zi in [('Back', 0), ('Front', 1)]:
                for side, xi in [('Left', 0), ('Right', 1)]:
                    points[f'{entity}.{vertical}{depth}{side}'] = [(box['low'], box['high'])[xi][0], (box['low'], box['high'])[yi][1], (box['low'], box['high'])[zi][2]]
    for vertical, y in [('Floor', boxes['floor']['high'][1]), ('Ceiling', boxes['ceiling']['low'][1])]:
        for side, x in [('Left', boxes['left-wall']['high'][0]), ('Right', boxes['right-wall']['low'][0])]:
            points[f'room.back{vertical}{side}'] = [x, y, wall['high'][2]]
    return points



def expected_semantic_surface(face):
    primitive, axis, sign = face['primitive'], face['axis'], face['sign']
    room = {('floor', 1, 1): 'room.floor', ('ceiling', 1, -1): 'room.ceiling',
            ('left-wall', 0, 1): 'room.left', ('right-wall', 0, -1): 'room.right',
            ('rear-wall', 2, -1): 'room.rear'}
    if (primitive, axis, sign) in room:
        return room[(primitive, axis, sign)]
    if primitive.startswith('front-') and axis == 2 and sign == 1:
        return 'room.back'
    side = {(0,-1): 'left', (0,1): 'right', (1,-1): 'bottom', (1,1): 'top', (2,-1): 'back', (2,1): 'front'}[(axis,sign)]
    return primitive + '.' + side


def semantic_report(report, structure, original_sha256):
    result = copy.deepcopy(report)
    result['version'] = 'structural-evaluation-v1.1'
    result['report_history'] = dict(original_file='report.json', original_version=report['version'], original_sha256=original_sha256,
                                    correction='Add semantic completeness; retain every prior numerical metric and legacy geometric missing boolean unchanged. No fit, registration, rescoring or overlay rewrite.')
    present = {surface['id'] for surface in structure.get('surfaces', []) if surface['id'] != 'opening.mouth' and len(surface.get('vertexIds', surface.get('vertices', []))) >= 3}
    for face in result['surfaces']:
        semantic = expected_semantic_surface(face)
        face['expected_semantic_surface_id'] = semantic
        face['semantic_missing'] = semantic not in present
        face['geometric_contact_within_5cm'] = not face['missing']
        face['missing_field_note'] = 'legacy missing means zero geometric contact only; semantic_missing determines whether corresponding surface was proposed'
    expected = sorted({face['expected_semantic_surface_id'] for face in result['surfaces']})
    result['semantic_completeness'] = dict(expected_visible_surface_ids=expected, proposed_surface_ids=sorted(present),
                                         missing_surface_ids=[key for key in expected if key not in present],
                                         unsupported_proposal_ids=sorted(present-set(expected)),
                                         policy='Exact declared semantic identity, independent of proximity to another surface. Presence alone does not establish geometric accuracy or extent coverage.')
    return result


def upgrade_semantics(structure_path, output):
    output = Path(output)
    target = output / 'report-v1.1.json'
    if target.exists():
        raise ValueError('refusing to overwrite semantic report revision')
    original = (output/'report.json').read_bytes()
    report = json.loads(original)
    structure = json.loads(Path(structure_path).read_text(encoding='utf-8-sig'))
    if hashlib.sha256(Path(structure_path).read_bytes()).hexdigest() != report['structure_sha256']:
        raise ValueError('structure differs from the first evaluated artifact')
    result = semantic_report(report, structure, hashlib.sha256(original).hexdigest())
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def evaluate(fixture, structure_path, output):
    fixture, output = Path(fixture), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'report.json'
    if result_path.exists():
        raise ValueError('refusing to overwrite preserved first evaluation; use another output directory')
    structure = json.loads(Path(structure_path).read_text(encoding='utf-8-sig'))
    coordinates = structure.get('coordinates', {})
    if coordinates.get('units') != 'meters' or coordinates.get('frame') != 'scene-y-up' or coordinates.get('handedness') != 'right' or coordinates.get('openingBottomMidpoint') != [0, 0, 0]:
        raise ValueError('undeclared or incompatible fitted datum; no automatic registration allowed')
    truth = truth_geometry(fixture, 640, 480)
    metadata = truth['metadata']
    translation = datum_translation(metadata)
    raw_points = structure.get('points', {})
    points = {key: np.asarray(value, float) + translation for key, value in raw_points.items()}
    targets = named_truth_points(metadata)
    point_scores = [dict(id=key, error_m=float(np.linalg.norm(point-np.array(targets[key]))) if key in targets else None, status='datum-constrained' if key in ('opening.bottomLeft','opening.bottomRight') else 'evaluated' if key in targets else 'unresolved-semantic-datum') for key, point in points.items()]
    face_ids, eligible, faces = surface_membership(truth['cloud'], truth['labels'], truth['normals'], metadata)
    polygons = []
    wire_polygons = []
    for surface in structure.get('surfaces', []):
        vertices = surface_vertices(surface, points)
        if len(vertices) >= 3 and np.isfinite(vertices).all():
            wire_polygons.append((surface, vertices))
            if surface.get('id') != 'opening.mouth':
                polygons.append((surface, vertices, polygon_samples(vertices)))
    all_samples = np.concatenate([s[2] for s in polygons]) if polygons else np.empty((0, 3))
    tree = cKDTree(all_samples) if len(all_samples) else None
    truth_tree = cKDTree(truth['cloud'][np.isfinite(truth['cloud']).all(-1)])
    proposal_surfaces = []
    for surface, vertices, samples in polygons:
        distance = truth_tree.query(samples)[0]
        proposal_surfaces.append(dict(id=surface['id'], sampled_points=len(samples), nearest_visible_truth_median_m=float(np.median(distance)), nearest_visible_truth_p95_m=float(np.percentile(distance,95)), unsupported_by_visible_truth_fraction=float(np.mean(distance > THRESHOLD_M)), note='Distance to all visible truth; hidden regions remain unverified, not asserted false.'))
    face_results = []
    for identity, face in faces.items():
        selected = face_ids == identity
        cloud = truth['cloud'][selected]
        distance = tree.query(cloud)[0] if tree is not None else np.full(len(cloud), np.inf)
        normal = np.array(face['normal'])
        plane = face['plane_m'] * face['sign']
        candidates = []
        for surface, vertices, samples in polygons:
            errors = plane_error(vertices, normal, plane)
            # Match solely for evaluation; fitted outputs are never changed.
            near = float(cKDTree(samples).query(cloud)[0].mean())
            candidates.append((near, surface['id'], errors))
        best = min(candidates, key=lambda x: x[0]) if candidates else None
        face_results.append(dict(**face, truth_visible_samples=len(cloud), within_5cm_fraction=float(np.mean(distance <= THRESHOLD_M)),
                                 finite_extent_mean_distance_m=float(np.mean(distance)) if tree else None, finite_extent_median_distance_m=float(np.median(distance)) if tree else None, finite_extent_p95_distance_m=float(np.percentile(distance, 95)) if tree else None,
                                 missing=not np.any(distance <= THRESHOLD_M), closest_surface_id=best[1] if best else None,
                                 nearest_proposal_plane=best[2] if best else None))
    views = []
    for index, (k, e) in enumerate(zip(truth['intrinsics'], truth['extrinsics'])):
        cloud = truth['cloud'][index]
        valid = np.isfinite(cloud).all(-1) & (truth['labels'][index] > 0)
        distance = tree.query(cloud[valid])[0] if tree else np.full(valid.sum(), np.inf)
        indices, projected, depth = visible_samples(all_samples, k, e, truth['depths'][index])
        pixel = np.rint(projected).astype(int)
        depth_residual = depth - truth['depths'][index][pixel[:, 1], pixel[:, 0]]
        architecture = valid & np.isin(truth['labels'][index], list(range(1, 11)))
        architectural_distance = tree.query(cloud[architecture])[0] if tree else np.full(architecture.sum(), np.inf)
        boundary = np.zeros(valid.shape, bool)
        labels = truth['labels'][index]
        boundary[1:-1,1:-1] = ((labels[1:-1,1:-1] != labels[:-2,1:-1]) | (labels[1:-1,1:-1] != labels[2:,1:-1]) | (labels[1:-1,1:-1] != labels[1:-1,:-2]) | (labels[1:-1,1:-1] != labels[1:-1,2:]))
        boundary &= architecture
        truth_edges = np.argwhere(boundary)[:, ::-1]
        edge_pixels = []
        for surface, vertices in wire_polygons:
            samples = np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(a-b)/.01))+1)) for a,b in zip(vertices,np.roll(vertices,-1,axis=0))])
            _, xy, _ = visible_samples(samples,k,e,truth['depths'][index])
            edge_pixels.extend(xy.tolist())
        if edge_pixels and len(truth_edges):
            forward = cKDTree(truth_edges).query(edge_pixels)[0]
            backward = cKDTree(edge_pixels).query(truth_edges)[0]
            edge_agreement = dict(proposal_to_truth_median_px=float(np.median(forward)),proposal_to_truth_p95_px=float(np.percentile(forward,95)),truth_to_proposal_median_px=float(np.median(backward)),truth_to_proposal_p95_px=float(np.percentile(backward,95)), policy='oracle projected wire edges versus all architectural label boundaries including occlusion; diagnostic only')
        else:
            edge_agreement = dict(status='unavailable', policy='no projectable model edges or truth boundaries')
        view = dict(view=index + 1, oracle_boundary_agreement=edge_agreement, role='held-out' if index == 3 else 'source', camera_status='oracle diagnostic only; not estimated-camera validation',
                    truth_samples=int(valid.sum()), all_entity_coverage_within_5cm=float(np.mean(distance <= THRESHOLD_M)),
                    architectural_coverage_within_5cm=float(np.mean(architectural_distance <= THRESHOLD_M)),
                    visible_proposal_samples=len(indices), oracle_depth_rmse_m=float(np.sqrt(np.mean(depth_residual**2))) if len(indices) else None,
                    false_visible_proposal_fraction=float(np.mean(np.abs(depth_residual) > THRESHOLD_M)) if len(indices) else None)
        views.append(view)
        image = Image.open(fixture / 'inputs' / metadata['input_hashes'][index]['file']).convert('RGB')
        draw = ImageDraw.Draw(image)
        for surface, vertices in wire_polygons:
            xy, z = project(vertices, k, e)
            if np.isfinite(xy).all() and (z > 0).all():
                color = '#ffb347' if structure.get('status') != 'provisional-fit' or surface.get('uncertainty') is not None else '#00ffff'
                draw.line([tuple(p) for p in xy] + [tuple(xy[0])], fill=color, width=2)
        draw.rectangle((0, 0, 639, 43), fill='black')
        draw.text((6, 4), f'ORACLE CAMERA DIAGNOSTIC | view {index+1} | {view["role"]}', fill='white')
        draw.text((6, 18), 'cyan: inferred faces; orange: uncertain; unoutlined: missing/unsupported', fill='white')
        draw.text((6, 30), 'Wireframe includes occluded edges. This does not validate camera solving.', fill='white')
        image.save(output / f'overlay-{index+1:02}.png')
    camera_scores = []
    for index, camera in enumerate(structure.get('cameras', [])):
        if index >= 3:
            raise ValueError('held-out camera unexpectedly present in fit output')
        estimated = np.array(camera['worldToCamera'], float)
        k = np.array(camera['intrinsics'], float)
        expected = truth['extrinsics'][index]
        center = -estimated[:, :3].T @ estimated[:, 3] + translation
        target_center = -expected[:, :3].T @ expected[:, 3]
        delta = estimated[:, :3] @ expected[:, :3].T
        camera_scores.append(dict(id=camera['id'], center_error_m=float(np.linalg.norm(center-target_center)), rotation_error_degrees=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))), focal_x_error_percent=float(100*abs(k[0,0]-truth['intrinsics'][index,0,0])/truth['intrinsics'][index,0,0]), policy='estimated cameras scored, never replaced'))
    entities = []
    for entity in metadata['dimensions']:
        selected = np.isin(truth['labels'], entity['labels']) & np.isfinite(truth['cloud']).all(-1)
        cloud = truth['cloud'][selected]
        relevant_names = [key for key in points if (key.startswith('room.') if entity['id'] == 'room' else key.startswith('opening.') if entity['id'].startswith('opening-') else key.startswith(entity['id'] + '.'))]
        relevant = [p for key, p in points.items() if (key.startswith('room.') if entity['id'] == 'room' else key.startswith('opening.') if entity['id'].startswith('opening-') else key.startswith(entity['id'] + '.'))]
        span = np.ptp(relevant, axis=0) if relevant else [None]*3
        visible_span = np.ptp(cloud, axis=0) if len(cloud) else [None]*3
        axes = []
        for axis, name in enumerate(['width', 'height', 'depth']):
            unsupported = not relevant or (entity['id'] in ('room', 'opening-recess') and axis == 2) or entity['id'] == 'opening-back'
            if not axis_supported(relevant_names, entity['id'], axis):
                unsupported = True
            value = None if unsupported else float(span[axis])
            excluded = entity['id'].startswith('opening-') and axis == 0
            target = entity['dimensions_m'][axis]
            axes.append(dict(axis=name, estimated_m=value, truth_m=target, truth_visible_span_m=None if visible_span[axis] is None else float(visible_span[axis]),
                             status='unsupported' if unsupported else 'constraint' if excluded else 'inferred',
                             independent_absolute_error_m=None if value is None or excluded else abs(value-target)))
        entities.append(dict(id=entity['id'], labels=entity['labels'], axes=axes))
    report = dict(version='structural-evaluation-v1', structure_sha256=hashlib.sha256(Path(structure_path).read_bytes()).hexdigest(),
                  fixture=metadata['version'], alignment=dict(rotation=np.eye(3).tolist(), translation_m=translation.tolist(), scale=1,
                  policy='declared anchor datum axes; opening mouth bottom midpoint; one global rigid transform; no optimization'),
                  evaluation_policy=dict(threshold_m=THRESHOLD_M, triangle_sampling_spacing_m=SAMPLE_STEP_M, orientation='unoriented polygon plane; signs not geometry accuracy',
                  coverage='all visible truth pixels, no fitted inlier exclusions', oracle='post-fit diagnostics only, saved cameras unchanged'),
                  points=point_scores, source_camera_accuracy=camera_scores, surfaces=face_results, proposal_surfaces=proposal_surfaces, views=views, entities=entities,
                  geometry_inspection=dict(points=len(points), polygons=len(polygons), sampled_points=len(all_samples),
                  finite=all(np.isfinite(v).all() for v in points.values()), closed_mesh_required=False),
                  unresolved=structure.get('unknowns', []), held_out_estimated_camera_reprojection=dict(status='unavailable', gate_pass=False, reason='No independent view4 camera estimate; oracle diagnostics cannot satisfy gate'))
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture')
    parser.add_argument('--semantic-upgrade', action='store_true')
    parser.add_argument('--structure', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.semantic_upgrade:
        result = upgrade_semantics(args.structure, args.out)
        print(json.dumps(dict(version=result['version'], semantic_completeness=result['semantic_completeness']),indent=2))
    else:
        if not args.fixture:
            parser.error('--fixture required for scoring')
        result = evaluate(args.fixture, args.structure, args.out)
        print(json.dumps(dict(fixture=result['fixture'], views=result['views'], inspection=result['geometry_inspection']), indent=2))
